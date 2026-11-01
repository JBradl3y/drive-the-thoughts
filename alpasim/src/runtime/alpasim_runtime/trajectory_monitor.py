"""Monitor 2: trajectory-based, live during the simulation.

Every time Alpamayo returns a trajectory (every 0.1 s, after the warm-up): if EGO
follows it, does EGO's zone overlap an actor's zone at the same moment? Checked at
now, waypoint 1 (+1 s) and waypoint 2 (+2 s); each moment is its own version of the
monitor. EGO's position comes from the trajectory. The actors' positions come from
the recorded drive that AlpaSim replays (sent once at the start of the run), so
their positions 1 s and 2 s ahead are known.

A zone is the vehicle's box + a cone ahead of it, speed x 1 s long, widening by
1/2 x 3 m/s^2 x (1 s)^2 = 1.5 m on each side (just the box if it isn't moving).

Actors that don't count:
  - parked cars outside the driving lanes
  - actors behind EGO's rear bumper

How it gets the data: AlpaSim sends every log entry to a list of receivers while it
runs (one of them writes rollout.asl). This monitor is one more receiver on that
list, added in event_loop.py. It only watches and writes alerts; it never changes
what the car does, and an error in it is logged without stopping the simulation.

Alerts are saved next to rollout.asl as each trajectory is checked:
    trajectory_alerts.jsonl
    {"time_s": ..., "now": [ids], "waypoint 1": [ids], "waypoint 2": [ids]}, one line per
    trajectory. null = that moment is past the end of the simulation, so it wasn't checked.
"""

import bisect
import json
import logging
import math
import os

from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

# ---- Settings ----
HORIZON = 2.0             # seconds of the trajectory to check
N_WAYPOINTS = 2           # waypoint 1 at 1 s, waypoint 2 at 2 s
STEP = HORIZON / N_WAYPOINTS    # each zone covers the 1 s until the next moment
SIDE_ACCEL = 3.0          # steering acceleration, m/s^2 (how fast a cone widens)
STOPPED = 0.1             # m/s; slower than this = not moving (box only)
IGNORE_PARKED = True      # parked cars count only inside a driving lane
IGNORE_BEHIND = True      # actors behind EGO's rear bumper don't count
ALERTS_FILE = "trajectory_alerts.jsonl"


def moment_label(k):
    return "now" if k == 0 else f"waypoint {k}"


class TrajectoryMonitor:
    def __init__(self, rollout_dir, vector_map=None, end_us=None):
        self.alerts_path = os.path.join(rollout_dir, ALERTS_FILE)
        self.lanes = lanes_from_map(vector_map)
        self.end_us = end_us             # last simulated time
        self.sizes = {}                  # actor id -> (length, width, label, is_static)
        self.ego_size = None
        self.rig_to_box = (0.0, 0.0)     # Alpamayo plans the rig; zones use EGO's box centre
        self.warmup_us = 0
        self.start_us = None
        self.tracks = {}                 # actor id -> Track of where it has been (from AlpaSim)
        self.recorded = {}               # actor id -> Track of the recorded drive (start to end)
        self.pending_now = []            # times of the plans Alpamayo has been asked for
        self.latest_alerts = {}          # the most recent check's alerts
        if self.lanes is None:
            logger.warning("TrajectoryMonitor: no lanes in the scene map; every parked car "
                           "will count as outside the lanes")

    async def on_message(self, entry):
        """AlpaSim calls this for every log entry it sends."""
        try:
            self._handle(entry)
        except Exception:
            logger.exception("TrajectoryMonitor failed on a %s entry", entry.WhichOneof("log_entry"))

    def _handle(self, entry):
        kind = entry.WhichOneof("log_entry")
        if kind == "rollout_metadata":                    # sizes and warm-up length, once at the start
            md = entry.rollout_metadata
            for a in md.actor_definitions.actor_aabb:
                if a.actor_id == "EGO":
                    self.ego_size = (a.aabb.size_x, a.aabb.size_y)
                else:
                    static = self.sizes.get(a.actor_id, (0, 0, "", False))[3]
                    self.sizes[a.actor_id] = (a.aabb.size_x, a.aabb.size_y, a.actor_label, static)
            v = md.transform_ego_coords_rig_to_aabb.vec
            self.rig_to_box = (v.x, v.y)
            self.warmup_us = md.force_gt_duration
        elif kind == "traffic_session_request":          # parked or not + the recorded drive, once
            for o in entry.traffic_session_request.logged_object_trajectories:
                if o.object_id == "EGO":
                    continue
                L, W, label, _ = self.sizes.get(o.object_id, (o.aabb.size_x, o.aabb.size_y, "unknown", False))
                self.sizes[o.object_id] = (L, W, label, o.is_static)
                track = Track()
                for p in sorted(o.trajectory.poses, key=lambda p: p.timestamp_us):
                    track.add(p.timestamp_us, x_y_direction(p.pose))
                self.recorded[o.object_id] = track
        elif kind == "actor_poses":                       # where everyone is, every 0.1 s
            t = entry.actor_poses.timestamp_us
            if self.start_us is None or t < self.start_us:
                self.start_us = t
            for ap in entry.actor_poses.actor_poses:
                self.tracks.setdefault(ap.actor_id, Track()).add(t, x_y_direction(ap.actor_pose))
        elif kind == "driver_request":
            self.pending_now.append(entry.driver_request.time_now_us)
        elif kind == "driver_return" and self.pending_now:  # Alpamayo just returned a trajectory
            now = self.pending_now.pop(0)
            pts = [(p.timestamp_us, x_y_direction(p.pose)) for p in entry.driver_return.trajectory.poses]
            self._check(now, pts)

    def _actor_path(self, actor_id, now):
        """t (s after now) -> the actor's pose: where AlpaSim already put it, or else
        the recorded drive (which is where it will be: AlpaSim replays it)."""
        seen, recorded = self.tracks.get(actor_id), self.recorded.get(actor_id)

        def path(t_s):
            t = now + int(t_s * 1e6)
            pose = seen.at(t) if seen else None
            if pose is None and recorded:
                pose = recorded.at(t)
            return pose
        return path

    def _plan_path(self, pts, now):
        """Alpamayo's trajectory (rig poses) as t (s after now) -> EGO's box centre."""
        track = Track()
        tx, ty = self.rig_to_box
        for t, (x, y, d) in pts:
            track.add(t, (x + tx * math.cos(d) - ty * math.sin(d), y + tx * math.sin(d) + ty * math.cos(d), d))
        return lambda t_s: track.at(now + int(t_s * 1e6))

    def _end_us(self):
        if self.end_us is not None:
            return self.end_us
        ends = [t.times[-1] for t in self.recorded.values() if t.times]
        return max(ends) if ends else 0

    def _check(self, now, pts):
        if self.start_us is None or now - self.start_us < self.warmup_us:
            return                                        # warm-up: EGO follows the recorded drive
        if self.ego_size is None:
            logger.warning("TrajectoryMonitor: no EGO size yet at t=%d us; trajectory skipped", now)
            return
        ego_path = self._plan_path(pts, now)
        ego_len, ego_wid = self.ego_size
        actors = (set(self.tracks) | set(self.recorded)) - {"EGO"}
        others = {aid: self._actor_path(aid, now) for aid in actors}
        recorded_ahead_s = (self._end_us() - now) / 1e6

        result = {}
        for k in range(N_WAYPOINTS + 1):
            t = k * STEP
            if recorded_ahead_s < t - 1e-6:               # past the end of the simulation
                result[moment_label(k)] = None
                continue
            hits = set()
            if ego_path(t) is not None:
                mine = zone_at(ego_path, t, ego_len, ego_wid)
                for actor_id, path in others.items():
                    length, width, _, is_static = self.sizes.get(actor_id, (4.5, 2.0, "unknown", False))
                    pose_now = path(0)
                    if pose_now is None or path(t) is None:
                        continue
                    if skipped(pose_now, (length, width), is_static, ego_path(0), ego_len, self.lanes):
                        continue
                    if mine.intersects(zone_at(path, t, length, width)):     # same moment only
                        hits.add(actor_id)
            result[moment_label(k)] = hits

        time_s = round((now - self.start_us) / 1e6, 2)
        self.latest_alerts = result
        os.makedirs(os.path.dirname(self.alerts_path), exist_ok=True)
        with open(self.alerts_path, "a") as f:
            f.write(json.dumps({"time_s": time_s, **{k: (sorted(v) if v is not None else None)
                                                    for k, v in result.items()}}) + "\n")
        for name, ids in result.items():
            if ids:
                logger.info("TrajectoryMonitor: %.1f s  ALERT (%s)  %s", time_s, name,
                            ", ".join(a.split("@")[0] for a in sorted(ids)))


# ---- Shapes ----

def place(shape, x, y, direction):
    """Turn a shape to face `direction` and move it to (x, y)."""
    return translate(rotate(shape, direction, origin=(0, 0), use_radians=True), x, y)


def object_box(x, y, direction, length, width):
    return place(box(-length / 2, -width / 2, length / 2, width / 2), x, y, direction)


def cone(x, y, direction, speed, length, width, seconds):
    """The area ahead of a vehicle: from its front bumper, speed x seconds long,
    widening by 1/2 x SIDE_ACCEL x seconds^2 on each side."""
    reach = speed * seconds
    widen = min(0.5 * SIDE_ACCEL * seconds**2, reach)
    front, side = length / 2, width / 2
    shape = Polygon([(front, side), (front + reach, side + widen),
                     (front + reach, -side - widen), (front, -side)])
    return place(shape, x, y, direction)


def speed_at(path, t):
    """Distance moved in the last 0.1 s, / 0.1 s."""
    here, before = path(t), path(t - 0.1)
    if before is None:                      # no point 0.1 s earlier: use the next 0.1 s
        before, here = here, path(t + 0.1)
        if here is None:
            return 0.0
    return math.hypot(here[0] - before[0], here[1] - before[1]) / 0.1


def zone_at(path, t, length, width):
    """A vehicle's zone at time t: its box + its cone (just the box if it isn't moving)."""
    x, y, d = path(t)
    body = object_box(x, y, d, length, width)
    speed = speed_at(path, t)
    if speed <= STOPPED:
        return body
    return unary_union([body, cone(x, y, d, speed, length, width, STEP)])


# ---- Which actors count ----

def skipped(actor_pose, size, is_static, ego_pose, ego_length, lanes):
    """True if this actor doesn't count right now: a parked car outside the driving
    lanes, or an actor behind EGO's rear bumper."""
    if IGNORE_PARKED and is_static:
        centre = object_box(*actor_pose, *size).centroid
        if lanes is None or not lanes.contains(centre):
            return True
    if IGNORE_BEHIND:
        ex, ey, ed = ego_pose
        ahead = (actor_pose[0] - ex) * math.cos(ed) + (actor_pose[1] - ey) * math.sin(ed)
        if ahead < -ego_length / 2:
            return True
    return False


def lanes_from_map(vector_map):
    """The driving lanes as one area, from AlpaSim's scene map: each lane's left and
    right edges make one polygon."""
    if vector_map is None:
        return None
    try:
        from trajdata.maps.vec_map_elements import MapElementType
        lanes = vector_map.elements[MapElementType.ROAD_LANE].values()
    except Exception:
        return None
    polys = []
    for lane in lanes:
        if lane.left_edge is None or lane.right_edge is None:
            continue
        left = [tuple(p[:2]) for p in lane.left_edge.xy]
        right = [tuple(p[:2]) for p in lane.right_edge.xy]
        try:
            poly = Polygon(left + list(reversed(right)))
        except (ValueError, TypeError):
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty and poly.area > 0:
            polys.append(poly)
    return unary_union(polys) if polys else None


# ---- Positions over time ----

def x_y_direction(pose):
    q = pose.quat
    return pose.vec.x, pose.vec.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))


class Track:
    """(time_us, pose) pairs you can ask 'where at time t?' (in-between times are blended)."""

    def __init__(self):
        self.times, self.poses = [], []

    def add(self, t, pose):
        if self.times and t <= self.times[-1]:
            return
        self.times.append(t)
        self.poses.append(pose)

    def at(self, t):
        if not self.times or t < self.times[0] or t > self.times[-1]:
            return None
        i = bisect.bisect_left(self.times, t)
        if self.times[i] == t:
            return self.poses[i]
        a, b = self.poses[i - 1], self.poses[i]
        f = (t - self.times[i - 1]) / (self.times[i] - self.times[i - 1])
        dd = math.atan2(math.sin(b[2] - a[2]), math.cos(b[2] - a[2]))
        return a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * dd
