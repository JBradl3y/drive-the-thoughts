"""Monitor 1: the two-second rule, live during the simulation.

Every time Alpamayo returns a trajectory (every 0.1 s, after the warm-up): is any
actor inside the area EGO could reach in the next 2 s? The area starts at EGO's
front bumper, is speed x 2 s long and widens by 1/2 x 3 m/s^2 x (2 s)^2 = 6 m on
each side. EGO and the actors are where they actually are now.

Actors that don't count:
  - parked cars outside the driving lanes
  - actors behind EGO's rear bumper

How it gets the data: AlpaSim sends every log entry to a list of receivers while it
runs (one of them writes rollout.asl). This monitor is one more receiver on that
list, added in event_loop.py. It only watches and writes alerts; it never changes
what the car does, and an error in it is logged without stopping the simulation.

Alerts are saved next to rollout.asl as each trajectory is checked:
    two_second_alerts.jsonl    {"time_s": ..., "alerts": [actor ids]}, one line per trajectory
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
SECONDS = 2.0             # length of the area: speed x 2 s
SIDE_ACCEL = 3.0          # steering acceleration, m/s^2 (how fast the area widens)
STOPPED = 0.1             # m/s; slower than this = not moving (no area)
IGNORE_PARKED = True      # parked cars count only inside a driving lane
IGNORE_BEHIND = True      # actors behind EGO's rear bumper don't count
ALERTS_FILE = "two_second_alerts.jsonl"


class TwoSecondMonitor:
    def __init__(self, rollout_dir, vector_map=None):
        self.alerts_path = os.path.join(rollout_dir, ALERTS_FILE)
        self.lanes = lanes_from_map(vector_map)
        self.sizes = {}                  # actor id -> (length, width, label, is_static)
        self.ego_size = None
        self.warmup_us = 0
        self.start_us = None
        self.tracks = {}                 # actor id -> Track of where it has been
        self.pending_now = []            # times of the plans Alpamayo has been asked for
        self.latest_alerts = set()       # the most recent check's alerts
        if self.lanes is None:
            logger.warning("TwoSecondMonitor: no lanes in the scene map; every parked car "
                           "will count as outside the lanes")

    async def on_message(self, entry):
        """AlpaSim calls this for every log entry it sends."""
        try:
            self._handle(entry)
        except Exception:
            logger.exception("TwoSecondMonitor failed on a %s entry", entry.WhichOneof("log_entry"))

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
            self.warmup_us = md.force_gt_duration
        elif kind == "traffic_session_request":          # which actors are parked, once at the start
            for o in entry.traffic_session_request.logged_object_trajectories:
                if o.object_id != "EGO":
                    L, W, label, _ = self.sizes.get(o.object_id, (o.aabb.size_x, o.aabb.size_y, "unknown", False))
                    self.sizes[o.object_id] = (L, W, label, o.is_static)
        elif kind == "actor_poses":                       # where everyone is, every 0.1 s
            t = entry.actor_poses.timestamp_us
            if self.start_us is None or t < self.start_us:
                self.start_us = t
            for ap in entry.actor_poses.actor_poses:
                self.tracks.setdefault(ap.actor_id, Track()).add(t, x_y_direction(ap.actor_pose))
        elif kind == "driver_request":
            self.pending_now.append(entry.driver_request.time_now_us)
        elif kind == "driver_return" and self.pending_now:  # Alpamayo just returned a trajectory
            self._check(self.pending_now.pop(0))

    def _check(self, now):
        if self.start_us is None or now - self.start_us < self.warmup_us:
            return                                        # warm-up: EGO follows the recorded drive
        ego = self.tracks.get("EGO")
        ego_pose = ego.at(now) if ego else None
        if ego_pose is None or self.ego_size is None:
            logger.warning("TwoSecondMonitor: no EGO position at t=%d us; trajectory skipped", now)
            return
        before = ego.at(now - 100_000) or ego_pose
        ego_speed = math.hypot(ego_pose[0] - before[0], ego_pose[1] - before[1]) / 0.1

        alerts = set()
        if ego_speed > STOPPED:
            area = cone(*ego_pose, ego_speed, *self.ego_size, SECONDS)
            for actor_id, track in self.tracks.items():
                if actor_id == "EGO":
                    continue
                pose = track.at(now)
                if pose is None:
                    continue
                length, width, _, is_static = self.sizes.get(actor_id, (4.5, 2.0, "unknown", False))
                if skipped(pose, (length, width), is_static, ego_pose, self.ego_size[0], self.lanes):
                    continue
                if area.intersects(object_box(*pose, length, width)):
                    alerts.add(actor_id)

        time_s = round((now - self.start_us) / 1e6, 2)
        self.latest_alerts = alerts
        os.makedirs(os.path.dirname(self.alerts_path), exist_ok=True)
        with open(self.alerts_path, "a") as f:
            f.write(json.dumps({"time_s": time_s, "alerts": sorted(alerts)}) + "\n")
        if alerts:
            logger.info("TwoSecondMonitor: %.1f s  ALERT  %s", time_s,
                        ", ".join(a.split("@")[0] for a in sorted(alerts)))


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
