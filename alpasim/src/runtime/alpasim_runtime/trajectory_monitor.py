# SPDX-License-Identifier: Apache-2.0

"""Monitor 2: trajectory-based, checked live during the simulation.

Each time Alpamayo returns a trajectory, check whether EGO, if it follows that
trajectory, would come too close to another actor. This is checked at three
moments: now, waypoint 1 (+1 s) and waypoint 2 (+2 s). At each moment, EGO's
zone is compared with every actor's zone; an overlap is an alert.

A zone is the vehicle's box plus the area ahead of it: (speed x 1 s) long,
widening by 1/2 x 3 m/s^2 x (1 s)^2 = 1.5 m on each side. A stopped vehicle's
zone is just its box. Parked cars outside the driving lanes and actors behind
EGO are ignored.

EGO's future positions come from Alpamayo's trajectory. The other actors'
future positions come from the recorded drive, which AlpaSim replays and sends
once at the start of the run.

The monitor is added to AlpaSim's list of log receivers in event_loop.py. It
only writes alerts; it never changes what the car does.

Output, next to rollout.asl, one line per trajectory:
    trajectory_alerts.jsonl
    {"time_s": 15.1, "now": [], "waypoint 1": [{"id": "81", "label": "car", "gap_m": 3.4}],
     "waypoint 2": [...]}

    label   what the actor is (car, truck, pedestrian, ...)
    gap_m   distance between EGO and the actor at that moment, in meters
    null means that moment is after the end of the simulation, so it was not checked.
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

MOMENTS = {"now": 0.0, "waypoint 1": 1.0, "waypoint 2": 2.0}  # seconds after now
ZONE_SECONDS = 1.0  # length of the area ahead in each zone: speed x 1 s
SIDE_ACCEL = 3.0  # how fast a vehicle can move sideways, m/s^2
STOPPED_SPEED = 0.1  # m/s; below this a vehicle's zone is just its box


class TrajectoryMonitor:
    def __init__(self, rollout_dir, vector_map, end_us):
        self.alerts_path = os.path.join(rollout_dir, "trajectory_alerts.jsonl")
        self.lanes = lanes_from_map(vector_map)
        self.end_us = end_us  # last simulated time
        self.sizes = {}  # actor id -> (length, width)
        self.labels = {}  # actor id -> what the actor is (car, truck, ...)
        self.parked = set()  # ids of actors that never move
        self.recorded = {}  # actor id -> Track of its recorded drive
        self.rig_to_box = (0.0, 0.0)  # offset from where Alpamayo plans to EGO's center
        self.warmup_us = 0
        self.start_us = None
        self.now_us = None  # time of the trajectory Alpamayo is working on

    async def on_message(self, entry):
        """Called by AlpaSim for every log entry. Errors are logged, never raised."""
        try:
            self.handle(entry)
        except Exception:
            logger.exception("TrajectoryMonitor failed")

    def handle(self, entry):
        kind = entry.WhichOneof("log_entry")

        if kind == "rollout_metadata":  # at the start: sizes, labels and warm-up length
            metadata = entry.rollout_metadata
            for actor in metadata.actor_definitions.actor_aabb:
                self.sizes[actor.actor_id] = (actor.aabb.size_x, actor.aabb.size_y)
                self.labels[actor.actor_id] = actor.actor_label
            offset = metadata.transform_ego_coords_rig_to_aabb.vec
            self.rig_to_box = (offset.x, offset.y)
            self.warmup_us = metadata.force_gt_duration

        elif kind == "traffic_session_request":  # at the start: the recorded drive
            for actor in entry.traffic_session_request.logged_object_trajectories:
                if actor.object_id == "EGO":
                    continue
                if actor.is_static:
                    self.parked.add(actor.object_id)
                track = Track()
                for point in actor.trajectory.poses:
                    track.add(point.timestamp_us, to_xy_heading(point.pose))
                self.recorded[actor.object_id] = track

        elif kind == "actor_poses" and self.start_us is None:  # the first time step
            self.start_us = entry.actor_poses.timestamp_us

        elif kind == "driver_request":  # Alpamayo is asked for a trajectory
            self.now_us = entry.driver_request.time_now_us

        elif kind == "driver_return":  # Alpamayo returned it: check now
            self.check(self.now_us, entry.driver_return.trajectory)

    def check(self, now, trajectory):
        if now - self.start_us < self.warmup_us:
            return  # warm-up: EGO is still following the recording

        ego = self.ego_track(trajectory)

        row = {"time_s": round((now - self.start_us) / 1e6, 2)}
        for moment, seconds in MOMENTS.items():
            t = now + int(seconds * 1e6)
            if t > self.end_us:
                row[moment] = None  # after the end of the simulation
                continue
            alerts = []
            if ego.at(t) is not None:
                ego_zone = zone(ego, t, *self.sizes["EGO"])
                ego_outline = actor_box(*ego.at(t), *self.sizes["EGO"])
                for actor_id, track in sorted(self.recorded.items()):
                    if track.at(now) is None or track.at(t) is None:
                        continue  # the actor isn't in the scene at one of these times
                    if self.is_ignored(actor_id, track.at(now), ego.at(now)):
                        continue
                    if ego_zone.intersects(zone(track, t, *self.sizes[actor_id])):
                        outline = actor_box(*track.at(t), *self.sizes[actor_id])
                        alerts.append(
                            {
                                "id": actor_id,
                                "label": self.labels[actor_id],
                                "gap_m": round(ego_outline.distance(outline), 1),
                            }
                        )
            row[moment] = alerts

        with open(self.alerts_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        for moment in MOMENTS:
            for alert in row[moment] or []:
                logger.info(
                    "TrajectoryMonitor: %.1f s ALERT (%s) %s",
                    row["time_s"],
                    moment,
                    alert,
                )

    def ego_track(self, trajectory):
        """Alpamayo's trajectory as a Track of EGO's center (Alpamayo plans the
        position of the car's rig, which is offset from the center)."""
        track = Track()
        dx, dy = self.rig_to_box
        for point in trajectory.poses:
            x, y, heading = to_xy_heading(point.pose)
            center_x = x + dx * math.cos(heading) - dy * math.sin(heading)
            center_y = y + dx * math.sin(heading) + dy * math.cos(heading)
            track.add(point.timestamp_us, (center_x, center_y, heading))
        return track

    def is_ignored(self, actor_id, pose, ego):
        """A parked car outside the driving lanes, or an actor behind EGO's rear bumper."""
        if actor_id in self.parked:
            center = actor_box(*pose, *self.sizes[actor_id]).centroid
            if self.lanes is None or not self.lanes.contains(center):
                return True
        dx, dy = pose[0] - ego[0], pose[1] - ego[1]
        distance_ahead = dx * math.cos(ego[2]) + dy * math.sin(ego[2])
        return distance_ahead < -self.sizes["EGO"][0] / 2


class Track:
    """A vehicle's poses over time. at(t) gives the pose at time t (blending the
    two nearest poses), or None if t is outside the recorded times."""

    def __init__(self):
        self.times = []
        self.poses = []

    def add(self, t, pose):
        self.times.append(t)
        self.poses.append(pose)

    def at(self, t):
        if not self.times or t < self.times[0] or t > self.times[-1]:
            return None
        i = bisect.bisect_left(self.times, t)
        if self.times[i] == t:
            return self.poses[i]
        (x0, y0, h0), (x1, y1, h1) = self.poses[i - 1], self.poses[i]
        f = (t - self.times[i - 1]) / (self.times[i] - self.times[i - 1])
        turn = math.atan2(math.sin(h1 - h0), math.cos(h1 - h0))  # shortest way around
        return x0 + f * (x1 - x0), y0 + f * (y1 - y0), h0 + f * turn


def zone(track, t, length, width):
    """A vehicle's zone at time t: its box plus the area ahead (just the box if stopped)."""
    x, y, heading = track.at(t)
    body = actor_box(x, y, heading, length, width)
    speed = speed_at(track, t)
    if speed <= STOPPED_SPEED:
        return body
    return unary_union(
        [body, area_ahead(x, y, heading, speed, length, width, ZONE_SECONDS)]
    )


def speed_at(track, t):
    """Distance moved in the 0.1 s before t (or after t, at the start), divided by 0.1 s."""
    here, before = track.at(t), track.at(t - 100_000)
    if before is None:
        before, here = here, track.at(t + 100_000)
        if here is None:
            return 0.0
    return math.dist(here[:2], before[:2]) / 0.1


def actor_box(x, y, heading, length, width):
    """The actor's outline: a length x width rectangle at (x, y), facing `heading`."""
    shape = box(-length / 2, -width / 2, length / 2, width / 2)
    return translate(rotate(shape, heading, origin=(0, 0), use_radians=True), x, y)


def area_ahead(x, y, heading, speed, length, width, seconds):
    """Area ahead of a vehicle's front bumper: (speed x seconds) long, widening
    by 1/2 x SIDE_ACCEL x seconds^2 on each side."""
    reach = speed * seconds
    widen = min(0.5 * SIDE_ACCEL * seconds**2, reach)
    front, side = length / 2, width / 2
    corners = [
        (front, side),
        (front + reach, side + widen),
        (front + reach, -side - widen),
        (front, -side),
    ]
    return translate(
        rotate(Polygon(corners), heading, origin=(0, 0), use_radians=True), x, y
    )


def lanes_from_map(vector_map):
    """All driving lanes from AlpaSim's scene map, combined into one shape."""
    try:
        from trajdata.maps.vec_map_elements import MapElementType

        lanes = []
        for lane in vector_map.elements[MapElementType.ROAD_LANE].values():
            if lane.left_edge is not None and lane.right_edge is not None:
                left = [tuple(p[:2]) for p in lane.left_edge.xy]
                right = [tuple(p[:2]) for p in lane.right_edge.xy]
                # buffer(0) repairs outlines that cross themselves
                lanes.append(Polygon(left + right[::-1]).buffer(0))
        return unary_union(lanes)
    except Exception:
        logger.warning(
            "TrajectoryMonitor: no lanes in the scene map; all parked cars are ignored"
        )
        return None


def to_xy_heading(pose):
    """(x, y, heading) from a log pose. The heading is the car's direction on the map."""
    q = pose.quat
    heading = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
    return pose.vec.x, pose.vec.y, heading
