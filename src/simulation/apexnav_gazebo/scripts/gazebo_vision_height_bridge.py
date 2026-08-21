#!/usr/bin/env python3
"""Feed Gazebo's local vertical pose to PX4 through MAVROS external vision.

The stock iris barometer can let its estimated bias absorb slow real vertical
motion.  This simulation-only bridge supplies an independent, launch-relative
height measurement to EKF2.  PX4 remains the sole position controller; the
bridge publishes measurements, never actuator or setpoint commands.
"""

import math

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State


class GazeboVisionHeightBridge:
    def __init__(self):
        rospy.init_node("apexnav_gazebo_vision_height_bridge")
        self.model_name = str(rospy.get_param("~gazebo_model_name", "iris_depth_camera"))
        self.world_frame = str(rospy.get_param("~world_frame", "map"))
        self.max_rate = float(rospy.get_param("~publish_rate", 30.0))
        if not self.model_name or not self.world_frame or not math.isfinite(self.max_rate) or self.max_rate <= 0.0:
            raise ValueError("invalid Gazebo vision-height bridge configuration")

        self.fcu = State()
        self.origin = None
        self.last_clock = rospy.Time(0)
        self.last_publish = rospy.Time(0)
        self.publisher = rospy.Publisher(
            "/mavros/vision_pose/pose", PoseStamped, queue_size=5)
        rospy.Subscriber("/mavros/state", State, self._state_cb, queue_size=5)
        rospy.Subscriber("/gazebo/model_states", ModelStates,
                         self._models_cb, queue_size=5)

    def _state_cb(self, msg):
        self.fcu = msg

    @staticmethod
    def _finite_pose(pose):
        p, q = pose.position, pose.orientation
        values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        if not all(math.isfinite(value) for value in values):
            return False
        qnorm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        return 0.95 <= qnorm <= 1.05

    @staticmethod
    def relative_position(pose, origin):
        """Return launch-relative ENU position; kept pure for regression tests."""
        return (pose.position.x - origin[0],
                pose.position.y - origin[1],
                pose.position.z - origin[2])

    def _reset_epoch(self, now):
        self.origin = None
        self.last_publish = rospy.Time(0)
        self.last_clock = now

    def _models_cb(self, msg):
        now = rospy.Time.now()
        if now.is_zero():
            return
        if not self.last_clock.is_zero() and now < self.last_clock:
            rospy.logwarn("Gazebo vision-height bridge clock reset; recapturing launch origin")
            self._reset_epoch(now)
        self.last_clock = now

        try:
            index = msg.name.index(self.model_name)
        except ValueError:
            return
        if index >= len(msg.pose):
            return
        pose = msg.pose[index]
        if not self._finite_pose(pose):
            rospy.logwarn_throttle(2.0, "Rejecting non-finite Gazebo vision pose")
            return

        # Continuously recapture the launch origin while disarmed.  Freeze it
        # at arm so the measurement becomes true AGL through the whole flight.
        if self.origin is None or not self.fcu.armed:
            self.origin = (pose.position.x, pose.position.y, pose.position.z)
        period = 1.0 / self.max_rate
        if (not self.last_publish.is_zero() and
                0.0 <= (now - self.last_publish).to_sec() < period):
            return

        x, y, z = self.relative_position(pose, self.origin)
        out = PoseStamped()
        out.header.stamp = now
        out.header.frame_id = self.world_frame
        out.pose.position.x = x
        out.pose.position.y = y
        out.pose.position.z = z
        # EKF2_EV_CTRL is restricted to vertical position, nevertheless send a
        # valid normalized orientation so the MAVROS message is well formed.
        out.pose.orientation = pose.orientation
        self.publisher.publish(out)
        self.last_publish = now


if __name__ == "__main__":
    GazeboVisionHeightBridge()
    rospy.spin()
