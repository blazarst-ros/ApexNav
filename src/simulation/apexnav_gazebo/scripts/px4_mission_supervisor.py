#!/usr/bin/env python3
"""Lifecycle and watchdog boundary between the 2-D planner and PX4 Offboard."""

import math

import rospy
import tf.transformations as tft
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from geometry_msgs.msg import PoseStamped, Twist
from mavros_msgs.msg import ExtendedState, PositionTarget, State
from mavros_msgs.srv import CommandBool, SetMode
from nav_msgs.msg import Odometry
from plan_env.msg import MissionState, SemanticObservation
from plan_env.srv import StartMission, StartMissionResponse
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, Int32, String
from std_srvs.srv import Trigger, TriggerResponse


class Supervisor:
    def __init__(self):
        rospy.init_node("apexnav_px4_mission_supervisor")
        self.rate_hz = float(rospy.get_param("~setpoint_rate", 30.0))
        self.prestream_sec = float(rospy.get_param("~prestream_sec", 2.0))
        self.altitude = float(rospy.get_param("~takeoff_altitude", 1.0))
        self.alt_tol = float(rospy.get_param("~altitude_tolerance", 0.10))
        self.alt_stable_sec = float(rospy.get_param("~altitude_stable_sec", 2.0))
        self.completion_hold = float(rospy.get_param("~completion_hold_sec", 2.0))
        self.fault_land_delay = float(rospy.get_param("~fault_land_delay_sec", 3.0))
        self.cmd_timeout = float(rospy.get_param("~command_timeout_sec", 0.30))
        self.planner_start_timeout = float(rospy.get_param("~planner_start_timeout_sec", 3.0))
        self.arm_transition_timeout = float(rospy.get_param("~arm_transition_timeout_sec", 10.0))
        self.odom_timeout = float(rospy.get_param("~odom_timeout_sec", 0.20))
        self.sensor_timeout = float(rospy.get_param("~sensor_timeout_sec", 0.50))
        self.semantic_timeout = float(rospy.get_param("~semantic_timeout_sec", 5.0))
        self.max_speed = float(rospy.get_param("~max_horizontal_speed", 0.25))
        self.max_yaw_rate = float(rospy.get_param("~max_yaw_rate", 0.40))

        self.state = MissionState.WAIT_FCU
        self.detail = "waiting for mission and FCU"
        self.target = ""
        self.requested = False
        self.fcu = State()
        self.extended = ExtendedState()
        self.odom = None
        self.cmd = Twist()
        self.last_odom = rospy.Time(0)
        self.last_cmd = rospy.Time(0)
        self.last_rgb = rospy.Time(0)
        self.last_depth = rospy.Time(0)
        self.last_vlm_health = rospy.Time(0)
        self.last_semantic = rospy.Time(0)
        self.semantic_target = ""
        self.state_since = rospy.Time.now()
        self.altitude_stable_since = None
        self.hold_pose = None
        self.last_service_attempt = rospy.Time(0)
        self.planner_started = False

        self.setpoint_pub = rospy.Publisher(
            "/mavros/setpoint_raw/local", PositionTarget, queue_size=10)
        self.state_pub = rospy.Publisher("/apexnav/mission/state", MissionState, queue_size=2, latch=True)
        self.label_pub = rospy.Publisher("/detector/label", String, queue_size=1, latch=True)
        self.trigger_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        self.stop_pub = rospy.Publisher("/traj_server/stop", Empty, queue_size=1)
        rospy.Subscriber("/mavros/state", State, self._fcu_cb, queue_size=5)
        rospy.Subscriber("/mavros/extended_state", ExtendedState, self._extended_cb, queue_size=5)
        rospy.Subscriber("/mavros/local_position/odom", Odometry, self._odom_cb, queue_size=20)
        rospy.Subscriber("/apexnav/planner/cmd_vel_raw", Twist, self._cmd_cb, queue_size=5)
        rospy.Subscriber("/apexnav/camera/rgb/image_raw", Image, self._rgb_cb, queue_size=2)
        rospy.Subscriber("/apexnav/camera/depth/image_raw", Image, self._depth_cb, queue_size=2)
        rospy.Subscriber("/apexnav/vlm/diagnostics", DiagnosticArray, self._vlm_cb, queue_size=2)
        rospy.Subscriber("/apexnav/vlm/semantic_observation", SemanticObservation,
                         self._semantic_cb, queue_size=3)
        rospy.Subscriber("/ros/expl_result", Int32, self._result_cb, queue_size=5)
        self.arm = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.set_mode = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        rospy.Service("/apexnav/mission/start", StartMission, self._start)
        rospy.Service("/apexnav/mission/stop", Trigger, self._stop)
        rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self._tick)
        rospy.Timer(rospy.Duration(0.2), self._publish_state)

    def _transition(self, new_state, detail):
        if new_state == self.state and detail == self.detail:
            return
        rospy.logwarn("Mission state %d -> %d: %s", self.state, new_state, detail)
        self.state = new_state
        self.detail = detail
        self.state_since = rospy.Time.now()
        if new_state in (MissionState.HOLD, MissionState.FAULT, MissionState.LAND):
            self.stop_pub.publish(Empty())
            self.cmd = Twist()
        if new_state in (MissionState.HOLD_READY, MissionState.HOLD, MissionState.FAULT):
            self._latch_hold()

    def _fcu_cb(self, msg): self.fcu = msg
    def _extended_cb(self, msg): self.extended = msg
    def _rgb_cb(self, msg): self.last_rgb = rospy.Time.now()
    def _depth_cb(self, msg): self.last_depth = rospy.Time.now()

    def _vlm_cb(self, msg):
        if msg.status and all(s.level == DiagnosticStatus.OK for s in msg.status):
            self.last_vlm_health = rospy.Time.now()

    def _semantic_cb(self, msg):
        if not msg.yolo_valid or not msg.clip_valid or not msg.target_label:
            return
        now = rospy.Time.now()
        age = (now - msg.header.stamp).to_sec()
        if age < -0.02 or age > self.semantic_timeout:
            return
        self.last_semantic = now
        self.semantic_target = msg.target_label

    def _odom_cb(self, msg):
        self.odom = msg
        self.last_odom = rospy.Time.now()

    def _cmd_cb(self, msg):
        self.cmd = msg
        self.last_cmd = rospy.Time.now()

    def _result_cb(self, msg):
        if self.state != MissionState.AUTO:
            return
        if msg.data == 4:
            self._transition(MissionState.HOLD, "target reached")
        elif msg.data in (2, 3):
            self._transition(MissionState.FAULT, "planner terminated with result %d" % msg.data)

    def _healthy(self):
        now = rospy.Time.now()
        return (self.fcu.connected and self.odom is not None and
                (now - self.last_odom).to_sec() <= self.odom_timeout and
                (now - self.last_rgb).to_sec() <= self.sensor_timeout and
                (now - self.last_depth).to_sec() <= self.sensor_timeout and
                not self.last_vlm_health.is_zero() and
                (now - self.last_vlm_health).to_sec() <= 3.5 and
                not self.last_semantic.is_zero() and
                self.semantic_target == self.target and
                (now - self.last_semantic).to_sec() <= self.semantic_timeout)

    def _start(self, req):
        label = req.target_label.strip()
        if not label:
            return StartMissionResponse(False, "target_label is required")
        if self.state not in (MissionState.WAIT_FCU, MissionState.DISARMED):
            return StartMissionResponse(False, "another mission is active")
        self.target = label
        self.requested = True
        self.planner_started = False
        self.last_semantic = rospy.Time(0)
        self.semantic_target = ""
        # Publish before health gating so perception can produce evidence for this
        # exact target while the vehicle remains in WAIT_FCU.
        self.label_pub.publish(String(data=self.target))
        if self.state == MissionState.DISARMED:
            self._transition(MissionState.WAIT_FCU, "new mission accepted")
        self.detail = "mission accepted; waiting for healthy inputs"
        return StartMissionResponse(True, "mission accepted")

    def _stop(self, _req):
        if self.state in (MissionState.WAIT_FCU, MissionState.DISARMED):
            return TriggerResponse(True, "vehicle is already inactive")
        self._transition(MissionState.LAND, "operator requested stop")
        return TriggerResponse(True, "controlled landing requested")

    def _latch_hold(self):
        if self.odom is None:
            return
        p = self.odom.pose.pose.position
        q = self.odom.pose.pose.orientation
        yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.hold_pose = (p.x, p.y, max(p.z, 0.0), yaw)

    def _setpoint(self, auto=False, takeoff=False):
        if self.odom is None:
            return
        if self.hold_pose is None:
            self._latch_hold()
        msg = PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        ignore_accel = (PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY |
                        PositionTarget.IGNORE_AFZ)
        if auto:
            q = self.odom.pose.pose.orientation
            yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
            forward = max(-self.max_speed, min(self.max_speed, self.cmd.linear.x))
            msg.velocity.x = forward * math.cos(yaw)
            msg.velocity.y = forward * math.sin(yaw)
            msg.position.z = self.altitude
            msg.yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, self.cmd.angular.z))
            msg.type_mask = (PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY |
                             PositionTarget.IGNORE_VZ | ignore_accel | PositionTarget.IGNORE_YAW)
        else:
            x, y, z, yaw = self.hold_pose
            msg.position.x, msg.position.y = x, y
            msg.position.z = self.altitude if takeoff else z
            msg.yaw = yaw
            msg.type_mask = (PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY |
                             PositionTarget.IGNORE_VZ | ignore_accel |
                             PositionTarget.IGNORE_YAW_RATE)
        self.setpoint_pub.publish(msg)

    def _call_limited(self, function, *args):
        now = rospy.Time.now()
        if (now - self.last_service_attempt).to_sec() < 1.0:
            return None

    @staticmethod
    def _service_succeeded(response, field):
        return response is not None and bool(getattr(response, field, False))
        self.last_service_attempt = now
        try:
            return function(*args)
        except rospy.ServiceException as exc:
            rospy.logwarn_throttle(2.0, "MAVROS service failed: %s", exc)
            return None

    def _tick(self, _event):
        now = rospy.Time.now()
        if self.state == MissionState.WAIT_FCU:
            if self.requested and self._healthy():
                self._latch_hold()
                self._transition(MissionState.PRESTREAM, "prestreaming offboard setpoints")
            return
        if self.state == MissionState.PRESTREAM:
            self._setpoint(takeoff=True)
            if (now - self.state_since).to_sec() >= self.prestream_sec:
                self._transition(MissionState.ARM, "arming")
            return
        if self.state == MissionState.ARM:
            self._setpoint(takeoff=True)
            if (now - self.state_since).to_sec() > self.arm_transition_timeout:
                self._transition(MissionState.FAULT, "OFFBOARD/arm transition timed out")
                return
            # PX4's intended external-control sequence is prestream -> OFFBOARD
            # -> ARM.  Arming in AUTO.LOITER first only spins the rotors and can
            # trigger ground auto-disarm before OFFBOARD is accepted.
            if self.fcu.mode != "OFFBOARD":
                response = self._call_limited(self.set_mode, custom_mode="OFFBOARD")
                if response is not None and not self._service_succeeded(response, "mode_sent"):
                    rospy.logwarn("PX4 rejected OFFBOARD mode request")
                return
            if not self.fcu.armed:
                response = self._call_limited(self.arm, True)
                if response is not None and not self._service_succeeded(response, "success"):
                    rospy.logwarn("PX4 rejected arm request after OFFBOARD entry")
                return
            self._transition(MissionState.OFFBOARD_TAKEOFF, "taking off")
            return
        if self.state == MissionState.OFFBOARD_TAKEOFF:
            self._setpoint(takeoff=True)
            if not self._healthy():
                self._transition(MissionState.FAULT, "critical input stale during takeoff")
                return
            current_z = self.odom.pose.pose.position.z
            if abs(current_z - self.altitude) <= self.alt_tol:
                if self.altitude_stable_since is None:
                    self.altitude_stable_since = now
                elif (now - self.altitude_stable_since).to_sec() >= self.alt_stable_sec:
                    self._transition(MissionState.HOLD_READY, "takeoff stable; map warmup")
            else:
                self.altitude_stable_since = None
            return
        if self.state == MissionState.HOLD_READY:
            self._setpoint()
            if (now - self.state_since).to_sec() >= 1.0 and self._healthy():
                goal = PoseStamped()
                goal.header.stamp = now
                goal.header.frame_id = "map"
                goal.pose.orientation.w = 1.0
                self.trigger_pub.publish(goal)
                self.planner_started = True
                self._transition(MissionState.AUTO, "planner active")
            return
        if self.state == MissionState.AUTO:
            if not self._healthy():
                self._transition(MissionState.FAULT, "critical input stale")
                return
            if self.last_cmd.is_zero() and (now - self.state_since).to_sec() <= self.planner_start_timeout:
                self._setpoint()
                return
            if self.last_cmd.is_zero() or (now - self.last_cmd).to_sec() > self.cmd_timeout:
                self._transition(MissionState.FAULT, "planner command timeout")
                return
            self._setpoint(auto=True)
            return
        if self.state in (MissionState.HOLD, MissionState.FAULT):
            self._setpoint()
            delay = self.completion_hold if self.state == MissionState.HOLD else self.fault_land_delay
            if (now - self.state_since).to_sec() >= delay:
                self._transition(MissionState.LAND, self.detail + "; landing")
            return
        if self.state == MissionState.LAND:
            self._setpoint()
            if self.fcu.mode != "AUTO.LAND":
                self._call_limited(self.set_mode, custom_mode="AUTO.LAND")
            if self.extended.landed_state == ExtendedState.LANDED_STATE_ON_GROUND:
                if self.fcu.armed:
                    self._call_limited(self.arm, False)
                else:
                    self.requested = False
                    self._transition(MissionState.DISARMED, "landed and disarmed")

    def _publish_state(self, _event):
        msg = MissionState()
        msg.header.stamp = rospy.Time.now()
        msg.state = self.state
        msg.target_label = self.target
        msg.detail = self.detail
        msg.fcu_connected = self.fcu.connected
        msg.armed = self.fcu.armed
        msg.px4_mode = self.fcu.mode
        msg.altitude = self.odom.pose.pose.position.z if self.odom else float("nan")
        msg.planner_active = self.state == MissionState.AUTO
        self.state_pub.publish(msg)


if __name__ == "__main__":
    Supervisor()
    rospy.spin()
