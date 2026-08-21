#!/usr/bin/env python3
"""Lifecycle and watchdog boundary between the 2-D planner and PX4 Offboard."""

import math

import rospy
import tf.transformations as tft
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped, Twist
from mavros_msgs.msg import ExtendedState, PositionTarget, State
from mavros_msgs.srv import CommandBool, ParamGet, SetMode
from nav_msgs.msg import Odometry
from plan_env.msg import MissionState, SemanticObservation
from plan_env.srv import StartMission, StartMissionResponse
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Empty, Header, Int32, String
from std_srvs.srv import Trigger, TriggerResponse


class Supervisor:
    PLANNER_WAIT_TRIGGER = 1
    PLANNER_PLAN_TRAJ = 2
    PLANNER_ACTIVE_STATES = (2, 3, 4)
    PLANNER_EXECUTION_STATES = (3, 4)
    PLANNER_FINISH = 5
    RESULT_EXPLORE = 0
    RESULT_SEARCH_OBJECT = 1

    def __init__(self):
        rospy.init_node("apexnav_px4_mission_supervisor")
        self.rate_hz = float(rospy.get_param("~setpoint_rate", 30.0))
        self.prestream_sec = float(rospy.get_param("~prestream_sec", 2.0))
        self.altitude = float(rospy.get_param("~takeoff_altitude", 1.0))
        self.alt_stable_sec = float(rospy.get_param("~altitude_stable_sec", 2.5))
        self.takeoff_ready_altitude_tolerance = float(
            rospy.get_param("~takeoff_ready_altitude_tolerance", 0.05))
        self.takeoff_ready_vertical_speed_tolerance = float(
            rospy.get_param("~takeoff_ready_vertical_speed_tolerance", 0.025))
        self.completion_hold = float(rospy.get_param("~completion_hold_sec", 2.0))
        self.fault_land_delay = float(rospy.get_param("~fault_land_delay_sec", 3.0))
        self.cmd_timeout = float(rospy.get_param("~command_timeout_sec", 0.30))
        self.first_reference_timeout = float(
            rospy.get_param("~first_reference_timeout_sec", 0.80))
        self.planner_start_timeout = float(rospy.get_param("~planner_start_timeout_sec", 10.0))
        self.planner_replan_timeout = float(
            rospy.get_param("~planner_replan_timeout_sec", 5.0))
        self.planner_replan_total_timeout = float(
            rospy.get_param("~planner_replan_total_timeout_sec", 12.0))
        self.planner_ready_timeout = float(rospy.get_param("~planner_ready_timeout_sec", 10.0))
        self.planner_ack_timeout = float(rospy.get_param("~planner_ack_timeout_sec", 2.0))
        self.navigation_ready_timeout = float(
            rospy.get_param("~navigation_ready_timeout_sec", 30.0))
        self.arm_transition_timeout = float(rospy.get_param("~arm_transition_timeout_sec", 10.0))
        self.odom_timeout = float(rospy.get_param("~odom_timeout_sec", 0.20))
        self.extended_state_timeout = float(
            rospy.get_param("~extended_state_timeout_sec", 2.0))
        self.sensor_timeout = float(rospy.get_param("~sensor_timeout_sec", 0.50))
        self.vision_timeout = float(
            rospy.get_param("~vision_height_timeout_sec", 0.15))
        self.semantic_timeout = float(rospy.get_param("~semantic_timeout_sec", 5.0))
        self.input_future_tolerance = float(
            rospy.get_param("~input_future_tolerance_sec", 0.02))
        self.world_frame = str(rospy.get_param("~world_frame", "map"))
        self.base_frame = str(rospy.get_param("~base_frame", "base_link"))
        self.camera_frame = str(rospy.get_param(
            "~camera_frame", "iris_camera_optical_frame"))
        self.tracking_mode = str(rospy.get_param("~tracking_mode", "px4_native")).strip().lower()
        if self.tracking_mode not in ("px4_native", "legacy_mpc"):
            raise ValueError("tracking_mode must be px4_native or legacy_mpc")
        self.max_speed = float(rospy.get_param("~max_horizontal_speed", 0.40))
        self.max_yaw_rate = float(rospy.get_param("~max_yaw_rate", 0.65))
        self.max_reference_position_jump = float(
            rospy.get_param("~max_reference_position_jump", 0.20))
        self.px4_height_profile_check_enabled = bool(
            rospy.get_param("~px4_height_profile_check_enabled", True))
        self.px4_expected_height_reference = int(
            rospy.get_param("~px4_expected_ekf2_hgt_ref", 3))
        self.px4_expected_ev_control = int(
            rospy.get_param("~px4_expected_ekf2_ev_ctrl", 2))
        self.px4_expected_baro_noise = float(
            rospy.get_param("~px4_expected_ekf2_baro_noise", 3.5))
        self.px4_expected_hover_thrust = float(
            rospy.get_param("~px4_expected_mpc_thr_hover", 0.70))
        self.px4_expected_offboard_loss_action = int(
            rospy.get_param("~px4_expected_com_obl_rc_act", 4))
        self.px4_expected_rc_input_mode = int(
            rospy.get_param("~px4_expected_com_rc_in_mode", 4))
        self.px4_expected_offboard_loss_timeout = float(
            rospy.get_param("~px4_expected_com_of_loss_t", 1.0))
        self.px4_expected_land_speed = float(
            rospy.get_param("~px4_expected_mpc_land_speed", 0.60))
        self.px4_profile_tolerance = float(
            rospy.get_param("~px4_height_profile_tolerance", 0.02))
        self.gazebo_truth_guard_enabled = bool(
            rospy.get_param("~gazebo_truth_guard_enabled", True))
        self.gazebo_model_name = str(
            rospy.get_param("~gazebo_model_name", "iris_depth_camera"))
        self.gazebo_truth_max_error = float(
            rospy.get_param("~gazebo_truth_max_error", 0.15))
        self.gazebo_truth_hold_sec = float(
            rospy.get_param("~gazebo_truth_hold_sec", 1.0))
        self.gazebo_truth_timeout = float(
            rospy.get_param("~gazebo_truth_timeout_sec", 0.50))
        self.gazebo_truth_min_airborne_altitude = float(
            rospy.get_param("~gazebo_truth_min_airborne_altitude", 0.20))
        self.controlled_land_speed = float(
            rospy.get_param("~controlled_land_speed", 0.25))
        self.controlled_land_timeout = float(
            rospy.get_param("~controlled_land_timeout_sec", 10.0))
        self.mapping_update_timeout = float(
            rospy.get_param("~mapping_update_timeout_sec", 3.0))
        self.altitude_soft_error = float(
            rospy.get_param("~altitude_soft_error", 0.30))
        self.altitude_recovery_error = float(
            rospy.get_param("~altitude_recovery_error", 0.15))
        self.altitude_hard_error = float(
            rospy.get_param("~altitude_hard_error", 0.60))
        self.altitude_hard_hold_sec = float(
            rospy.get_param("~altitude_hard_hold_sec", 2.0))
        self.altitude_recovery_timeout = float(
            rospy.get_param("~altitude_recovery_timeout_sec", 8.0))
        self.recovery_stable_sec = float(
            rospy.get_param("~recovery_stable_sec", 1.0))

        self._validate_configuration()

        self.state = MissionState.WAIT_FCU
        self.detail = "waiting for mission and FCU"
        self.target = ""
        self.requested = False
        self.fcu = State()
        self.extended = ExtendedState()
        self.last_extended_state = rospy.Time(0)
        self.odom = None
        self.vertical_speed = 0.0
        self.last_odom_z = None
        self.last_odom_sample_time = rospy.Time(0)
        self.cmd = Twist()
        self.trajectory_reference = PositionTarget()
        self.last_odom = rospy.Time(0)
        self.last_cmd = rospy.Time(0)
        self.last_trajectory_reference = rospy.Time(0)
        self.trajectory_reference_epoch = rospy.Time(0)
        self.last_tick_time = rospy.Time(0)
        self.last_rgb = rospy.Time(0)
        self.last_depth = rospy.Time(0)
        self.last_vision_height = rospy.Time(0)
        self.last_vlm_health = rospy.Time(0)
        self.last_semantic = rospy.Time(0)
        self.semantic_target = ""
        self.state_since = rospy.Time.now()
        self.hold_pose = None
        self.ground_z = None
        self.cruise_z = None
        self.altitude_stable_since = None
        self.last_service_attempt = rospy.Time(0)
        self.planner_started = False
        self.planner_state = None
        self.last_planner_state = rospy.Time(0)
        self.planner_state_since = rospy.Time(0)
        self.planner_phase_since = rospy.Time(0)
        self.planner_fallback_budget_granted = False
        self.planner_execution_since = rospy.Time(0)
        self.planner_trigger_time = rospy.Time(0)
        self.last_result = None
        self.last_mapping_update = rospy.Time(0)
        self.mapping_enabled = False
        self.navigation_enabled = False
        self.altitude_recovery_active = False
        self.altitude_recovery_since = rospy.Time(0)
        self.altitude_recovery_stable_since = rospy.Time(0)
        self.altitude_hard_since = rospy.Time(0)
        self.px4_height_profile_verified = False
        self.last_px4_profile_check = rospy.Time(0)
        self.cached_px4_profile_failure = None
        self.gazebo_z = None
        self.gazebo_ground_z = None
        self.last_gazebo_truth = rospy.Time(0)
        self.gazebo_truth_error_since = rospy.Time(0)
        self.landing_target_z = None
        self.landing_last_update = rospy.Time(0)

        self.setpoint_pub = rospy.Publisher(
            "/mavros/setpoint_raw/local", PositionTarget, queue_size=10)
        self.state_pub = rospy.Publisher("/apexnav/mission/state", MissionState, queue_size=2, latch=True)
        self.label_pub = rospy.Publisher("/detector/label", String, queue_size=1, latch=True)
        self.trigger_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        self.stop_pub = rospy.Publisher("/traj_server/stop", Empty, queue_size=1)
        self.cancel_planner_pub = rospy.Publisher("/apexnav/planner/cancel", Empty, queue_size=1)
        self.navigation_enabled_pub = rospy.Publisher(
            "/apexnav/mission/navigation_enabled", Bool, queue_size=1, latch=True)
        self.mapping_enabled_pub = rospy.Publisher(
            "/apexnav/mission/mapping_enabled", Bool, queue_size=1, latch=True)
        self.mapping_enabled_pub.publish(Bool(data=False))
        self.navigation_enabled_pub.publish(Bool(data=False))
        rospy.Subscriber("/mavros/state", State, self._fcu_cb, queue_size=5)
        rospy.Subscriber("/mavros/extended_state", ExtendedState, self._extended_cb, queue_size=5)
        rospy.Subscriber("/mavros/local_position/odom", Odometry, self._odom_cb, queue_size=20)
        rospy.Subscriber("/mavros/vision_pose/pose", PoseStamped,
                         self._vision_height_cb, queue_size=10)
        if self.gazebo_truth_guard_enabled:
            rospy.Subscriber("/gazebo/model_states", ModelStates,
                             self._gazebo_model_states_cb, queue_size=5)
        rospy.Subscriber("/apexnav/planner/cmd_vel_raw", Twist, self._cmd_cb, queue_size=5)
        rospy.Subscriber("/apexnav/planner/trajectory_reference", PositionTarget,
                         self._trajectory_reference_cb, queue_size=5)
        rospy.Subscriber("/apexnav/camera/rgb/image_raw", Image, self._rgb_cb, queue_size=2)
        rospy.Subscriber("/apexnav/camera/depth/image_raw", Image, self._depth_cb, queue_size=2)
        rospy.Subscriber("/grid_map/commit", Header,
                         self._mapping_update_cb, queue_size=2)
        rospy.Subscriber("/apexnav/vlm/diagnostics", DiagnosticArray, self._vlm_cb, queue_size=2)
        rospy.Subscriber("/apexnav/vlm/semantic_observation", SemanticObservation,
                         self._semantic_cb, queue_size=3)
        rospy.Subscriber("/ros/expl_result", Int32, self._result_cb, queue_size=5)
        rospy.Subscriber("/ros/state", Int32, self._planner_state_cb, queue_size=5)
        self.arm = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.set_mode = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        self.param_get = rospy.ServiceProxy("/mavros/param/get", ParamGet)
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
        self._set_mapping_enabled(
            new_state in (MissionState.HOLD_READY, MissionState.AUTO))
        self._set_navigation_enabled(
            new_state in (MissionState.HOLD_READY, MissionState.AUTO))
        if new_state in (MissionState.HOLD, MissionState.FAULT, MissionState.LAND):
            self.stop_pub.publish(Empty())
            self.cmd = Twist()
            self.last_cmd = rospy.Time(0)
            self.last_trajectory_reference = rospy.Time(0)
            if self.planner_started:
                self.cancel_planner_pub.publish(Empty())
                self.planner_started = False
        if new_state in (MissionState.HOLD_READY, MissionState.HOLD, MissionState.FAULT):
            self._latch_hold()
        if new_state == MissionState.LAND:
            self._latch_hold()
            self.landing_target_z = (
                self.odom.pose.pose.position.z if self.odom is not None else None)
            self.landing_last_update = self.state_since

    def _validate_configuration(self):
        """Fail before creating ROS endpoints when a safety limit is invalid."""
        positive = {
            "setpoint_rate": self.rate_hz,
            "takeoff_altitude": self.altitude,
            "command_timeout_sec": self.cmd_timeout,
            "first_reference_timeout_sec": self.first_reference_timeout,
            "planner_start_timeout_sec": self.planner_start_timeout,
            "planner_replan_timeout_sec": self.planner_replan_timeout,
            "planner_replan_total_timeout_sec": self.planner_replan_total_timeout,
            "planner_ready_timeout_sec": self.planner_ready_timeout,
            "planner_ack_timeout_sec": self.planner_ack_timeout,
            "navigation_ready_timeout_sec": self.navigation_ready_timeout,
            "arm_transition_timeout_sec": self.arm_transition_timeout,
            "odom_timeout_sec": self.odom_timeout,
            "extended_state_timeout_sec": self.extended_state_timeout,
            "sensor_timeout_sec": self.sensor_timeout,
            "vision_height_timeout_sec": self.vision_timeout,
            "semantic_timeout_sec": self.semantic_timeout,
            "mapping_update_timeout_sec": self.mapping_update_timeout,
            "max_horizontal_speed": self.max_speed,
            "max_yaw_rate": self.max_yaw_rate,
            "max_reference_position_jump": self.max_reference_position_jump,
            "takeoff_ready_altitude_tolerance": self.takeoff_ready_altitude_tolerance,
            "takeoff_ready_vertical_speed_tolerance":
                self.takeoff_ready_vertical_speed_tolerance,
            "altitude_recovery_timeout_sec": self.altitude_recovery_timeout,
            "altitude_hard_hold_sec": self.altitude_hard_hold_sec,
            "gazebo_truth_max_error": self.gazebo_truth_max_error,
            "px4_expected_ekf2_baro_noise": self.px4_expected_baro_noise,
            "px4_expected_mpc_thr_hover": self.px4_expected_hover_thrust,
            "px4_expected_mpc_land_speed": self.px4_expected_land_speed,
            "px4_expected_com_of_loss_t": self.px4_expected_offboard_loss_timeout,
            "controlled_land_speed": self.controlled_land_speed,
            "controlled_land_timeout_sec": self.controlled_land_timeout,
        }
        nonnegative = {
            "prestream_sec": self.prestream_sec,
            "altitude_stable_sec": self.alt_stable_sec,
            "completion_hold_sec": self.completion_hold,
            "fault_land_delay_sec": self.fault_land_delay,
            "input_future_tolerance_sec": self.input_future_tolerance,
            "gazebo_truth_hold_sec": self.gazebo_truth_hold_sec,
            "gazebo_truth_timeout_sec": self.gazebo_truth_timeout,
            "gazebo_truth_min_airborne_altitude":
                self.gazebo_truth_min_airborne_altitude,
            "recovery_stable_sec": self.recovery_stable_sec,
            "px4_height_profile_tolerance": self.px4_profile_tolerance,
        }
        invalid = [name for name, value in positive.items()
                   if not math.isfinite(value) or value <= 0.0]
        invalid.extend(name for name, value in nonnegative.items()
                       if not math.isfinite(value) or value < 0.0)
        if (not all((self.world_frame, self.base_frame, self.camera_frame)) or
                len({self.world_frame, self.base_frame, self.camera_frame}) != 3):
            invalid.append("world/base/camera frame contract")
        if not (math.isfinite(self.altitude_recovery_error) and
                math.isfinite(self.altitude_soft_error) and
                math.isfinite(self.altitude_hard_error) and
                0.0 <= self.altitude_recovery_error < self.altitude_soft_error <
                self.altitude_hard_error):
            invalid.append("altitude recovery/soft/hard ordering")
        if self.first_reference_timeout < self.cmd_timeout:
            invalid.append("first_reference_timeout_sec >= command_timeout_sec")
        if self.planner_replan_total_timeout < self.planner_replan_timeout:
            invalid.append("planner_replan_total_timeout_sec >= planner_replan_timeout_sec")
        if (self.px4_expected_height_reference != 3 or
                self.px4_expected_ev_control != 2 or
                self.px4_expected_offboard_loss_action != 4 or
                self.px4_expected_rc_input_mode != 4 or
                self.px4_expected_offboard_loss_timeout > 2.0):
            invalid.append("safe PX4 height/Offboard-loss profile")
        if invalid:
            raise ValueError("invalid supervisor configuration: " + ", ".join(invalid))

    def _reset_px4_height_profile_verification(self):
        self.px4_height_profile_verified = False
        self.last_px4_profile_check = rospy.Time(0)
        self.cached_px4_profile_failure = None

    def _fcu_cb(self, msg):
        if msg.connected != self.fcu.connected:
            self._reset_px4_height_profile_verification()
        self.fcu = msg
    def _extended_cb(self, msg):
        self.extended = msg
        self.last_extended_state = rospy.Time.now()

    def _valid_source_header(self, header, now, timeout, expected_frame,
                             previous_stamp):
        stamp = header.stamp
        if stamp.is_zero() or (expected_frame and header.frame_id != expected_frame):
            return False
        age = (now - stamp).to_sec()
        if age < -self.input_future_tolerance or age > timeout:
            return False
        if not previous_stamp.is_zero() and stamp <= previous_stamp:
            return False
        return True

    def _image_source_valid(self, msg, now, previous_stamp, expected_encoding):
        if not self._valid_source_header(
                msg.header, now, self.sensor_timeout, self.camera_frame,
                previous_stamp):
            return False
        if msg.width <= 0 or msg.height <= 0:
            return False
        if msg.encoding not in expected_encoding:
            return False
        bytes_per_pixel = {"rgb8": 3, "32FC1": 4, "16UC1": 2}[msg.encoding]
        minimum_step = int(msg.width) * bytes_per_pixel
        return (msg.step >= minimum_step and
                len(msg.data) >= int(msg.step) * int(msg.height))

    def _rgb_cb(self, msg):
        now = rospy.Time.now()
        if self._image_source_valid(msg, now, self.last_rgb, ("rgb8",)):
            self.last_rgb = msg.header.stamp

    def _depth_cb(self, msg):
        now = rospy.Time.now()
        if self._image_source_valid(
                msg, now, self.last_depth, ("32FC1", "16UC1")):
            self.last_depth = msg.header.stamp

    def _vision_height_cb(self, msg):
        now = rospy.Time.now()
        if not self._valid_source_header(
                msg.header, now, self.vision_timeout, self.world_frame,
                self.last_vision_height):
            return
        p, q = msg.pose.position, msg.pose.orientation
        values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        if not all(math.isfinite(value) for value in values):
            return
        qnorm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        if qnorm < 0.95 or qnorm > 1.05:
            return
        self.last_vision_height = msg.header.stamp

    def _mapping_update_cb(self, msg):
        if not self.mapping_enabled:
            return
        now = rospy.Time.now()
        if self._valid_source_header(
                msg, now, self.mapping_update_timeout, self.world_frame,
                self.last_mapping_update):
            self.last_mapping_update = msg.stamp

    def _vlm_cb(self, msg):
        now = rospy.Time.now()
        if not self._valid_source_header(
                msg.header, now, 3.5, "", self.last_vlm_health):
            return
        ready = {}
        for status in msg.status:
            values = {value.key: value.value.strip().lower()
                      for value in status.values}
            if "yoloe_ready" in values:
                ready["yoloe"] = values["yoloe_ready"] == "true"
            if "clipitm_ready" in values:
                ready["clipitm"] = values["clipitm_ready"] == "true"
        if ready.get("yoloe") and ready.get("clipitm"):
            self.last_vlm_health = msg.header.stamp

    def _semantic_cb(self, msg):
        if not msg.yolo_valid or not msg.clip_valid or not msg.target_label:
            return
        now = rospy.Time.now()
        if not self._valid_source_header(
                msg.header, now, self.semantic_timeout, self.world_frame,
                self.last_semantic):
            return
        self.last_semantic = msg.header.stamp
        self.semantic_target = msg.target_label

    def _odom_cb(self, msg):
        sample_time = msg.header.stamp
        now = rospy.Time.now()
        if not self._valid_source_header(
                msg.header, now, self.odom_timeout, self.world_frame,
                self.last_odom):
            return
        if msg.child_frame_id != self.base_frame:
            return
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        linear = msg.twist.twist.linear
        angular = msg.twist.twist.angular
        values = [p.x, p.y, p.z, q.x, q.y, q.z, q.w,
                  linear.x, linear.y, linear.z,
                  angular.x, angular.y, angular.z]
        values.extend(msg.pose.covariance)
        values.extend(msg.twist.covariance)
        if not all(math.isfinite(value) for value in values):
            return
        quaternion_norm = math.sqrt(q.x * q.x + q.y * q.y +
                                    q.z * q.z + q.w * q.w)
        if quaternion_norm < 0.95 or quaternion_norm > 1.05:
            return
        current_z = msg.pose.pose.position.z
        if self.last_odom_z is not None and not self.last_odom_sample_time.is_zero():
            dt = (sample_time - self.last_odom_sample_time).to_sec()
            # Derive vertical motion in the same world frame as the altitude
            # gate. PX4/MAVROS twist.z can use a different sign/frame in SITL.
            if 0.005 <= dt <= 0.50:
                raw_vertical_speed = (current_z - self.last_odom_z) / dt
                self.vertical_speed = 0.65 * self.vertical_speed + 0.35 * raw_vertical_speed
            elif dt < -0.001 or dt > 0.50:
                self.vertical_speed = 0.0
        self.last_odom_z = current_z
        self.last_odom_sample_time = sample_time
        self.odom = msg
        self.last_odom = sample_time

    def _gazebo_model_states_cb(self, msg):
        if not self.gazebo_truth_guard_enabled:
            return
        try:
            model_index = msg.name.index(self.gazebo_model_name)
        except ValueError:
            return
        if model_index >= len(msg.pose):
            return
        gazebo_z = msg.pose[model_index].position.z
        if not math.isfinite(gazebo_z):
            return
        self.gazebo_z = gazebo_z
        self.last_gazebo_truth = rospy.Time.now()
        # Refresh the independent guard zero only while disarmed.  The
        # separate vision bridge feeds PX4; this callback never generates a
        # command and remains an independent consistency check.
        if not self.fcu.armed:
            self.gazebo_ground_z = self.gazebo_z
            self.gazebo_truth_error_since = rospy.Time(0)

    def _cmd_cb(self, msg):
        if not all(math.isfinite(value) for value in (
                msg.linear.x, msg.linear.y, msg.linear.z,
                msg.angular.x, msg.angular.y, msg.angular.z)):
            rospy.logwarn_throttle(2.0, "Discarding non-finite legacy planner command")
            return
        self.cmd = msg
        if self.tracking_mode == "legacy_mpc":
            self.last_cmd = rospy.Time.now()

    def _trajectory_reference_cb(self, msg):
        now = rospy.Time.now()
        if not self._valid_trajectory_reference(msg, now):
            rospy.logwarn_throttle(
                2.0, "Discarding invalid or stale trajectory reference")
            return
        self.trajectory_reference = msg
        # Preserve source time separately from callback receipt time so a
        # delayed reference cannot revive a new trajectory execution epoch.
        self.last_trajectory_reference = msg.header.stamp
        if self.tracking_mode == "px4_native":
            # last_cmd is the generic execution-stream timestamp used by the
            # supervisor's epoch and staleness watchdogs.
            self.last_cmd = now

    @staticmethod
    def _trajectory_reference_is_finite(reference):
        return all(math.isfinite(value) for value in (
            reference.position.x, reference.position.y,
            reference.velocity.x, reference.velocity.y,
            reference.yaw, reference.yaw_rate))

    def _trajectory_reference_execution_epoch(self):
        epoch = getattr(self, "trajectory_reference_epoch", rospy.Time(0))
        if not epoch.is_zero():
            return epoch
        epoch = self.planner_execution_since
        if epoch.is_zero():
            epoch = self.planner_state_since
        return epoch

    def _valid_trajectory_reference(self, reference, now):
        stamp = reference.header.stamp
        if (stamp.is_zero() or reference.header.frame_id != self.world_frame or
                reference.coordinate_frame != PositionTarget.FRAME_LOCAL_NED or
                not self._trajectory_reference_is_finite(reference)):
            return False
        if stamp > now:
            return False
        if (now - stamp).to_sec() > self.cmd_timeout:
            return False
        execution_epoch = self._trajectory_reference_execution_epoch()
        if not execution_epoch.is_zero() and stamp < execution_epoch:
            return False
        if (not self.last_trajectory_reference.is_zero() and
                (execution_epoch.is_zero() or
                 self.last_trajectory_reference >= execution_epoch) and
                stamp <= self.last_trajectory_reference):
            return False
        previous_is_current = (
            not self.last_trajectory_reference.is_zero() and
            (execution_epoch.is_zero() or
             self.last_trajectory_reference >= execution_epoch))
        if previous_is_current:
            dx = reference.position.x - self.trajectory_reference.position.x
            dy = reference.position.y - self.trajectory_reference.position.y
        elif self.odom is not None:
            dx = reference.position.x - self.odom.pose.pose.position.x
            dy = reference.position.y - self.odom.pose.pose.position.y
        else:
            return True
        return math.hypot(dx, dy) <= self.max_reference_position_jump

    def _planner_state_cb(self, msg):
        if msg.data != self.planner_state:
            previous_state = self.planner_state
            transition_time = rospy.Time.now()
            self.planner_state_since = transition_time
            if msg.data == self.PLANNER_PLAN_TRAJ:
                self.planner_phase_since = transition_time
                self.planner_fallback_budget_granted = False
                self.trajectory_reference = PositionTarget()
                self.last_trajectory_reference = rospy.Time(0)
                self.trajectory_reference_epoch = transition_time
            # A command from the trajectory that existed before PLAN_TRAJ is
            # stale for the newly generated trajectory.  Start a fresh command
            # epoch only when planning enters an executable state; EXEC_TRAJ ->
            # REPLAN is the normal streaming phase and keeps the same epoch.
            if (msg.data in self.PLANNER_EXECUTION_STATES and
                    previous_state not in self.PLANNER_EXECUTION_STATES):
                self.planner_execution_since = transition_time
            elif msg.data not in self.PLANNER_EXECUTION_STATES:
                self.planner_execution_since = rospy.Time(0)
            # Replanning hold must use the current aircraft pose, not the
            # original takeoff hold pose saved before exploration began.
            if msg.data == self.PLANNER_PLAN_TRAJ and self.state == MissionState.AUTO:
                self._latch_hold()
        self.planner_state = msg.data
        self.last_planner_state = rospy.Time.now()

    def _handle_planner_result(self, result):
        if result == 4:
            self._transition(MissionState.HOLD, "target reached")
            return True
        if result in (2, 3):
            self._transition(MissionState.FAULT, "planner terminated with result %d" % result)
            return True
        return False

    def _result_cb(self, msg):
        if self.state not in (MissionState.HOLD_READY, MissionState.AUTO) or not self.planner_started:
            return
        previous_result = self.last_result
        self.last_result = msg.data
        # Object approach and ordinary exploration are different recovery
        # phases. Grant one fresh per-phase timeout when the planner explicitly
        # falls back, while the independent total timeout still bounds the
        # complete PLAN_TRAJ episode.
        if (self.state == MissionState.AUTO and
                self.planner_state == self.PLANNER_PLAN_TRAJ and
                previous_result == self.RESULT_SEARCH_OBJECT and
                msg.data == self.RESULT_EXPLORE and
                not self.planner_fallback_budget_granted):
            self.planner_phase_since = rospy.Time.now()
            self.planner_fallback_budget_granted = True
            rospy.logwarn(
                "Object approach unavailable; reset planning phase budget for exploration fallback")
        self._handle_planner_result(msg.data)

    @staticmethod
    def _fresh(now, stamp, timeout):
        if stamp.is_zero():
            return False
        age = (now - stamp).to_sec()
        return 0.0 <= age <= timeout

    def _health_failure(self):
        failure = self._flight_inputs_failure()
        if failure is not None:
            return failure
        now = rospy.Time.now()
        if not self.mapping_enabled:
            return "depth mapping disabled"
        if self.last_mapping_update.is_zero():
            return "depth map output unavailable"
        mapping_age = (now - self.last_mapping_update).to_sec()
        if mapping_age < 0.0:
            return "depth map timestamp is %.3fs in the future" % -mapping_age
        if mapping_age > self.mapping_update_timeout:
            return "depth map output stale (age %.3fs > %.3fs)" % (
                mapping_age, self.mapping_update_timeout)
        if not self._fresh(now, self.last_vlm_health, 3.5):
            return "healthy YOLOE/CLIPITM"
        if (self.semantic_target != self.target or
                not self._fresh(now, self.last_semantic, self.semantic_timeout)):
            return "fresh semantic observation for '%s'" % self.target
        return None

    def _perception_servers_failure(self, now=None):
        if now is None:
            now = rospy.Time.now()
        if not self._fresh(now, self.last_vlm_health, 3.5):
            return "ready YOLOE/CLIPITM servers"
        return None

    def _flight_inputs_failure(self):
        now = rospy.Time.now()
        if not self.fcu.connected:
            return "FCU connection"
        if self.odom is None or not self._fresh(now, self.last_odom, self.odom_timeout):
            return "fresh odometry"
        if not self._fresh(now, self.last_rgb, self.sensor_timeout):
            return "fresh RGB"
        if not self._fresh(now, self.last_depth, self.sensor_timeout):
            return "fresh depth"
        if (self.px4_expected_height_reference == 3 and
                not self._fresh(now, self.last_vision_height,
                                self.vision_timeout)):
            return "fresh Gazebo vision height"
        return None

    def _px4_height_profile_failure(self):
        """Verify the estimator/controller profile before sending arm setpoints."""
        if not self.px4_height_profile_check_enabled:
            return None
        if self.px4_height_profile_verified:
            return None
        now = rospy.Time.now()
        if (not self.last_px4_profile_check.is_zero() and
                0.0 <= (now - self.last_px4_profile_check).to_sec() < 1.0):
            return self.cached_px4_profile_failure
        self.last_px4_profile_check = now
        expected = (
            ("EKF2_HGT_REF", "integer", self.px4_expected_height_reference, 0.0),
            ("EKF2_EV_CTRL", "integer", self.px4_expected_ev_control, 0.0),
            ("EKF2_BARO_NOISE", "real", self.px4_expected_baro_noise,
             self.px4_profile_tolerance),
            ("MPC_THR_HOVER", "real", self.px4_expected_hover_thrust,
             self.px4_profile_tolerance),
            ("MPC_LAND_SPEED", "real", self.px4_expected_land_speed,
             self.px4_profile_tolerance),
            ("COM_OBL_RC_ACT", "integer",
             self.px4_expected_offboard_loss_action, 0.0),
            ("COM_RC_IN_MODE", "integer",
             self.px4_expected_rc_input_mode, 0.0),
            ("COM_OF_LOSS_T", "real",
             self.px4_expected_offboard_loss_timeout,
             self.px4_profile_tolerance),
        )
        for param_id, value_kind, target, tolerance in expected:
            try:
                response = self.param_get(param_id=param_id)
            except rospy.ServiceException as exc:
                self.cached_px4_profile_failure = (
                    "PX4 parameter service unavailable: %s" % exc)
                return self.cached_px4_profile_failure
            if response is None or not response.success:
                self.cached_px4_profile_failure = (
                    "PX4 parameter %s unavailable" % param_id)
                return self.cached_px4_profile_failure
            actual = (response.value.integer if value_kind == "integer"
                      else response.value.real)
            if abs(float(actual) - float(target)) > tolerance:
                self.cached_px4_profile_failure = (
                    "PX4 parameter %s=%g, expected %g; reboot PX4 after applying "
                    "the ApexNav SITL height profile" % (param_id, actual, target))
                return self.cached_px4_profile_failure
        self.px4_height_profile_verified = True
        self.cached_px4_profile_failure = None
        rospy.loginfo("PX4 SITL height and Offboard-loss profile verified")
        return None

    def _height_consistency_failure(self, now):
        """Compare PX4's vision-aided estimate against the direct Gazebo guard."""
        if not self.gazebo_truth_guard_enabled:
            return None
        if self.gazebo_z is None or self.gazebo_ground_z is None:
            self.gazebo_truth_error_since = rospy.Time(0)
            return "Gazebo vehicle height reference unavailable"
        if not self._fresh(now, self.last_gazebo_truth,
                           self.gazebo_truth_timeout):
            self.gazebo_truth_error_since = rospy.Time(0)
            return "Gazebo vehicle height truth stale"
        if self.odom is None or self.ground_z is None:
            self.gazebo_truth_error_since = rospy.Time(0)
            return "MAVROS height reference unavailable"
        estimator_agl = self.odom.pose.pose.position.z - self.ground_z
        truth_agl = self.gazebo_z - self.gazebo_ground_z
        if max(estimator_agl, truth_agl) < self.gazebo_truth_min_airborne_altitude:
            self.gazebo_truth_error_since = rospy.Time(0)
            return None
        error = abs(estimator_agl - truth_agl)
        if error <= self.gazebo_truth_max_error:
            self.gazebo_truth_error_since = rospy.Time(0)
            return None
        if self.gazebo_truth_error_since.is_zero():
            self.gazebo_truth_error_since = now
            return ("pending height estimator/truth disagreement %.3fm "
                    "(0.00/%.2fs)" % (error, self.gazebo_truth_hold_sec))
        held = (now - self.gazebo_truth_error_since).to_sec()
        if held < 0.0:
            self.gazebo_truth_error_since = now
            return ("pending height estimator/truth disagreement %.3fm "
                    "after clock reset" % error)
        if held < self.gazebo_truth_hold_sec:
            return ("pending height estimator/truth disagreement %.3fm "
                    "(%.2f/%.2fs)" %
                    (error, held, self.gazebo_truth_hold_sec))
        return ("height estimator/truth disagreement %.3fm for %.2fs "
                "(MAVROS AGL %.3fm, Gazebo AGL %.3fm)" %
                (error, held, estimator_agl, truth_agl))

    def _healthy(self):
        return self._health_failure() is None

    def _planner_ready(self):
        now = rospy.Time.now()
        return (self.planner_state == self.PLANNER_WAIT_TRIGGER and
                not self.last_planner_state.is_zero() and
                0.0 <= (now - self.last_planner_state).to_sec() <= 1.0)

    def _flight_control_ready(self):
        return self.fcu.connected and self.fcu.armed and self.fcu.mode == "OFFBOARD"

    def _set_navigation_enabled(self, enabled):
        enabled = bool(enabled)
        if enabled == self.navigation_enabled:
            return
        self.navigation_enabled = enabled
        self.last_vlm_health = rospy.Time(0)
        self.last_semantic = rospy.Time(0)
        self.semantic_target = ""
        self.navigation_enabled_pub.publish(Bool(data=enabled))
        rospy.logwarn("VLM navigation %s", "enabled" if enabled else "disabled")

    def _set_mapping_enabled(self, enabled):
        enabled = bool(enabled)
        if enabled == self.mapping_enabled:
            return
        self.mapping_enabled = enabled
        if enabled:
            self.last_mapping_update = rospy.Time(0)
        self.mapping_enabled_pub.publish(Bool(data=enabled))
        rospy.logwarn("Depth mapping %s", "enabled" if enabled else "disabled")

    def _start(self, req):
        label = req.target_label.strip()
        if not label:
            return StartMissionResponse(False, "target_label is required")
        if self.state not in (MissionState.WAIT_FCU, MissionState.DISARMED):
            return StartMissionResponse(False, "another mission is active")
        if self.fcu.armed:
            return StartMissionResponse(False, "vehicle must be disarmed before mission start")
        if self.planner_state == self.PLANNER_FINISH:
            return StartMissionResponse(False, "planner is finished; restart gazebo_planner.launch")
        self.target = label
        self.requested = True
        self.planner_started = False
        self.planner_trigger_time = rospy.Time(0)
        self.planner_phase_since = rospy.Time(0)
        self.planner_fallback_budget_granted = False
        self.planner_execution_since = rospy.Time(0)
        self.last_result = None
        self.cmd = Twist()
        self.trajectory_reference = PositionTarget()
        self.last_cmd = rospy.Time(0)
        self.last_trajectory_reference = rospy.Time(0)
        self.trajectory_reference_epoch = rospy.Time(0)
        self.last_semantic = rospy.Time(0)
        self.semantic_target = ""
        self.hold_pose = None
        self.ground_z = None
        self.cruise_z = None
        self.altitude_stable_since = None
        self.altitude_recovery_active = False
        self.altitude_recovery_since = rospy.Time(0)
        self.altitude_recovery_stable_since = rospy.Time(0)
        self.altitude_hard_since = rospy.Time(0)
        self.landing_target_z = None
        self.landing_last_update = rospy.Time(0)
        self.last_mapping_update = rospy.Time(0)
        # A mission and an FCU/Gazebo epoch must share fresh preflight data.
        # Never carry parameter verification or a truth zero across missions.
        self._reset_px4_height_profile_verification()
        self.gazebo_z = None
        self.gazebo_ground_z = None
        self.last_gazebo_truth = rospy.Time(0)
        self.gazebo_truth_error_since = rospy.Time(0)
        self._set_mapping_enabled(False)
        self._set_navigation_enabled(False)
        self.stop_pub.publish(Empty())
        # Configure the target now; inference remains gated until fixed-height
        # takeoff is stable and HOLD_READY enables navigation.
        self.label_pub.publish(String(data=self.target))
        if self.state == MissionState.DISARMED:
            self._transition(MissionState.WAIT_FCU, "new mission accepted")
        self.detail = "mission accepted; waiting for healthy inputs"
        return StartMissionResponse(True, "mission accepted")

    def _stop(self, _req):
        if self.state == MissionState.DISARMED:
            return TriggerResponse(True, "vehicle is already inactive")
        if self.state == MissionState.WAIT_FCU:
            self.requested = False
            self.target = ""
            self.semantic_target = ""
            self.last_semantic = rospy.Time(0)
            self.cmd = Twist()
            self.last_cmd = rospy.Time(0)
            self._set_navigation_enabled(False)
            self._set_mapping_enabled(False)
            self.stop_pub.publish(Empty())
            self.detail = "pending mission cancelled"
            return TriggerResponse(True, "pending mission cancelled")
        if self.state == MissionState.LAND:
            return TriggerResponse(True, "controlled landing already in progress")
        if not self.fcu.armed:
            self.requested = False
            self.stop_pub.publish(Empty())
            self._transition(MissionState.DISARMED, "mission cancelled before arming")
            return TriggerResponse(True, "mission cancelled before arming")
        self._transition(MissionState.LAND, "operator requested stop")
        return TriggerResponse(True, "controlled landing requested")

    def _latch_hold(self):
        if self.odom is None:
            return
        p = self.odom.pose.pose.position
        q = self.odom.pose.pose.orientation
        yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.hold_pose = (p.x, p.y, p.z, yaw)

    def _capture_ground_reference(self):
        if self.odom is None:
            return False
        self.ground_z = self.odom.pose.pose.position.z
        self.cruise_z = self.ground_z + self.altitude
        return True

    def _relative_altitude(self):
        if self.odom is None:
            return float("nan")
        if self.ground_z is None:
            return 0.0
        # Mission altitude is height above the recorded launch level. PX4's
        # local origin can reset or drift during AUTO.LAND, so an AGL status
        # value must not become negative after touchdown.
        return max(0.0, self.odom.pose.pose.position.z - self.ground_z)

    def _altitude_guard(self, now):
        """Return a fault reason, or recover progressively to cruise altitude."""
        if self.odom is None or self.cruise_z is None:
            return None

        signed_error = self.odom.pose.pose.position.z - self.cruise_z
        error = abs(signed_error)
        if error >= self.altitude_hard_error:
            if self.altitude_hard_since.is_zero():
                self.altitude_hard_since = now
            held = (now - self.altitude_hard_since).to_sec()
            if held >= self.altitude_hard_hold_sec:
                return ("persistent altitude deviation %.3fm >= %.3fm for %.2fs" %
                        (error, self.altitude_hard_error, held))
        else:
            self.altitude_hard_since = rospy.Time(0)

        if error > self.altitude_soft_error and not self.altitude_recovery_active:
            self.altitude_recovery_active = True
            self.altitude_recovery_since = now
            self.altitude_recovery_stable_since = rospy.Time(0)
            self._latch_hold()

        if not self.altitude_recovery_active:
            return None

        recovery_age = ((now - self.altitude_recovery_since).to_sec()
                        if not self.altitude_recovery_since.is_zero() else 0.0)
        if recovery_age >= self.altitude_recovery_timeout:
            return ("altitude recovery timed out (error %.3fm, %.2fs >= %.2fs)" %
                    (error, recovery_age, self.altitude_recovery_timeout))

        if error <= self.altitude_recovery_error:
            if self.altitude_recovery_stable_since.is_zero():
                self.altitude_recovery_stable_since = now
            elif ((now - self.altitude_recovery_stable_since).to_sec() >=
                  self.recovery_stable_sec):
                self.altitude_recovery_active = False
                self.altitude_recovery_since = rospy.Time(0)
                self.altitude_recovery_stable_since = rospy.Time(0)
                return None
        else:
            self.altitude_recovery_stable_since = rospy.Time(0)

        direction = "above" if signed_error > 0.0 else "below"
        self.detail = ("altitude recovery: %.3fm %s cruise; holding XY" %
                       (error, direction))
        return "recovering"

    def _setpoint(self, auto=False, takeoff=False, landing=False):
        if self.odom is None:
            return
        if self.hold_pose is None:
            self._latch_hold()
        msg = PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        ignore_accel = (PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY |
                        PositionTarget.IGNORE_AFZ)
        # PX4's native position controller is the only Z feedback loop.  A
        # supervisor-generated VZ from the same position error would be added
        # to PX4's own position-to-velocity loop and amplify estimator error.
        msg.velocity.z = 0.0
        vertical_velocity_mask = PositionTarget.IGNORE_VZ
        if (auto and self.tracking_mode == "px4_native" and
                not self._trajectory_reference_is_finite(self.trajectory_reference)):
            # The callback is the first boundary; retain this final FCU
            # boundary in case a reference is changed outside that callback.
            auto = False
        if auto:
            if self.tracking_mode == "px4_native":
                ref = self.trajectory_reference
                msg.position.x = ref.position.x
                msg.position.y = ref.position.y
                vx, vy = ref.velocity.x, ref.velocity.y
                speed = math.hypot(vx, vy)
                if speed > self.max_speed and speed > 1e-6:
                    scale = self.max_speed / speed
                    vx, vy = vx * scale, vy * scale
                msg.velocity.x, msg.velocity.y = vx, vy
                msg.position.z = (self.cruise_z if self.cruise_z is not None else
                                  self.altitude)
                msg.yaw = ref.yaw
                msg.yaw_rate = max(
                    -self.max_yaw_rate, min(self.max_yaw_rate, ref.yaw_rate))
                # PX4 receives XY position feedback plus world-frame velocity
                # feed-forward. Z remains an independent position hold at the
                # mission's fixed cruise reference.
                msg.type_mask = vertical_velocity_mask | ignore_accel
            else:
                q = self.odom.pose.pose.orientation
                yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
                forward = max(-self.max_speed, min(self.max_speed, self.cmd.linear.x))
                msg.velocity.x = forward * math.cos(yaw)
                msg.velocity.y = forward * math.sin(yaw)
                msg.position.z = self.cruise_z if self.cruise_z is not None else self.altitude
                msg.yaw_rate = max(
                    -self.max_yaw_rate, min(self.max_yaw_rate, self.cmd.angular.z))
                msg.type_mask = (PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY |
                                 vertical_velocity_mask | ignore_accel |
                                 PositionTarget.IGNORE_YAW)
        else:
            x, y, z, yaw = self.hold_pose
            msg.position.x, msg.position.y = x, y
            # In flight, every hold mode must converge to the fixed cruise
            # reference. Latching the measured Z at each replan turns normal
            # controller error into a permanent, accumulating altitude offset.
            if landing and self.landing_target_z is not None:
                msg.position.z = self.landing_target_z
            else:
                msg.position.z = (self.cruise_z if self.cruise_z is not None else
                                  (self.altitude if takeoff else z))
            msg.yaw = yaw
            msg.type_mask = (PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY |
                             vertical_velocity_mask | ignore_accel |
                             PositionTarget.IGNORE_YAW_RATE)
        if not self._setpoint_is_finite(msg):
            # Do not emit NaN/Inf to PX4 even if a caller bypasses the normal
            # callback path.  Preserve the fixed-PZ/IGNORE_VZ hold contract.
            self._safe_hold_setpoint(msg, ignore_accel, vertical_velocity_mask)
        self.setpoint_pub.publish(msg)

    @staticmethod
    def _setpoint_is_finite(msg):
        return all(math.isfinite(value) for value in (
            msg.position.x, msg.position.y, msg.position.z,
            msg.velocity.x, msg.velocity.y, msg.velocity.z,
            msg.yaw, msg.yaw_rate))

    def _safe_hold_setpoint(self, msg, ignore_accel, vertical_velocity_mask):
        hold = self.hold_pose if self.hold_pose is not None else (0.0, 0.0, 0.0, 0.0)
        x, y, z, yaw = (value if math.isfinite(value) else 0.0 for value in hold)
        cruise_z = self.cruise_z if self.cruise_z is not None else self.altitude
        msg.position.x, msg.position.y = x, y
        msg.position.z = cruise_z if math.isfinite(cruise_z) else z
        msg.velocity.x = 0.0
        msg.velocity.y = 0.0
        msg.velocity.z = 0.0
        msg.yaw = yaw
        msg.yaw_rate = 0.0
        msg.type_mask = (PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY |
                         vertical_velocity_mask | ignore_accel |
                         PositionTarget.IGNORE_YAW_RATE)

    def _call_limited(self, function, *args, **kwargs):
        now = rospy.Time.now()
        elapsed = (now - self.last_service_attempt).to_sec()
        if 0.0 <= elapsed < 1.0:
            return None
        self.last_service_attempt = now
        try:
            return function(*args, **kwargs)
        except rospy.ServiceException as exc:
            rospy.logwarn_throttle(2.0, "MAVROS service failed: %s", exc)
            return None

    def _handle_clock_rollback(self, now):
        """Reset time-derived state after a global ROS clock regression.

        Returns True when the caller must wait for the next tick before
        evaluating the state machine.  LAND deliberately returns False so its
        mode/disarm services continue immediately in the new clock epoch.
        """
        previous_tick = getattr(self, "last_tick_time", rospy.Time(0))
        self.last_tick_time = now
        if previous_tick.is_zero() or now >= previous_tick:
            return False

        rospy.logwarn("ROS clock reset from %.3f to %.3f; clearing mission time epoch",
                      previous_tick.to_sec(), now.to_sec())
        self.state_since = now
        self.planner_state_since = now if self.planner_state is not None else rospy.Time(0)
        self.planner_phase_since = (
            now if self.planner_state == self.PLANNER_PLAN_TRAJ else rospy.Time(0))
        self.planner_execution_since = (
            now if self.planner_state in self.PLANNER_EXECUTION_STATES else rospy.Time(0))
        self.planner_trigger_time = now if self.planner_started else rospy.Time(0)
        self.trajectory_reference_epoch = now
        self.trajectory_reference = PositionTarget()
        self.last_trajectory_reference = rospy.Time(0)
        self.last_cmd = rospy.Time(0)
        self.last_result = None
        self.last_planner_state = rospy.Time(0)
        self.last_service_attempt = rospy.Time(0)
        self.landing_last_update = now

        # Input freshness cannot be carried across a clock epoch.  Keep the
        # current hold/altitude geometry, but require new source samples before
        # any future AUTO execution can resume.
        self.last_odom = rospy.Time(0)
        self.last_extended_state = rospy.Time(0)
        self.last_odom_sample_time = rospy.Time(0)
        self.last_odom_z = None
        self.vertical_speed = 0.0
        self.last_rgb = rospy.Time(0)
        self.last_depth = rospy.Time(0)
        self.last_vision_height = rospy.Time(0)
        self.last_mapping_update = rospy.Time(0)
        self.last_vlm_health = rospy.Time(0)
        self.last_semantic = rospy.Time(0)
        self.semantic_target = ""
        self.last_gazebo_truth = rospy.Time(0)
        self.gazebo_z = None
        self.gazebo_ground_z = None
        self.gazebo_truth_error_since = rospy.Time(0)
        self.altitude_stable_since = None
        self.altitude_recovery_since = rospy.Time(0)
        self.altitude_recovery_stable_since = rospy.Time(0)
        self.altitude_hard_since = rospy.Time(0)

        if self.state == MissionState.AUTO and self.fcu.armed:
            self._transition(MissionState.FAULT,
                             "ROS clock reset during AUTO; landing safely")
            return True

        # Revoke any streaming trajectory even for non-AUTO states.  LAND is
        # intentionally allowed to continue below so AUTO.LAND can be sent in
        # the first tick of the new clock epoch.
        self.stop_pub.publish(Empty())
        if self.state == MissionState.LAND:
            return False
        return True

    @staticmethod
    def _service_succeeded(response, field):
        return response is not None and bool(getattr(response, field, False))

    def _tick(self, _event):
        now = rospy.Time.now()
        if self._handle_clock_rollback(now):
            return
        if self.state == MissionState.WAIT_FCU:
            if not self.requested:
                self.detail = "waiting for mission request"
                return
            health_failure = self._flight_inputs_failure()
            if health_failure is not None:
                self.detail = "waiting for " + health_failure
                return
            profile_failure = self._px4_height_profile_failure()
            if profile_failure is not None:
                self.detail = "waiting for " + profile_failure
                return
            perception_failure = self._perception_servers_failure(now)
            if perception_failure is not None:
                self.detail = "waiting for " + perception_failure
                return
            if (self.gazebo_truth_guard_enabled and
                    self.gazebo_ground_z is None):
                self.detail = ("waiting for Gazebo model '%s' ground height" %
                               self.gazebo_model_name)
                return
            if (self.gazebo_truth_guard_enabled and
                    not self._fresh(now, self.last_gazebo_truth,
                                    self.gazebo_truth_timeout)):
                self.detail = ("waiting for fresh Gazebo model '%s' height" %
                               self.gazebo_model_name)
                return
            if self.ground_z is None and not self._capture_ground_reference():
                self.detail = "waiting for takeoff altitude reference"
                return
            if not self._planner_ready():
                self.detail = "waiting for planner WAIT_TRIGGER"
                return
            self._latch_hold()
            self._transition(MissionState.PRESTREAM, "prestreaming offboard setpoints")
            return
        if self.state == MissionState.PRESTREAM:
            if self._flight_inputs_failure() is not None or not self._planner_ready():
                self._transition(MissionState.FAULT, "critical input stale before arming")
                return
            self._setpoint(takeoff=True)
            if (now - self.state_since).to_sec() >= self.prestream_sec:
                self._transition(MissionState.ARM, "arming")
            return
        if self.state == MissionState.ARM:
            self._setpoint(takeoff=True)
            if self._flight_inputs_failure() is not None or not self._planner_ready():
                self._transition(MissionState.FAULT, "critical input stale while arming")
                return
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
            if self._flight_inputs_failure() is not None:
                self._transition(MissionState.FAULT, "critical input stale during takeoff")
                return
            if not self._flight_control_ready():
                self._transition(MissionState.FAULT, "OFFBOARD or arm lost during takeoff")
                return
            current_z = self.odom.pose.pose.position.z
            target_z = self.cruise_z if self.cruise_z is not None else self.altitude
            self._setpoint(takeoff=True)
            height_failure = self._height_consistency_failure(now)
            if height_failure is not None:
                if height_failure.startswith("pending "):
                    self.altitude_stable_since = None
                    self.detail = "verifying " + height_failure[len("pending "):]
                elif "disagreement" in height_failure:
                    self._transition(MissionState.FAULT, height_failure)
                else:
                    self.detail = "waiting for " + height_failure
                return
            altitude_ready = (
                abs(current_z - target_z) <= self.takeoff_ready_altitude_tolerance and
                abs(self.vertical_speed) <= self.takeoff_ready_vertical_speed_tolerance)
            if altitude_ready:
                if self.altitude_stable_since is None:
                    self.altitude_stable_since = now
                elif (now - self.altitude_stable_since).to_sec() >= self.alt_stable_sec:
                    # Do not command an initial yaw sweep: it was the only
                    # persistent source of pre-navigation altitude loss in
                    # the recorded SITL runs. Enter ordinary fixed-Z search;
                    # HOLD_READY enables mapping and waits for its first
                    # healthy output before the planner is triggered.
                    self._latch_hold()
                    self._transition(
                        MissionState.HOLD_READY,
                        "cruise altitude stable; initial scan disabled")
            else:
                self.altitude_stable_since = None
                self.detail = ("settling at cruise altitude: error %.3fm, vertical %.3fm/s" %
                               (abs(current_z - target_z), abs(self.vertical_speed)))
            return
        if self.state == MissionState.HOLD_READY:
            self._setpoint()
            flight_failure = self._flight_inputs_failure()
            if flight_failure is not None:
                self._transition(MissionState.FAULT, flight_failure + " lost before planner start")
                return
            if not self._flight_control_ready():
                self._transition(MissionState.FAULT, "OFFBOARD or arm lost before planner start")
                return
            height_failure = self._height_consistency_failure(now)
            if height_failure is not None:
                if height_failure.startswith("pending "):
                    self.detail = "verifying " + height_failure[len("pending "):]
                elif "disagreement" in height_failure:
                    self._transition(MissionState.FAULT, height_failure)
                else:
                    self.detail = "waiting for " + height_failure
                return
            navigation_failure = self._health_failure()
            if navigation_failure is not None:
                self.detail = "at cruise altitude; waiting for " + navigation_failure
                if (now - self.state_since).to_sec() > self.navigation_ready_timeout:
                    self._transition(MissionState.FAULT, "navigation pipeline readiness timed out")
                return
            if not self.planner_started and self._planner_ready():
                goal = PoseStamped()
                goal.header.stamp = now
                goal.header.frame_id = self.world_frame
                goal.pose.orientation.w = 1.0
                self.trigger_pub.publish(goal)
                self.planner_started = True
                self.planner_trigger_time = now
                # The mission-start stop command produces one zero Twist. It
                # is not a planner command and must not defeat the AUTO startup
                # grace period several seconds later.
                self.cmd = Twist()
                self.last_cmd = rospy.Time(0)
                self.detail = "planner trigger sent; waiting for acknowledgement"
                return
            if not self.planner_started:
                if (now - self.state_since).to_sec() > self.planner_ready_timeout:
                    self._transition(MissionState.FAULT, "planner did not reach WAIT_TRIGGER")
                return
            if self.last_result is not None and self._handle_planner_result(self.last_result):
                return
            if self.planner_state in self.PLANNER_ACTIVE_STATES:
                self._transition(MissionState.AUTO, "planner active")
                return
            if (now - self.planner_trigger_time).to_sec() > self.planner_ack_timeout:
                self._transition(MissionState.FAULT, "planner trigger acknowledgement timed out")
            return
        if self.state == MissionState.AUTO:
            height_failure = self._height_consistency_failure(now)
            if (height_failure is not None and
                    not height_failure.startswith("pending ")):
                self._transition(MissionState.FAULT, height_failure)
                return
            health_failure = self._health_failure()
            if health_failure is not None:
                self._transition(MissionState.FAULT, health_failure)
                return
            if not self._flight_control_ready():
                self._transition(MissionState.FAULT, "OFFBOARD or arm lost")
                return
            altitude_status = self._altitude_guard(now)
            if altitude_status is not None:
                if altitude_status != "recovering":
                    self._transition(MissionState.FAULT, altitude_status)
                    return
                if self.planner_state != self.PLANNER_PLAN_TRAJ:
                    self._setpoint()
                    return
                # PLAN_TRAJ already holds XY and commands cruise_z, so height
                # recovery and the bounded planning watchdog can run together.
                # Do not let an altitude recovery silently suspend the total
                # planning deadline.
            command_from_this_auto_period = (
                not self.last_cmd.is_zero() and self.last_cmd >= self.state_since)
            # PLAN_TRAJ intentionally has no executable command. Hold the
            # current pose while it plans and apply a bounded planning timeout;
            # command_timeout_sec only governs EXEC_TRAJ/REPLAN streaming.
            if self.planner_state == self.PLANNER_PLAN_TRAJ:
                initial_plan = not command_from_this_auto_period
                timeout = (self.planner_start_timeout if initial_plan
                           else self.planner_replan_timeout)
                total_age = ((now - self.planner_state_since).to_sec()
                             if not self.planner_state_since.is_zero()
                             else (now - self.state_since).to_sec())
                phase_since = (self.planner_phase_since
                               if not self.planner_phase_since.is_zero()
                               else self.planner_state_since)
                phase_age = ((now - phase_since).to_sec()
                             if not phase_since.is_zero() else total_age)
                total_timeout = (self.planner_start_timeout if initial_plan else
                                 self.planner_replan_total_timeout)
                if phase_age <= timeout and total_age <= total_timeout:
                    if initial_plan:
                        planning_detail = "waiting for initial trajectory"
                    elif self.planner_fallback_budget_granted:
                        planning_detail = ("exploration fallback planning; holding position "
                                           "(phase %.2f/%.2fs, total %.2f/%.2fs)" %
                                           (phase_age, timeout, total_age, total_timeout))
                    else:
                        planning_detail = ("planner replanning; holding position "
                                           "(phase %.2f/%.2fs, total %.2f/%.2fs)" %
                                           (phase_age, timeout, total_age, total_timeout))
                    if altitude_status == "recovering":
                        self.detail = "altitude recovery during " + planning_detail
                    else:
                        self.detail = planning_detail
                    self._setpoint()
                    return
                phase = "initial planning" if initial_plan else "replanning"
                limit_name = "phase" if phase_age > timeout else "total"
                age = phase_age if limit_name == "phase" else total_age
                limit = timeout if limit_name == "phase" else total_timeout
                self._transition(
                    MissionState.FAULT,
                    "%s %s timed out (%.2fs > %.2fs)" %
                    (phase, limit_name, age, limit))
                return
            if (not command_from_this_auto_period and
                    (now - self.state_since).to_sec() <= self.planner_start_timeout):
                self._setpoint()
                return

            # A newly published trajectory and its first cmd_vel arrive on
            # separate ROS callbacks.  Reject commands left over from the
            # previous trajectory, but hold position for one command timeout
            # after PLAN_TRAJ -> EXEC_TRAJ instead of faulting in that callback
            # race window.
            execution_epoch = self.planner_execution_since
            if execution_epoch.is_zero():
                execution_epoch = self.planner_state_since
            if execution_epoch.is_zero() or execution_epoch < self.state_since:
                execution_epoch = self.state_since
            command_from_current_execution = (
                not self.last_cmd.is_zero() and self.last_cmd >= execution_epoch)
            if self.tracking_mode == "px4_native":
                reference_epoch = self._trajectory_reference_execution_epoch()
                command_from_current_execution = (
                    not self.last_trajectory_reference.is_zero() and
                    (reference_epoch.is_zero() or
                     self.last_trajectory_reference >= reference_epoch))
                if (command_from_current_execution and
                        not self._fresh(now, self.last_trajectory_reference,
                                        self.cmd_timeout)):
                    reference_age = (now - self.last_trajectory_reference).to_sec()
                    self._transition(
                        MissionState.FAULT,
                        "trajectory reference timeout in state %s "
                        "(source age %.3fs > %.3fs)" %
                        (self.planner_state, reference_age, self.cmd_timeout))
                    return
            execution_age = (now - execution_epoch).to_sec()
            if (not command_from_current_execution and
                    execution_age <= self.first_reference_timeout):
                self.detail = "waiting for first command from new trajectory; holding position"
                self._setpoint()
                return
            if not command_from_current_execution:
                stale_age = ((now - self.last_cmd).to_sec()
                             if not self.last_cmd.is_zero() else float("inf"))
                self._transition(
                    MissionState.FAULT,
                    "planner first command timeout in state %s "
                    "(wait %.3fs > %.3fs; previous command age %.3fs)" %
                    (self.planner_state, execution_age,
                     self.first_reference_timeout, stale_age))
                return
            if (now - self.last_cmd).to_sec() > self.cmd_timeout:
                cmd_age = ((now - self.last_cmd).to_sec()
                           if not self.last_cmd.is_zero() else float("inf"))
                self._transition(
                    MissionState.FAULT,
                    "planner command timeout in state %s (age %.3fs > %.3fs)" %
                    (self.planner_state, cmd_age, self.cmd_timeout))
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
            if not self.fcu.armed:
                self.requested = False
                self.planner_started = False
                self._transition(MissionState.DISARMED, "landed and disarmed")
                return
            if (self._fresh(now, self.last_extended_state,
                            self.extended_state_timeout) and
                    self.extended.landed_state ==
                    ExtendedState.LANDED_STATE_ON_GROUND):
                self._call_limited(self.arm, False)
                return

            # Normal termination remains in OFFBOARD and descends a bounded
            # position target.  With EKF2 external-vision height this is a
            # genuine closed loop on Gazebo Z and avoids PX4's comparatively
            # fast default land descent.  Loss of any prerequisite falls back
            # to the preflight-verified PX4 AUTO.LAND path.
            inputs_fresh = (
                self.odom is not None and
                self._fresh(now, self.last_odom, self.odom_timeout) and
                (self.px4_expected_height_reference != 3 or
                 self._fresh(now, self.last_vision_height,
                             self.vision_timeout)) and
                (not self.gazebo_truth_guard_enabled or
                 self._fresh(now, self.last_gazebo_truth,
                             self.gazebo_truth_timeout)))
            land_age = (now - self.state_since).to_sec()
            if (self.fcu.mode == "OFFBOARD" and inputs_fresh and
                    0.0 <= land_age <= self.controlled_land_timeout):
                if self.landing_target_z is None:
                    self.landing_target_z = self.odom.pose.pose.position.z
                dt = (now - self.landing_last_update).to_sec()
                if dt < 0.0 or dt > 0.5:
                    dt = 0.0
                floor_z = (self.ground_z if self.ground_z is not None else
                           self.landing_target_z)
                self.landing_target_z = max(
                    floor_z,
                    self.landing_target_z - self.controlled_land_speed * dt)
                self.landing_last_update = now
                self.detail = ("controlled landing %.2fm/s; target AGL %.2fm" %
                               (self.controlled_land_speed,
                                max(0.0, self.landing_target_z - floor_z)))
                self._setpoint(landing=True)
                return

            if self.fcu.mode != "AUTO.LAND":
                reason = ("controlled landing unavailable" if not inputs_fresh
                          else "controlled landing timeout")
                self.detail = reason + "; PX4 AUTO.LAND fallback"
                self._call_limited(self.set_mode, custom_mode="AUTO.LAND")

    def _publish_state(self, _event):
        msg = MissionState()
        msg.header.stamp = rospy.Time.now()
        msg.state = self.state
        msg.target_label = self.target
        msg.detail = self.detail
        msg.fcu_connected = self.fcu.connected
        msg.armed = self.fcu.armed
        msg.px4_mode = self.fcu.mode
        msg.altitude = self._relative_altitude()
        msg.planner_active = self.state == MissionState.AUTO
        self.state_pub.publish(msg)


if __name__ == "__main__":
    Supervisor()
    rospy.spin()
