#!/usr/bin/env python3

import importlib.util
import math
from pathlib import Path
import unittest
from unittest import mock

import rospy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Pose, PoseStamped, Twist
from gazebo_msgs.msg import ModelStates
from mavros_msgs.msg import ExtendedState, PositionTarget, State
from nav_msgs.msg import Odometry
from plan_env.msg import MissionState, SemanticObservation
from sensor_msgs.msg import Image
from std_msgs.msg import Header, Int32


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "px4_mission_supervisor.py"
SPEC = importlib.util.spec_from_file_location("px4_mission_supervisor", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
Supervisor = MODULE.Supervisor


class PX4MissionSupervisorTest(unittest.TestCase):
    def setUp(self):
        self.supervisor = Supervisor.__new__(Supervisor)
        self.supervisor.navigation_enabled = False
        self.supervisor.navigation_enabled_pub = mock.Mock()
        self.supervisor.mapping_enabled = False
        self.supervisor.mapping_enabled_pub = mock.Mock()
        self.supervisor.altitude = 1.0
        self.supervisor.ground_z = 0.0
        self.supervisor.cruise_z = 1.0
        self.supervisor.odom = None
        self.supervisor.planner_state = None
        self.supervisor.planner_started = False
        self.supervisor.planner_state_since = rospy.Time(0)
        self.supervisor.planner_phase_since = rospy.Time(0)
        self.supervisor.planner_fallback_budget_granted = False
        self.supervisor.planner_execution_since = rospy.Time(0)
        self.supervisor.planner_replan_timeout = 3.0
        self.supervisor.planner_replan_total_timeout = 10.0
        self.supervisor.cmd_timeout = 0.3
        self.supervisor.first_reference_timeout = 0.8
        self.supervisor.tracking_mode = "legacy_mpc"
        self.supervisor.max_reference_position_jump = 0.20
        self.supervisor.trajectory_reference = PositionTarget()
        self.supervisor.last_cmd = rospy.Time(0)
        self.supervisor.last_trajectory_reference = rospy.Time(0)
        self.supervisor.trajectory_reference_epoch = rospy.Time(0)
        self.supervisor.last_tick_time = rospy.Time(0)
        self.supervisor.altitude_soft_error = 0.30
        self.supervisor.altitude_recovery_error = 0.15
        self.supervisor.altitude_hard_error = 0.60
        self.supervisor.altitude_hard_hold_sec = 2.0
        self.supervisor.altitude_recovery_timeout = 8.0
        self.supervisor.recovery_stable_sec = 1.0
        self.supervisor.altitude_recovery_active = False
        self.supervisor.altitude_recovery_since = rospy.Time(0)
        self.supervisor.altitude_recovery_stable_since = rospy.Time(0)
        self.supervisor.altitude_hard_since = rospy.Time(0)
        self.supervisor.vertical_speed = 0.0
        self.supervisor.last_odom_z = None
        self.supervisor.last_odom_sample_time = rospy.Time(0)
        self.supervisor.takeoff_ready_altitude_tolerance = 0.05
        self.supervisor.takeoff_ready_vertical_speed_tolerance = 0.025
        self.supervisor.px4_height_profile_check_enabled = False
        self.supervisor.px4_height_profile_verified = False
        self.supervisor.last_px4_profile_check = rospy.Time(0)
        self.supervisor.cached_px4_profile_failure = None
        self.supervisor.gazebo_truth_guard_enabled = False
        self.supervisor.gazebo_truth_timeout = 0.50
        self.supervisor.last_gazebo_truth = rospy.Time(0)
        self.supervisor.last_extended_state = rospy.Time(0)
        self.supervisor.extended_state_timeout = 2.0
        self.supervisor.controlled_land_speed = 0.25
        self.supervisor.controlled_land_timeout = 10.0
        self.supervisor.landing_target_z = None
        self.supervisor.landing_last_update = rospy.Time(0)
        self.supervisor.last_odom = rospy.Time(0)
        self.supervisor.last_rgb = rospy.Time(0)
        self.supervisor.last_depth = rospy.Time(0)
        self.supervisor.last_mapping_update = rospy.Time(0)
        self.supervisor.last_vlm_health = rospy.Time(0)
        self.supervisor.last_semantic = rospy.Time(0)
        self.supervisor.semantic_target = ""
        self.supervisor.world_frame = "map"
        self.supervisor.base_frame = "base_link"
        self.supervisor.camera_frame = "iris_camera_optical_frame"
        self.supervisor.input_future_tolerance = 0.02
        self.supervisor.odom_timeout = 0.20
        self.supervisor.sensor_timeout = 0.50
        self.supervisor.vision_timeout = 0.15
        self.supervisor.last_vision_height = rospy.Time(0)
        self.supervisor.px4_expected_height_reference = 0
        self.supervisor.mapping_update_timeout = 3.0
        self.supervisor.semantic_timeout = 5.0

    @staticmethod
    def _time(seconds):
        return rospy.Time.from_sec(seconds)

    def _native_reference(self, seconds):
        reference = PositionTarget()
        reference.header.stamp = self._time(seconds)
        reference.header.frame_id = self.supervisor.world_frame
        reference.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        return reference

    def test_call_limited_forwards_keyword_arguments(self):
        self.supervisor.last_service_attempt = self._time(0.0)
        calls = []

        def service(*args, **kwargs):
            calls.append((args, kwargs))
            return "accepted"

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(2.0)):
            response = self.supervisor._call_limited(
                service, custom_mode="OFFBOARD")

        self.assertEqual(response, "accepted")
        self.assertEqual(calls, [((), {"custom_mode": "OFFBOARD"})])
        self.assertEqual(self.supervisor.last_service_attempt, self._time(2.0))

    def test_call_limited_retries_immediately_after_clock_regression(self):
        self.supervisor.last_service_attempt = self._time(10.0)
        service = mock.Mock(return_value="accepted")

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(2.0)):
            response = self.supervisor._call_limited(service)

        self.assertEqual(response, "accepted")
        service.assert_called_once_with()
        self.assertEqual(self.supervisor.last_service_attempt, self._time(2.0))

    def test_arm_state_requests_offboard(self):
        sup = self.supervisor
        sup.state = MissionState.ARM
        sup.state_since = self._time(1.0)
        sup.arm_transition_timeout = 10.0
        sup.last_service_attempt = self._time(0.0)
        sup.fcu = State(connected=True, armed=False, mode="AUTO.LOITER")
        sup._healthy = mock.Mock(return_value=True)
        sup._flight_inputs_failure = mock.Mock(return_value=None)
        sup._planner_ready = mock.Mock(return_value=True)
        sup._setpoint = mock.Mock()
        sup._transition = mock.Mock()
        calls = []

        def set_mode(*args, **kwargs):
            calls.append((args, kwargs))
            return type("Response", (), {"mode_sent": True})()

        sup.set_mode = set_mode
        sup.arm = mock.Mock()
        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(3.0)):
            sup._tick(None)

        self.assertEqual(calls, [((), {"custom_mode": "OFFBOARD"})])
        sup.arm.assert_not_called()
        sup._transition.assert_not_called()

    def test_waiting_stop_cancels_pending_mission(self):
        sup = self.supervisor
        sup.state = MissionState.WAIT_FCU
        sup.requested = True
        sup.target = "chair"
        sup.semantic_target = "chair"
        sup.last_semantic = self._time(4.0)
        sup.cmd = object()
        sup.last_cmd = self._time(4.0)
        sup.stop_pub = mock.Mock()

        response = sup._stop(None)

        self.assertTrue(response.success)
        self.assertFalse(sup.requested)
        self.assertEqual(sup.target, "")
        self.assertEqual(sup.semantic_target, "")
        self.assertTrue(sup.last_semantic.is_zero())
        self.assertTrue(sup.last_cmd.is_zero())
        sup.stop_pub.publish.assert_called_once()

    def test_planner_result_is_handled_before_auto_transition(self):
        sup = self.supervisor
        sup.state = MissionState.HOLD_READY
        sup.planner_started = True
        sup.last_result = None
        sup._transition = mock.Mock()

        sup._result_cb(Int32(data=4))

        self.assertEqual(sup.last_result, 4)
        sup._transition.assert_called_once_with(MissionState.HOLD, "target reached")

    def test_hold_ready_waits_for_planner_acknowledgement(self):
        sup = self.supervisor
        sup.state = MissionState.HOLD_READY
        sup.state_since = self._time(5.0)
        sup.planner_ready_timeout = 10.0
        sup.planner_ack_timeout = 2.0
        sup.planner_started = False
        sup.planner_trigger_time = self._time(0.0)
        sup.planner_state = Supervisor.PLANNER_WAIT_TRIGGER
        sup.last_result = None
        sup._healthy = mock.Mock(return_value=True)
        sup._health_failure = mock.Mock(return_value=None)
        sup._flight_inputs_failure = mock.Mock(return_value=None)
        sup._flight_control_ready = mock.Mock(return_value=True)
        sup._planner_ready = mock.Mock(return_value=True)
        sup._setpoint = mock.Mock()
        sup._transition = mock.Mock()
        sup.trigger_pub = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(6.0)):
            sup._tick(None)
        self.assertTrue(sup.planner_started)
        sup.trigger_pub.publish.assert_called_once()
        sup._transition.assert_not_called()

        sup.planner_state = Supervisor.PLANNER_ACTIVE_STATES[0]
        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(6.1)):
            sup._tick(None)
        sup._transition.assert_called_once_with(MissionState.AUTO, "planner active")

    def test_wait_fcu_requires_ready_perception_servers_before_takeoff(self):
        sup = self.supervisor
        sup.state = MissionState.WAIT_FCU
        sup.detail = "waiting"
        sup.requested = True
        sup._flight_inputs_failure = mock.Mock(return_value=None)
        sup._perception_servers_failure = mock.Mock(
            return_value="ready YOLOE/CLIPITM servers")
        sup._planner_ready = mock.Mock(return_value=True)
        sup._latch_hold = mock.Mock()
        sup._transition = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(1.0)):
            sup._tick(None)

        self.assertIn("ready YOLOE/CLIPITM servers", sup.detail)
        sup._transition.assert_not_called()

        sup._perception_servers_failure.return_value = None
        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(1.1)):
            sup._tick(None)
        sup._transition.assert_called_once_with(
            MissionState.PRESTREAM, "prestreaming offboard setpoints")

    def test_wait_fcu_refuses_prestream_when_px4_height_profile_is_wrong(self):
        sup = self.supervisor
        sup.state = MissionState.WAIT_FCU
        sup.detail = "waiting"
        sup.requested = True
        sup.px4_height_profile_check_enabled = True
        sup._flight_inputs_failure = mock.Mock(return_value=None)
        sup._px4_height_profile_failure = mock.Mock(
            return_value="PX4 parameter EKF2_HGT_REF=0, expected 3; reboot PX4")
        sup._capture_ground_reference = mock.Mock(return_value=True)
        sup._planner_ready = mock.Mock(return_value=True)
        sup._latch_hold = mock.Mock()
        sup._transition = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(1.0)):
            sup._tick(None)

        self.assertIn("EKF2_HGT_REF=0, expected 3", sup.detail)
        sup._transition.assert_not_called()

    def test_flight_inputs_require_fresh_vision_height_for_vision_profile(self):
        sup = self.supervisor
        sup.px4_expected_height_reference = 3
        sup.fcu = State(connected=True)
        sup.odom = Odometry()
        sup.last_odom = self._time(2.0)
        sup.last_rgb = self._time(2.0)
        sup.last_depth = self._time(2.0)
        pose = PoseStamped()
        pose.header.stamp = self._time(2.0)
        pose.header.frame_id = "map"
        pose.pose.orientation.w = 1.0

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(2.0)):
            self.assertEqual(sup._flight_inputs_failure(),
                             "fresh Gazebo vision height")
            sup._vision_height_cb(pose)
            self.assertIsNone(sup._flight_inputs_failure())

    def test_new_mission_invalidates_cached_px4_height_profile(self):
        sup = self.supervisor
        sup.state = MissionState.WAIT_FCU
        sup.fcu = State(connected=True, armed=False)
        sup.px4_height_profile_verified = True
        sup.last_px4_profile_check = self._time(8.0)
        sup.cached_px4_profile_failure = None
        sup.gazebo_z = 0.8
        sup.gazebo_ground_z = 0.2
        sup.last_gazebo_truth = self._time(8.0)
        sup.gazebo_truth_error_since = self._time(7.0)
        sup.target = ""
        sup.requested = False
        sup.planner_started = False
        sup.planner_trigger_time = rospy.Time(0)
        sup.planner_phase_since = rospy.Time(0)
        sup.planner_fallback_budget_granted = False
        sup.planner_execution_since = rospy.Time(0)
        sup.last_result = None
        sup.cmd = Twist()
        sup.last_cmd = rospy.Time(0)
        sup.last_semantic = rospy.Time(0)
        sup.semantic_target = ""
        sup.hold_pose = None
        sup.ground_z = None
        sup.cruise_z = None
        sup.altitude_stable_since = None
        sup.last_mapping_update = rospy.Time(0)
        sup.stop_pub = mock.Mock()
        sup.label_pub = mock.Mock()
        sup._set_mapping_enabled = mock.Mock()
        sup._set_navigation_enabled = mock.Mock()
        request = type("Request", (), {"target_label": "chair"})()

        response = sup._start(request)

        self.assertTrue(response.accepted)
        self.assertFalse(sup.px4_height_profile_verified)
        self.assertTrue(sup.last_px4_profile_check.is_zero())
        self.assertIsNone(sup.gazebo_z)
        self.assertIsNone(sup.gazebo_ground_z)
        self.assertTrue(sup.last_gazebo_truth.is_zero())
        self.assertTrue(sup.gazebo_truth_error_since.is_zero())

    def test_fcu_disconnect_invalidates_cached_px4_height_profile(self):
        sup = self.supervisor
        sup.fcu = State(connected=True)
        sup.px4_height_profile_verified = True
        sup.last_px4_profile_check = self._time(8.0)
        sup.cached_px4_profile_failure = None

        sup._fcu_cb(State(connected=False))

        self.assertFalse(sup.px4_height_profile_verified)
        self.assertTrue(sup.last_px4_profile_check.is_zero())

    def test_px4_height_profile_reads_all_required_parameters(self):
        sup = self.supervisor
        sup.px4_height_profile_check_enabled = True
        sup.px4_expected_height_reference = 3
        sup.px4_expected_ev_control = 2
        sup.px4_expected_baro_noise = 3.5
        sup.px4_expected_hover_thrust = 0.70
        sup.px4_expected_land_speed = 0.60
        sup.px4_expected_offboard_loss_action = 4
        sup.px4_expected_rc_input_mode = 4
        sup.px4_expected_offboard_loss_timeout = 1.0
        sup.px4_profile_tolerance = 0.02
        values = {
            "EKF2_HGT_REF": (3, 0.0),
            "EKF2_EV_CTRL": (2, 0.0),
            "EKF2_BARO_NOISE": (0, 3.5),
            "MPC_THR_HOVER": (0, 0.70),
            "MPC_LAND_SPEED": (0, 0.60),
            "COM_OBL_RC_ACT": (4, 0.0),
            "COM_RC_IN_MODE": (4, 0.0),
            "COM_OF_LOSS_T": (0, 1.0),
        }
        calls = []

        def param_get(param_id):
            calls.append(param_id)
            integer, real = values[param_id]
            value = type("Value", (), {"integer": integer, "real": real})()
            return type("Response", (), {"success": True, "value": value})()

        sup.param_get = param_get
        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(2.0)):
            failure = sup._px4_height_profile_failure()

        self.assertIsNone(failure)
        self.assertTrue(sup.px4_height_profile_verified)
        self.assertEqual(calls, [
            "EKF2_HGT_REF", "EKF2_EV_CTRL", "EKF2_BARO_NOISE",
            "MPC_THR_HOVER", "MPC_LAND_SPEED",
            "COM_OBL_RC_ACT", "COM_RC_IN_MODE", "COM_OF_LOSS_T"])

    def test_wait_fcu_requires_fresh_truth_from_current_mission(self):
        sup = self.supervisor
        sup.state = MissionState.WAIT_FCU
        sup.requested = True
        sup.gazebo_truth_guard_enabled = True
        sup.gazebo_model_name = "iris_depth_camera"
        sup.gazebo_ground_z = 0.2
        sup.last_gazebo_truth = self._time(1.0)
        sup.gazebo_truth_timeout = 0.5
        sup._flight_inputs_failure = mock.Mock(return_value=None)
        sup._px4_height_profile_failure = mock.Mock(return_value=None)
        sup._perception_servers_failure = mock.Mock(return_value=None)
        sup._transition = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(2.0)):
            sup._tick(None)

        self.assertIn("fresh Gazebo model", sup.detail)
        sup._transition.assert_not_called()

    def test_hold_ready_waits_for_post_takeoff_navigation_health(self):
        sup = self.supervisor
        sup.state = MissionState.HOLD_READY
        sup.state_since = self._time(1.0)
        sup.navigation_ready_timeout = 30.0
        sup.planner_started = False
        sup._setpoint = mock.Mock()
        sup._flight_inputs_failure = mock.Mock(return_value=None)
        sup._flight_control_ready = mock.Mock(return_value=True)
        sup._health_failure = mock.Mock(return_value="healthy YOLOE/CLIPITM")
        sup._planner_ready = mock.Mock(return_value=True)
        sup._transition = mock.Mock()
        sup.trigger_pub = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(2.0)):
            sup._tick(None)

        self.assertIn("waiting for healthy YOLOE/CLIPITM", sup.detail)
        sup.trigger_pub.publish.assert_not_called()
        sup._transition.assert_not_called()

    def test_flight_control_ready_requires_connection_arm_and_offboard(self):
        self.supervisor.fcu = State(connected=True, armed=True, mode="OFFBOARD")
        self.assertTrue(self.supervisor._flight_control_ready())
        self.supervisor.fcu.mode = "AUTO.LOITER"
        self.assertFalse(self.supervisor._flight_control_ready())
        self.supervisor.fcu.mode = "OFFBOARD"
        self.supervisor.fcu.armed = False
        self.assertFalse(self.supervisor._flight_control_ready())

    def test_fresh_rejects_clock_regression(self):
        now = self._time(10.0)
        self.assertTrue(Supervisor._fresh(now, self._time(9.9), 0.2))
        self.assertFalse(Supervisor._fresh(now, self._time(10.1), 0.2))
        self.assertFalse(Supervisor._fresh(now, self._time(9.7), 0.2))

    def test_auto_ignores_pre_auto_command_during_startup_grace(self):
        sup = self.supervisor
        sup.state = MissionState.AUTO
        sup.state_since = self._time(10.0)
        sup.last_cmd = self._time(2.0)
        sup.planner_start_timeout = 3.0
        sup.cmd_timeout = 0.3
        sup.first_reference_timeout = 0.8
        sup._health_failure = mock.Mock(return_value=None)
        sup._flight_control_ready = mock.Mock(return_value=True)
        sup._setpoint = mock.Mock()
        sup._transition = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(10.1)):
            sup._tick(None)

        sup._setpoint.assert_called_once_with()
        sup._transition.assert_not_called()

    def test_auto_holds_during_bounded_replanning_gap(self):
        sup = self.supervisor
        sup.state = MissionState.AUTO
        sup.state_since = self._time(10.0)
        sup.planner_state = 2
        sup.planner_state_since = self._time(12.0)
        sup.planner_phase_since = self._time(12.0)
        sup.last_cmd = self._time(11.9)
        sup.planner_start_timeout = 3.0
        sup.planner_replan_timeout = 3.0
        sup.cmd_timeout = 0.3
        sup._health_failure = mock.Mock(return_value=None)
        sup._flight_control_ready = mock.Mock(return_value=True)
        sup._setpoint = mock.Mock()
        sup._transition = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(14.2)):
            sup._tick(None)

        sup._setpoint.assert_called_once_with()
        sup._transition.assert_not_called()
        self.assertIn("replanning", sup.detail)

    def test_replanning_latches_current_pose_for_hold(self):
        sup = self.supervisor
        sup.state = MissionState.AUTO
        sup.planner_state = 4
        sup._latch_hold = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(12.0)):
            sup._planner_state_cb(Int32(data=2))

        self.assertEqual(sup.planner_state_since, self._time(12.0))
        self.assertEqual(sup.planner_phase_since, self._time(12.0))
        self.assertFalse(sup.planner_fallback_budget_granted)
        self.assertTrue(sup.planner_execution_since.is_zero())
        sup._latch_hold.assert_called_once_with()

    def test_object_to_exploration_fallback_gets_one_fresh_phase_budget(self):
        sup = self.supervisor
        sup.state = MissionState.AUTO
        sup.planner_started = True
        sup.planner_state = Supervisor.PLANNER_PLAN_TRAJ
        sup.last_result = Supervisor.RESULT_SEARCH_OBJECT
        sup.planner_phase_since = self._time(12.0)
        sup._handle_planner_result = mock.Mock(return_value=False)

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(15.0)):
            sup._result_cb(Int32(data=Supervisor.RESULT_EXPLORE))

        self.assertEqual(sup.planner_phase_since, self._time(15.0))
        self.assertTrue(sup.planner_fallback_budget_granted)

        sup.last_result = Supervisor.RESULT_SEARCH_OBJECT
        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(16.0)):
            sup._result_cb(Int32(data=Supervisor.RESULT_EXPLORE))
        self.assertEqual(sup.planner_phase_since, self._time(15.0))

    def test_new_trajectory_rejects_old_command_but_holds_during_first_command_grace(self):
        sup = self.supervisor
        sup.state = MissionState.AUTO
        sup.state_since = self._time(10.0)
        sup.planner_state = Supervisor.PLANNER_WAIT_TRIGGER
        sup.last_cmd = self._time(20.491)
        sup.planner_start_timeout = 3.0
        sup.cmd_timeout = 0.3
        sup.first_reference_timeout = 0.8
        sup._health_failure = mock.Mock(return_value=None)
        sup._flight_control_ready = mock.Mock(return_value=True)
        sup._setpoint = mock.Mock()
        sup._transition = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(22.092)):
            sup._planner_state_cb(Int32(data=Supervisor.PLANNER_EXECUTION_STATES[0]))
        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(22.592)):
            sup._tick(None)

        self.assertEqual(sup.planner_execution_since, self._time(22.092))
        self.assertIn("waiting for first command", sup.detail)
        sup._setpoint.assert_called_once_with()
        sup._transition.assert_not_called()

    def test_new_trajectory_accepts_only_command_from_current_execution_epoch(self):
        sup = self.supervisor
        sup.state = MissionState.AUTO
        sup.state_since = self._time(10.0)
        sup.planner_state = Supervisor.PLANNER_EXECUTION_STATES[0]
        sup.planner_state_since = self._time(22.092)
        sup.planner_execution_since = self._time(22.092)
        sup.last_cmd = self._time(22.120)
        sup.planner_start_timeout = 3.0
        sup.cmd_timeout = 0.3
        sup.first_reference_timeout = 0.8
        sup._health_failure = mock.Mock(return_value=None)
        sup._flight_control_ready = mock.Mock(return_value=True)
        sup._setpoint = mock.Mock()
        sup._transition = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(22.130)):
            sup._tick(None)

        sup._setpoint.assert_called_once_with(auto=True)
        sup._transition.assert_not_called()

    def test_new_trajectory_faults_if_first_current_command_misses_grace(self):
        sup = self.supervisor
        sup.state = MissionState.AUTO
        sup.state_since = self._time(10.0)
        sup.planner_state = Supervisor.PLANNER_EXECUTION_STATES[0]
        sup.planner_state_since = self._time(22.092)
        sup.planner_execution_since = self._time(22.092)
        sup.last_cmd = self._time(20.491)
        sup.planner_start_timeout = 3.0
        sup.cmd_timeout = 0.3
        sup.first_reference_timeout = 0.8
        sup._health_failure = mock.Mock(return_value=None)
        sup._flight_control_ready = mock.Mock(return_value=True)
        sup._setpoint = mock.Mock()
        sup._transition = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(22.893)):
            sup._tick(None)

        detail = sup._transition.call_args.args[1]
        self.assertEqual(sup._transition.call_args.args[0], MissionState.FAULT)
        self.assertIn("planner first command timeout", detail)
        self.assertIn("previous command age 2.402s", detail)

    def test_exec_to_replan_keeps_current_execution_epoch(self):
        sup = self.supervisor
        sup.state = MissionState.AUTO
        sup.planner_state = Supervisor.PLANNER_EXECUTION_STATES[0]
        sup.planner_execution_since = self._time(22.092)
        sup._latch_hold = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(22.120)):
            sup._planner_state_cb(Int32(data=Supervisor.PLANNER_EXECUTION_STATES[1]))

        self.assertEqual(sup.planner_execution_since, self._time(22.092))

    def test_auto_faults_when_replanning_exceeds_its_own_timeout(self):
        sup = self.supervisor
        sup.state = MissionState.AUTO
        sup.state_since = self._time(10.0)
        sup.planner_state = 2
        sup.planner_state_since = self._time(12.0)
        sup.planner_phase_since = self._time(12.0)
        sup.last_cmd = self._time(11.9)
        sup.planner_replan_timeout = 3.0
        sup.planner_start_timeout = 3.0
        sup.cmd_timeout = 0.3
        sup._health_failure = mock.Mock(return_value=None)
        sup._flight_control_ready = mock.Mock(return_value=True)
        sup._setpoint = mock.Mock()
        sup._transition = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(15.1)):
            sup._tick(None)

        sup._transition.assert_called_once_with(
            MissionState.FAULT, "replanning phase timed out (3.10s > 3.00s)")

    def test_every_armed_fault_transitions_to_controlled_landing(self):
        sup = self.supervisor
        sup.state = MissionState.FAULT
        sup.state_since = self._time(10.0)
        sup.detail = "planner timed out"
        sup.fault_land_delay = 3.0
        sup._setpoint = mock.Mock()
        sup._transition = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(30.0)):
            sup._tick(None)

        sup._setpoint.assert_called_once_with()
        sup._transition.assert_called_once_with(
            MissionState.LAND, "planner timed out; landing")

    def test_replanning_fallback_is_still_bounded_by_total_timeout(self):
        sup = self.supervisor
        sup.state = MissionState.AUTO
        sup.state_since = self._time(1.0)
        sup.planner_state = Supervisor.PLANNER_PLAN_TRAJ
        sup.planner_state_since = self._time(5.0)
        sup.planner_phase_since = self._time(15.5)
        sup.planner_fallback_budget_granted = True
        sup.last_cmd = self._time(4.9)
        sup.planner_replan_timeout = 5.0
        sup.planner_replan_total_timeout = 10.0
        sup.planner_start_timeout = 3.0
        sup.cmd_timeout = 0.3
        sup._health_failure = mock.Mock(return_value=None)
        sup._flight_control_ready = mock.Mock(return_value=True)
        sup._setpoint = mock.Mock()
        sup._transition = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(16.0)):
            sup._tick(None)

        sup._transition.assert_called_once_with(
            MissionState.FAULT, "replanning total timed out (11.00s > 10.00s)")

    def test_altitude_recovery_does_not_suspend_planning_total_timeout(self):
        sup = self.supervisor
        sup.state = MissionState.AUTO
        sup.state_since = self._time(1.0)
        sup.planner_state = Supervisor.PLANNER_PLAN_TRAJ
        sup.planner_state_since = self._time(5.0)
        sup.planner_phase_since = self._time(15.5)
        sup.planner_fallback_budget_granted = True
        sup.last_cmd = self._time(4.9)
        sup.planner_replan_timeout = 5.0
        sup.planner_replan_total_timeout = 10.0
        sup.planner_start_timeout = 3.0
        sup.cmd_timeout = 0.3
        sup.odom = Odometry()
        sup.odom.pose.pose.orientation.w = 1.0
        sup.odom.pose.pose.position.z = 1.35
        sup.hold_pose = (0.0, 0.0, 1.0, 0.0)
        sup.fcu = State(connected=True, armed=True, mode="OFFBOARD")
        sup._health_failure = mock.Mock(return_value=None)
        sup._flight_control_ready = mock.Mock(return_value=True)
        sup._setpoint = mock.Mock()
        sup._transition = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(16.0)):
            sup._tick(None)

        sup._transition.assert_called_once_with(
            MissionState.FAULT, "replanning total timed out (11.00s > 10.00s)")

    def test_altitude_is_relative_to_launch_level_and_non_negative(self):
        sup = self.supervisor
        sup.odom = Odometry()
        sup.ground_z = -0.2
        sup.odom.pose.pose.position.z = 0.8
        self.assertAlmostEqual(sup._relative_altitude(), 1.0)
        sup.odom.pose.pose.position.z = -3.0
        self.assertEqual(sup._relative_altitude(), 0.0)

    def test_position_hold_uses_cruise_z_instead_of_latched_drift(self):
        sup = self.supervisor
        sup.odom = Odometry()
        sup.hold_pose = (1.0, 2.0, 1.34, 0.25)
        sup.cruise_z = 1.0
        sup.setpoint_pub = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(2.0)):
            sup._setpoint()

        msg = sup.setpoint_pub.publish.call_args.args[0]
        self.assertAlmostEqual(msg.position.x, 1.0)
        self.assertAlmostEqual(msg.position.y, 2.0)
        self.assertAlmostEqual(msg.position.z, 1.0)

    def test_native_trajectory_reference_keeps_cruise_z_and_world_xy_feedforward(self):
        """PX4-native tracking must never inherit the planner's 2-D Z value."""
        sup = self.supervisor
        sup.tracking_mode = "px4_native"
        sup.odom = Odometry()
        sup.odom.pose.pose.orientation.w = 1.0
        sup.odom.pose.pose.position.z = 1.13
        sup.hold_pose = (1.0, 2.0, 0.84, 0.25)
        sup.cruise_z = 1.0
        sup.max_speed = 0.40
        sup.max_yaw_rate = 0.65
        sup.trajectory_reference = PositionTarget()
        sup.trajectory_reference.position.x = 1.25
        sup.trajectory_reference.position.y = 2.50
        sup.trajectory_reference.position.z = -99.0
        sup.trajectory_reference.velocity.x = 0.18
        sup.trajectory_reference.velocity.y = -0.12
        sup.trajectory_reference.velocity.z = -4.0
        sup.trajectory_reference.yaw = -0.40
        sup.trajectory_reference.yaw_rate = 0.30
        sup.setpoint_pub = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(2.0)):
            sup._setpoint(auto=True)

        msg = sup.setpoint_pub.publish.call_args.args[0]
        self.assertAlmostEqual(msg.position.x, 1.25)
        self.assertAlmostEqual(msg.position.y, 2.50)
        self.assertAlmostEqual(msg.position.z, 1.0)
        self.assertAlmostEqual(msg.velocity.x, 0.18)
        self.assertAlmostEqual(msg.velocity.y, -0.12)
        self.assertAlmostEqual(msg.velocity.z, 0.0)
        self.assertAlmostEqual(msg.yaw, -0.40)
        self.assertAlmostEqual(msg.yaw_rate, 0.30)
        self.assertFalse(msg.type_mask & PositionTarget.IGNORE_PX)
        self.assertFalse(msg.type_mask & PositionTarget.IGNORE_PY)
        self.assertTrue(msg.type_mask & PositionTarget.IGNORE_VZ)
        self.assertEqual(msg.type_mask, 480)

    def test_position_hold_uses_only_px4_position_z_loop(self):
        """The supervisor must not add a second Z-velocity feedback loop."""
        sup = self.supervisor
        sup.odom = Odometry()
        sup.odom.pose.pose.position.z = 1.30
        sup.hold_pose = (1.0, 2.0, 1.30, 0.25)
        sup.cruise_z = 1.0
        sup.setpoint_pub = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(2.0)):
            sup._setpoint()

        msg = sup.setpoint_pub.publish.call_args.args[0]
        self.assertAlmostEqual(msg.position.z, 1.0)
        self.assertAlmostEqual(msg.velocity.z, 0.0)
        self.assertTrue(msg.type_mask & PositionTarget.IGNORE_VZ)
        self.assertEqual(msg.type_mask, 2552)

    def test_native_trajectory_reference_caps_horizontal_vector_norm(self):
        sup = self.supervisor
        sup.tracking_mode = "px4_native"
        sup.odom = Odometry()
        sup.odom.pose.pose.orientation.w = 1.0
        sup.hold_pose = (0.0, 0.0, 1.0, 0.0)
        sup.cruise_z = 1.0
        sup.max_speed = 0.40
        sup.max_yaw_rate = 0.65
        sup.trajectory_reference = PositionTarget()
        sup.trajectory_reference.velocity.x = 0.60
        sup.trajectory_reference.velocity.y = 0.80
        sup.setpoint_pub = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(2.0)):
            sup._setpoint(auto=True)

        msg = sup.setpoint_pub.publish.call_args.args[0]
        self.assertAlmostEqual(math.hypot(msg.velocity.x, msg.velocity.y), 0.40)

    def test_legacy_tracking_also_uses_only_fixed_position_z(self):
        sup = self.supervisor
        sup.tracking_mode = "legacy_mpc"
        sup.odom = Odometry()
        sup.odom.pose.pose.orientation.w = 1.0
        sup.odom.pose.pose.position.z = 0.75
        sup.hold_pose = (0.0, 0.0, 0.75, 0.0)
        sup.cruise_z = 1.0
        sup.max_speed = 0.40
        sup.max_yaw_rate = 0.65
        sup.cmd = Twist()
        sup.cmd.linear.x = 0.20
        sup.setpoint_pub = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(2.0)):
            sup._setpoint(auto=True)

        msg = sup.setpoint_pub.publish.call_args.args[0]
        self.assertAlmostEqual(msg.position.z, 1.0)
        self.assertAlmostEqual(msg.velocity.z, 0.0)
        self.assertFalse(msg.type_mask & PositionTarget.IGNORE_PZ)
        self.assertTrue(msg.type_mask & PositionTarget.IGNORE_VZ)

    def test_native_reference_callback_starts_current_command_epoch(self):
        sup = self.supervisor
        sup.tracking_mode = "px4_native"
        sup.planner_execution_since = self._time(6.0)
        sup.trajectory_reference = PositionTarget()
        sup.last_cmd = self._time(0.0)
        ref = self._native_reference(7.0)
        ref.position.x = 2.0

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(7.0)):
            sup._trajectory_reference_cb(ref)

        self.assertIs(sup.trajectory_reference, ref)
        self.assertEqual(sup.last_trajectory_reference, self._time(7.0))
        self.assertEqual(sup.last_cmd, self._time(7.0))

    def test_legacy_command_callback_rejects_nonfinite_input(self):
        sup = self.supervisor
        sup.tracking_mode = "legacy_mpc"
        sup.cmd = Twist()
        sup.last_cmd = self._time(1.0)
        bad = Twist()
        bad.angular.z = float("nan")

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(2.0)):
            sup._cmd_cb(bad)

        self.assertIsNot(sup.cmd, bad)
        self.assertEqual(sup.last_cmd, self._time(1.0))

    def test_native_reference_callback_rejects_invalid_source_epoch_or_values(self):
        sup = self.supervisor
        sup.tracking_mode = "px4_native"
        sup.planner_execution_since = self._time(10.0)
        accepted = PositionTarget()
        accepted.header.stamp = self._time(9.0)
        sup.trajectory_reference = accepted
        sup.last_trajectory_reference = self._time(9.0)
        sup.last_cmd = self._time(9.0)

        invalid_references = []
        zero_stamp = PositionTarget()
        invalid_references.append(zero_stamp)
        stale_stamp = PositionTarget()
        stale_stamp.header.stamp = self._time(9.9)
        invalid_references.append(stale_stamp)
        future_stamp = PositionTarget()
        future_stamp.header.stamp = self._time(10.1)
        invalid_references.append(future_stamp)
        nonfinite = PositionTarget()
        nonfinite.header.stamp = self._time(10.0)
        nonfinite.velocity.y = float("nan")
        invalid_references.append(nonfinite)

        for reference in invalid_references:
            with self.subTest(reference=reference):
                with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(10.0)):
                    sup._trajectory_reference_cb(reference)
                self.assertIs(sup.trajectory_reference, accepted)
                self.assertEqual(sup.last_trajectory_reference, self._time(9.0))
                self.assertEqual(sup.last_cmd, self._time(9.0))

        valid = self._native_reference(10.0)
        valid.position.x = 1.0
        valid.position.y = -2.0
        valid.velocity.x = 0.2
        valid.velocity.y = -0.1
        valid.yaw = 0.3
        valid.yaw_rate = -0.2
        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(10.0)):
            sup._trajectory_reference_cb(valid)
        self.assertIs(sup.trajectory_reference, valid)
        self.assertEqual(sup.last_trajectory_reference, self._time(10.0))
        self.assertEqual(sup.last_cmd, self._time(10.0))

    def test_native_reference_callback_rejects_position_jump_at_fcu_boundary(self):
        sup = self.supervisor
        sup.tracking_mode = "px4_native"
        sup.odom = Odometry()
        sup.odom.pose.pose.position.x = 1.0
        sup.odom.pose.pose.position.y = 2.0
        sup.planner_execution_since = self._time(10.0)
        sup.trajectory_reference_epoch = self._time(10.0)
        jumped = self._native_reference(10.0)
        jumped.position.x = 1.25
        jumped.position.y = 2.0

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(10.0)):
            sup._trajectory_reference_cb(jumped)

        self.assertTrue(sup.last_trajectory_reference.is_zero())
        self.assertTrue(sup.last_cmd.is_zero())

    def test_native_reference_rejects_wrong_frame_and_source_time_regression(self):
        sup = self.supervisor
        sup.tracking_mode = "px4_native"
        sup.planner_execution_since = self._time(10.0)
        first = self._native_reference(10.1)
        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(10.1)):
            sup._trajectory_reference_cb(first)
        self.assertIs(sup.trajectory_reference, first)

        wrong_frame = self._native_reference(10.2)
        wrong_frame.header.frame_id = "odom"
        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(10.2)):
            sup._trajectory_reference_cb(wrong_frame)
        self.assertIs(sup.trajectory_reference, first)

        regressed = self._native_reference(10.05)
        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(10.2)):
            sup._trajectory_reference_cb(regressed)
        self.assertIs(sup.trajectory_reference, first)
        self.assertEqual(sup.last_trajectory_reference, self._time(10.1))

    def test_invalid_native_reference_keeps_auto_in_first_reference_hold(self):
        sup = self.supervisor
        sup.state = MissionState.AUTO
        sup.state_since = self._time(10.0)
        sup.tracking_mode = "px4_native"
        sup.planner_state = Supervisor.PLANNER_EXECUTION_STATES[0]
        sup.planner_execution_since = self._time(10.0)
        sup.planner_state_since = self._time(10.0)
        sup.cmd_timeout = 0.3
        sup.planner_start_timeout = 3.0
        sup.last_cmd = rospy.Time(0)
        sup._height_consistency_failure = mock.Mock(return_value=None)
        sup._health_failure = mock.Mock(return_value=None)
        sup._flight_control_ready = mock.Mock(return_value=True)
        sup._altitude_guard = mock.Mock(return_value=None)
        sup._setpoint = mock.Mock()
        sup._transition = mock.Mock()
        invalid = PositionTarget()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(10.1)):
            sup._trajectory_reference_cb(invalid)
            sup._tick(None)

        sup._setpoint.assert_called_once_with()
        sup._transition.assert_not_called()

    def test_native_first_reference_uses_source_epoch_across_state_callback_race(self):
        sup = self.supervisor
        sup.state = MissionState.AUTO
        sup.state_since = self._time(10.0)
        sup.tracking_mode = "px4_native"
        sup.planner_state = Supervisor.PLANNER_EXECUTION_STATES[0]
        sup.planner_state_since = self._time(10.2)
        sup.planner_execution_since = self._time(10.2)
        sup.trajectory_reference_epoch = self._time(10.0)
        sup.last_trajectory_reference = self._time(10.1)
        sup.last_cmd = self._time(10.1)
        sup.cmd_timeout = 0.3
        sup._height_consistency_failure = mock.Mock(return_value=None)
        sup._health_failure = mock.Mock(return_value=None)
        sup._flight_control_ready = mock.Mock(return_value=True)
        sup._altitude_guard = mock.Mock(return_value=None)
        sup._setpoint = mock.Mock()
        sup._transition = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(10.25)):
            sup._tick(None)

        sup._setpoint.assert_called_once_with(auto=True)
        sup._transition.assert_not_called()

    def test_native_auto_times_out_by_source_stamp_not_receipt_time(self):
        sup = self.supervisor
        sup.state = MissionState.AUTO
        sup.state_since = self._time(9.0)
        sup.last_tick_time = self._time(9.98)
        sup.tracking_mode = "px4_native"
        sup.planner_state = Supervisor.PLANNER_EXECUTION_STATES[0]
        sup.planner_state_since = self._time(9.0)
        sup.planner_execution_since = self._time(9.0)
        sup.trajectory_reference_epoch = self._time(9.0)
        sup.cmd_timeout = 0.3
        sup._height_consistency_failure = mock.Mock(return_value=None)
        sup._health_failure = mock.Mock(return_value=None)
        sup._flight_control_ready = mock.Mock(return_value=True)
        sup._altitude_guard = mock.Mock(return_value=None)
        sup._setpoint = mock.Mock()
        sup._transition = mock.Mock()

        reference = self._native_reference(9.70)
        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(9.99)):
            sup._trajectory_reference_cb(reference)
        self.assertEqual(sup.last_cmd, self._time(9.99))

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(10.01)):
            sup._tick(None)

        self.assertEqual(sup._transition.call_args.args[0], MissionState.FAULT)
        self.assertIn("trajectory reference timeout", sup._transition.call_args.args[1])
        sup._setpoint.assert_not_called()

    def test_clock_rollback_resets_fault_landing_delay(self):
        sup = self.supervisor
        sup.state = MissionState.FAULT
        sup.state_since = self._time(10.0)
        sup.last_tick_time = self._time(10.0)
        sup.detail = "clock reset fault"
        sup.fault_land_delay = 3.0
        sup.stop_pub = mock.Mock()
        sup._setpoint = mock.Mock()
        sup._transition = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(2.0)):
            sup._tick(None)
        self.assertEqual(sup.state_since, self._time(2.0))
        sup._transition.assert_not_called()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(5.1)):
            sup._tick(None)
        sup._transition.assert_called_once_with(
            MissionState.LAND, "clock reset fault; landing")

    def test_clock_rollback_clears_auto_plan_reference_and_accepts_new_epoch(self):
        sup = self.supervisor
        sup.state = MissionState.AUTO
        sup.state_since = self._time(10.0)
        sup.last_tick_time = self._time(10.0)
        sup.tracking_mode = "px4_native"
        sup.planner_state = Supervisor.PLANNER_PLAN_TRAJ
        sup.planner_state_since = self._time(10.0)
        sup.planner_phase_since = self._time(10.0)
        sup.planner_execution_since = self._time(9.0)
        sup.trajectory_reference_epoch = self._time(10.0)
        sup.trajectory_reference = PositionTarget()
        sup.trajectory_reference.header.stamp = self._time(9.9)
        sup.last_trajectory_reference = self._time(9.9)
        sup.last_cmd = self._time(10.0)
        sup.stop_pub = mock.Mock()
        sup.fcu = State(connected=True, armed=True, mode="OFFBOARD")
        sup.planner_started = False
        sup._set_mapping_enabled = mock.Mock()
        sup._set_navigation_enabled = mock.Mock()
        sup._latch_hold = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(2.0)):
            sup._tick(None)

        self.assertEqual(sup.state, MissionState.FAULT)
        self.assertTrue(sup.last_cmd.is_zero())
        self.assertTrue(sup.last_trajectory_reference.is_zero())
        self.assertTrue(sup.trajectory_reference.header.stamp.is_zero())
        self.assertEqual(sup.trajectory_reference_epoch, self._time(2.0))
        self.assertEqual(sup.planner_state_since, self._time(2.0))
        self.assertEqual(sup.planner_phase_since, self._time(2.0))

        new_reference = self._native_reference(2.1)
        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(2.1)):
            sup._trajectory_reference_cb(new_reference)
        self.assertIs(sup.trajectory_reference, new_reference)

    def test_clock_rollback_preserves_land_and_retries_mode_service(self):
        sup = self.supervisor
        sup.state = MissionState.LAND
        sup.state_since = self._time(10.0)
        sup.last_tick_time = self._time(10.0)
        sup.stop_pub = mock.Mock()
        sup._setpoint = mock.Mock()
        sup.fcu = State(connected=True, armed=True, mode="OFFBOARD")
        sup.extended = ExtendedState()
        sup.last_service_attempt = self._time(10.0)
        sup.set_mode = mock.Mock(return_value=object())

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(2.0)):
            sup._tick(None)

        self.assertEqual(sup.state, MissionState.LAND)
        sup.set_mode.assert_called_once_with(custom_mode="AUTO.LAND")
        sup._setpoint.assert_not_called()

    def test_landing_never_disarms_from_stale_on_ground_state(self):
        sup = self.supervisor
        sup.state = MissionState.LAND
        sup.state_since = self._time(1.0)
        sup.last_tick_time = self._time(9.9)
        sup.fcu = State(connected=True, armed=True, mode="AUTO.LAND")
        sup.extended = ExtendedState(
            landed_state=ExtendedState.LANDED_STATE_ON_GROUND)
        sup.last_extended_state = self._time(7.0)
        sup.last_service_attempt = rospy.Time(0)
        sup.arm = mock.Mock()
        sup._transition = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(10.0)):
            sup._tick(None)

        sup.arm.assert_not_called()
        sup._transition.assert_not_called()

    def test_landing_streams_a_gentle_descending_position_target(self):
        sup = self.supervisor
        sup.state = MissionState.LAND
        sup.state_since = self._time(1.0)
        sup.last_tick_time = self._time(1.9)
        sup.fcu = State(connected=True, armed=True, mode="OFFBOARD")
        sup.extended = ExtendedState()
        sup.odom = Odometry()
        sup.odom.pose.pose.position.z = 1.0
        sup.last_odom = self._time(2.0)
        sup.ground_z = 0.0
        sup.gazebo_truth_guard_enabled = False
        sup.landing_target_z = 1.0
        sup.landing_last_update = self._time(1.9)
        sup._setpoint = mock.Mock()
        sup.set_mode = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(2.0)):
            sup._tick(None)

        self.assertAlmostEqual(sup.landing_target_z, 0.975, places=6)
        sup._setpoint.assert_called_once_with(landing=True)
        sup.set_mode.assert_not_called()

    def test_landing_falls_back_to_auto_land_if_vision_height_stops(self):
        sup = self.supervisor
        sup.state = MissionState.LAND
        sup.state_since = self._time(1.0)
        sup.last_tick_time = self._time(1.9)
        sup.fcu = State(connected=True, armed=True, mode="OFFBOARD")
        sup.extended = ExtendedState()
        sup.odom = Odometry()
        sup.last_odom = self._time(2.0)
        sup.px4_expected_height_reference = 3
        sup.last_vision_height = self._time(1.0)
        sup.gazebo_truth_guard_enabled = False
        sup.last_service_attempt = rospy.Time(0)
        sup.set_mode = mock.Mock(return_value=object())
        sup._setpoint = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(2.0)):
            sup._tick(None)

        sup.set_mode.assert_called_once_with(custom_mode="AUTO.LAND")
        sup._setpoint.assert_not_called()

    def test_tick_without_clock_rollback_keeps_current_epoch(self):
        sup = self.supervisor
        sup.state = MissionState.FAULT
        sup.state_since = self._time(9.0)
        sup.last_tick_time = self._time(10.0)
        sup.fault_land_delay = 3.0
        sup.detail = "operator hold"
        sup.stop_pub = mock.Mock()
        sup._setpoint = mock.Mock()
        sup._transition = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(10.1)):
            sup._tick(None)

        self.assertEqual(sup.state_since, self._time(9.0))
        self.assertEqual(sup.last_tick_time, self._time(10.1))
        sup.stop_pub.publish.assert_not_called()

    def test_nonfinite_native_reference_publishes_safe_fixed_z_hold(self):
        sup = self.supervisor
        sup.tracking_mode = "px4_native"
        sup.odom = Odometry()
        sup.odom.pose.pose.orientation.w = 1.0
        sup.hold_pose = (1.0, 2.0, 0.75, 0.25)
        sup.cruise_z = 1.0
        sup.max_speed = 0.40
        sup.max_yaw_rate = 0.65
        sup.trajectory_reference = PositionTarget()
        sup.trajectory_reference.position.x = float("inf")
        sup.setpoint_pub = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(2.0)):
            sup._setpoint(auto=True)

        msg = sup.setpoint_pub.publish.call_args.args[0]
        self.assertEqual((msg.position.x, msg.position.y), (1.0, 2.0))
        self.assertEqual(msg.position.z, 1.0)
        self.assertTrue(msg.type_mask & PositionTarget.IGNORE_VZ)
        self.assertFalse(msg.type_mask & PositionTarget.IGNORE_PZ)
        self.assertTrue(all(math.isfinite(value) for value in (
            msg.position.x, msg.position.y, msg.position.z,
            msg.velocity.x, msg.velocity.y, msg.velocity.z,
            msg.yaw, msg.yaw_rate)))

    def test_plan_traj_clears_previous_native_reference_epoch(self):
        sup = self.supervisor
        sup.state = MissionState.AUTO
        sup.tracking_mode = "px4_native"
        sup.planner_state = Supervisor.PLANNER_EXECUTION_STATES[0]
        sup.trajectory_reference = PositionTarget()
        sup.trajectory_reference.header.stamp = self._time(8.0)
        sup.last_trajectory_reference = self._time(8.0)
        sup._latch_hold = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(10.0)):
            sup._planner_state_cb(Int32(data=Supervisor.PLANNER_PLAN_TRAJ))

        self.assertTrue(sup.last_trajectory_reference.is_zero())
        self.assertTrue(sup.trajectory_reference.header.stamp.is_zero())
        sup._latch_hold.assert_called_once_with()

    def test_altitude_soft_excursion_holds_xy_and_recovers_z(self):
        sup = self.supervisor
        sup.state = MissionState.AUTO
        sup.state_since = self._time(1.0)
        sup.odom = Odometry()
        sup.odom.pose.pose.orientation.w = 1.0
        sup.odom.pose.pose.position.z = 1.35
        sup.hold_pose = (0.0, 0.0, 1.0, 0.0)
        sup.fcu = State(connected=True, armed=True, mode="OFFBOARD")
        sup._health_failure = mock.Mock(return_value=None)
        sup._flight_control_ready = mock.Mock(return_value=True)
        sup._setpoint = mock.Mock()
        sup._transition = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(2.0)):
            sup._tick(None)

        self.assertTrue(sup.altitude_recovery_active)
        self.assertIn("altitude recovery", sup.detail)
        sup._setpoint.assert_called_once_with()
        sup._transition.assert_not_called()

    def test_persistent_hard_altitude_excursion_faults(self):
        sup = self.supervisor
        sup.state = MissionState.AUTO
        sup.state_since = self._time(1.0)
        sup.odom = Odometry()
        sup.odom.pose.pose.orientation.w = 1.0
        sup.odom.pose.pose.position.z = 1.70
        sup.hold_pose = (0.0, 0.0, 1.0, 0.0)
        sup.altitude_hard_since = self._time(2.0)
        sup.fcu = State(connected=True, armed=True, mode="OFFBOARD")
        sup._health_failure = mock.Mock(return_value=None)
        sup._flight_control_ready = mock.Mock(return_value=True)
        sup._setpoint = mock.Mock()
        sup._transition = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(4.1)):
            sup._tick(None)

        self.assertEqual(sup._transition.call_args.args[0], MissionState.FAULT)
        self.assertIn("persistent altitude deviation", sup._transition.call_args.args[1])

    def test_takeoff_enters_hold_ready_without_initial_yaw_scan(self):
        """A stable takeoff must enable normal mapping/search without commanding scan yaw."""
        sup = self.supervisor
        sup.state = MissionState.OFFBOARD_TAKEOFF
        sup.detail = "taking off"
        sup.state_since = self._time(1.0)
        sup.odom = Odometry()
        sup.odom.pose.pose.orientation.w = 1.0
        sup.odom.pose.pose.position.z = 1.0
        sup.vertical_speed = 0.0
        sup.alt_stable_sec = 2.5
        sup.altitude_stable_since = self._time(2.0)
        sup.fcu = State(connected=True, armed=True, mode="OFFBOARD")
        sup._flight_inputs_failure = mock.Mock(return_value=None)
        sup._flight_control_ready = mock.Mock(return_value=True)
        sup._latch_hold = mock.Mock()
        sup._setpoint = mock.Mock()
        sup._transition = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(4.6)):
            sup._tick(None)

        sup._latch_hold.assert_called_once()
        sup._transition.assert_called_once_with(
            MissionState.HOLD_READY, "cruise altitude stable; initial scan disabled")
        sup._setpoint.assert_called_once_with(takeoff=True)

    def test_persistent_gazebo_height_disagreement_is_reported(self):
        sup = self.supervisor
        sup.gazebo_truth_guard_enabled = True
        sup.gazebo_truth_max_error = 0.15
        sup.gazebo_truth_hold_sec = 1.0
        sup.gazebo_truth_min_airborne_altitude = 0.20
        sup.gazebo_ground_z = 0.0
        sup.gazebo_z = 0.45
        sup.last_gazebo_truth = self._time(10.0)
        sup.gazebo_truth_error_since = rospy.Time(0)
        sup.odom = Odometry()
        sup.odom.pose.pose.position.z = 1.0
        sup.ground_z = 0.0

        pending = sup._height_consistency_failure(self._time(10.0))
        self.assertTrue(pending.startswith("pending height estimator/truth disagreement"))
        sup.last_gazebo_truth = self._time(11.1)
        failure = sup._height_consistency_failure(self._time(11.1))

        self.assertIn("height estimator/truth disagreement", failure)
        self.assertIn("0.550m", failure)

    def test_takeoff_does_not_enable_navigation_during_height_debounce(self):
        sup = self.supervisor
        sup.state = MissionState.OFFBOARD_TAKEOFF
        sup.detail = "taking off"
        sup.odom = Odometry()
        sup.odom.pose.pose.position.z = 1.0
        sup.vertical_speed = 0.0
        sup.altitude_stable_since = self._time(2.0)
        sup.alt_stable_sec = 2.5
        sup.fcu = State(connected=True, armed=True, mode="OFFBOARD")
        sup._flight_inputs_failure = mock.Mock(return_value=None)
        sup._flight_control_ready = mock.Mock(return_value=True)
        sup._height_consistency_failure = mock.Mock(
            return_value="pending height estimator/truth disagreement 0.200m")
        sup._setpoint = mock.Mock()
        sup._transition = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(5.0)):
            sup._tick(None)

        self.assertIn("verifying", sup.detail)
        sup._transition.assert_not_called()

    def test_stale_gazebo_truth_resets_disagreement_timer(self):
        sup = self.supervisor
        sup.gazebo_truth_guard_enabled = True
        sup.gazebo_truth_max_error = 0.15
        sup.gazebo_truth_hold_sec = 1.0
        sup.gazebo_truth_timeout = 0.50
        sup.gazebo_truth_min_airborne_altitude = 0.20
        sup.gazebo_ground_z = 0.0
        sup.gazebo_z = 0.45
        sup.last_gazebo_truth = self._time(10.0)
        sup.gazebo_truth_error_since = rospy.Time(0)
        sup.odom = Odometry()
        sup.odom.pose.pose.position.z = 1.0
        sup.ground_z = 0.0

        sup._height_consistency_failure(self._time(10.1))
        failure = sup._height_consistency_failure(self._time(11.0))

        self.assertEqual(failure, "Gazebo vehicle height truth stale")
        self.assertTrue(sup.gazebo_truth_error_since.is_zero())

    def test_zero_timestamp_gazebo_truth_is_never_treated_as_fresh(self):
        sup = self.supervisor
        sup.gazebo_truth_guard_enabled = True
        sup.gazebo_truth_max_error = 0.15
        sup.gazebo_truth_hold_sec = 1.0
        sup.gazebo_truth_timeout = 0.50
        sup.gazebo_truth_min_airborne_altitude = 0.20
        sup.gazebo_ground_z = 0.0
        sup.gazebo_z = 0.45
        sup.last_gazebo_truth = rospy.Time(0)
        sup.gazebo_truth_error_since = rospy.Time(0)
        sup.odom = Odometry()
        sup.odom.pose.pose.position.z = 1.0
        sup.ground_z = 0.0

        failure = sup._height_consistency_failure(self._time(10.0))

        self.assertEqual(failure, "Gazebo vehicle height truth stale")
        self.assertTrue(sup.gazebo_truth_error_since.is_zero())

    def test_auto_height_disagreement_enters_bounded_landing_fault(self):
        sup = self.supervisor
        sup.state = MissionState.AUTO
        sup._height_consistency_failure = mock.Mock(
            return_value="height estimator/truth disagreement 0.550m for 1.10s")
        sup._transition = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(11.0)):
            sup._tick(None)

        sup._transition.assert_called_once_with(
            MissionState.FAULT,
            "height estimator/truth disagreement 0.550m for 1.10s")

    def test_gazebo_model_callback_uses_named_vehicle_and_captures_ground(self):
        sup = self.supervisor
        sup.gazebo_truth_guard_enabled = True
        sup.gazebo_model_name = "iris_depth_camera"
        sup.gazebo_ground_z = None
        sup.gazebo_z = None
        sup.fcu = State(armed=False)
        msg = ModelStates()
        msg.name = ["ground_plane", "iris_depth_camera"]
        msg.pose = [Pose(), Pose()]
        msg.pose[1].position.z = 0.265

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(1.0)):
            sup._gazebo_model_states_cb(msg)

        self.assertAlmostEqual(sup.gazebo_z, 0.265)
        self.assertAlmostEqual(sup.gazebo_ground_z, 0.265)

        msg.pose[1].position.z = float("nan")
        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(2.0)):
            sup._gazebo_model_states_cb(msg)
        self.assertAlmostEqual(sup.gazebo_z, 0.265)
        self.assertAlmostEqual(sup.gazebo_ground_z, 0.265)

    def test_odom_callback_derives_vertical_speed_from_position(self):
        """Catches trusting PX4 twist.z, whose sign/frame differed in the recorded run."""
        sup = self.supervisor
        first = Odometry()
        first.header.stamp = self._time(5.0)
        first.header.frame_id = "map"
        first.child_frame_id = "base_link"
        first.pose.pose.orientation.w = 1.0
        first.pose.pose.position.z = 0.40
        second = Odometry()
        second.header.stamp = self._time(5.5)
        second.header.frame_id = "map"
        second.child_frame_id = "base_link"
        second.pose.pose.orientation.w = 1.0
        second.pose.pose.position.z = 0.50
        second.twist.twist.linear.z = -3.0

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(5.0)):
            sup._odom_cb(first)
        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(5.5)):
            sup._odom_cb(second)

        self.assertGreater(sup.vertical_speed, 0.0)
        self.assertLess(sup.vertical_speed, 0.21)

    def test_flight_input_callbacks_use_valid_monotonic_source_stamps_and_frames(self):
        sup = self.supervisor
        odom = Odometry()
        odom.header.stamp = self._time(10.0)
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"
        odom.pose.pose.orientation.w = 1.0
        rgb = Image()
        rgb.header.stamp = self._time(10.0)
        rgb.header.frame_id = "iris_camera_optical_frame"
        rgb.encoding = "rgb8"
        rgb.width = rgb.height = 1
        rgb.step = 3
        rgb.data = b"\x00\x00\x00"
        depth = Image()
        depth.header.stamp = self._time(10.0)
        depth.header.frame_id = "iris_camera_optical_frame"
        depth.encoding = "32FC1"
        depth.width = depth.height = 1
        depth.step = 4
        depth.data = b"\x00\x00\x00\x00"

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(10.1)):
            sup._odom_cb(odom)
            sup._rgb_cb(rgb)
            sup._depth_cb(depth)

        self.assertIs(sup.odom, odom)
        self.assertEqual(sup.last_odom, self._time(10.0))
        self.assertEqual(sup.last_rgb, self._time(10.0))
        self.assertEqual(sup.last_depth, self._time(10.0))

        truncated_rgb = Image()
        truncated_rgb.header.stamp = self._time(10.1)
        truncated_rgb.header.frame_id = "iris_camera_optical_frame"
        truncated_rgb.encoding = "rgb8"
        truncated_rgb.width = truncated_rgb.height = 2
        truncated_rgb.step = 6
        truncated_rgb.data = b"\x00" * 6
        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(10.1)):
            sup._rgb_cb(truncated_rgb)
        self.assertEqual(sup.last_rgb, self._time(10.0))

        bad = Odometry()
        bad.header.stamp = self._time(10.1)
        bad.header.frame_id = "odom"
        bad.child_frame_id = "base_link"
        bad.pose.pose.orientation.w = 1.0
        bad.pose.pose.position.z = float("nan")
        regressed_rgb = Image()
        regressed_rgb.header.stamp = self._time(9.9)
        regressed_rgb.header.frame_id = "iris_camera_optical_frame"
        future_depth = Image()
        future_depth.header.stamp = self._time(10.3)
        future_depth.header.frame_id = "iris_camera_optical_frame"
        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(10.1)):
            sup._odom_cb(bad)
            sup._rgb_cb(regressed_rgb)
            sup._depth_cb(future_depth)

        self.assertIs(sup.odom, odom)
        self.assertEqual(sup.last_odom, self._time(10.0))
        self.assertEqual(sup.last_rgb, self._time(10.0))
        self.assertEqual(sup.last_depth, self._time(10.0))

    def test_mapping_semantic_and_vlm_health_keep_source_time(self):
        sup = self.supervisor
        sup.mapping_enabled = True
        cloud = Header()
        cloud.stamp = self._time(20.0)
        cloud.frame_id = "map"
        semantic = SemanticObservation()
        semantic.header.stamp = self._time(20.0)
        semantic.header.frame_id = "map"
        semantic.target_label = "chair"
        semantic.yolo_valid = True
        semantic.clip_valid = True
        diagnostics = DiagnosticArray()
        diagnostics.header.stamp = self._time(20.0)
        status = DiagnosticStatus()
        status.level = DiagnosticStatus.WARN
        status.values = [KeyValue("yoloe_ready", "True"),
                         KeyValue("clipitm_ready", "True")]
        diagnostics.status = [status]

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(20.1)):
            sup._mapping_update_cb(cloud)
            sup._semantic_cb(semantic)
            sup._vlm_cb(diagnostics)

        self.assertEqual(sup.last_mapping_update, self._time(20.0))
        self.assertEqual(sup.last_semantic, self._time(20.0))
        self.assertEqual(sup.semantic_target, "chair")
        self.assertEqual(sup.last_vlm_health, self._time(20.0))
        self.assertIsNone(sup._perception_servers_failure(self._time(20.1)))

        stale = Header()
        stale.stamp = self._time(19.0)
        stale.frame_id = "map"
        wrong_frame = SemanticObservation()
        wrong_frame.header.stamp = self._time(20.1)
        wrong_frame.header.frame_id = "odom"
        wrong_frame.target_label = "chair"
        wrong_frame.yolo_valid = True
        wrong_frame.clip_valid = True
        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(20.2)):
            sup._mapping_update_cb(stale)
            sup._semantic_cb(wrong_frame)
        self.assertEqual(sup.last_mapping_update, self._time(20.0))
        self.assertEqual(sup.last_semantic, self._time(20.0))

    def test_health_failure_reports_depth_map_age_and_threshold(self):
        sup = self.supervisor
        sup.mapping_enabled = True
        sup.mapping_update_timeout = 3.0
        sup.last_mapping_update = self._time(6.5)
        sup._flight_inputs_failure = mock.Mock(return_value=None)

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(10.0)):
            failure = sup._health_failure()

        self.assertEqual(
            failure, "depth map output stale (age 3.500s > 3.000s)")

    def test_auto_fault_preserves_specific_health_failure(self):
        sup = self.supervisor
        sup.state = MissionState.AUTO
        sup._health_failure = mock.Mock(
            return_value="depth map output stale (age 3.500s > 3.000s)")
        sup._transition = mock.Mock()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(10.0)):
            sup._tick(None)

        sup._transition.assert_called_once_with(
            MissionState.FAULT,
            "depth map output stale (age 3.500s > 3.000s)")

    def test_fault_stops_trajectory_and_cancels_active_planner(self):
        sup = self.supervisor
        sup.state = MissionState.AUTO
        sup.detail = "planner active"
        sup.state_since = self._time(1.0)
        sup.stop_pub = mock.Mock()
        sup.cancel_planner_pub = mock.Mock()
        sup.planner_started = True
        sup.navigation_enabled = True
        sup.mapping_enabled = True
        sup.cmd = object()
        sup.last_cmd = self._time(1.0)
        sup.odom = None

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(2.0)):
            sup._transition(MissionState.FAULT, "planner command timeout")

        sup.stop_pub.publish.assert_called_once()
        sup.cancel_planner_pub.publish.assert_called_once()
        self.assertTrue(sup.last_cmd.is_zero())
        self.assertFalse(sup.navigation_enabled)
        self.assertFalse(sup.mapping_enabled)

    def test_operator_stop_from_auto_enters_land_and_cancels_planner(self):
        sup = self.supervisor
        sup.state = MissionState.AUTO
        sup.detail = "planner active"
        sup.state_since = self._time(1.0)
        sup.fcu = State(connected=True, armed=True, mode="OFFBOARD")
        sup.stop_pub = mock.Mock()
        sup.cancel_planner_pub = mock.Mock()
        sup.planner_started = True
        sup.navigation_enabled = True
        sup.mapping_enabled = True
        sup.cmd = Twist()
        sup.last_cmd = self._time(1.0)
        sup.odom = None

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(2.0)):
            response = sup._stop(None)

        self.assertTrue(response.success)
        self.assertEqual(sup.state, MissionState.LAND)
        sup.stop_pub.publish.assert_called_once()
        sup.cancel_planner_pub.publish.assert_called_once()

    def test_complete_takeoff_search_result_and_landing_lifecycle(self):
        sup = self.supervisor
        sup.state = MissionState.WAIT_FCU
        sup.detail = "waiting"
        sup.state_since = self._time(0.0)
        sup.requested = True
        sup.target = "chair"
        sup.fcu = State(connected=True, armed=False, mode="AUTO.LOITER")
        sup.extended = ExtendedState()
        sup.odom = Odometry()
        sup.odom.pose.pose.orientation.w = 1.0
        sup.odom.pose.pose.position.z = -0.2
        sup.cmd = Twist()
        sup.last_cmd = self._time(0.0)
        sup.hold_pose = None
        sup.ground_z = None
        sup.cruise_z = None
        sup.altitude = 1.0
        sup.alt_tol = 0.1
        sup.alt_stable_sec = 2.0
        sup.altitude_stable_since = None
        sup.mapping_update_timeout = 0.75
        sup.last_mapping_update = self._time(0.0)
        sup.prestream_sec = 2.0
        sup.arm_transition_timeout = 10.0
        sup.completion_hold = 2.0
        sup.fault_land_delay = 3.0
        sup.planner_start_timeout = 3.0
        sup.planner_replan_timeout = 3.0
        sup.cmd_timeout = 0.3
        sup.planner_ready_timeout = 10.0
        sup.planner_ack_timeout = 2.0
        sup.navigation_ready_timeout = 30.0
        sup.planner_started = False
        sup.planner_state = Supervisor.PLANNER_WAIT_TRIGGER
        sup.planner_state_since = self._time(0.0)
        sup.planner_trigger_time = self._time(0.0)
        sup.last_result = None
        sup.last_service_attempt = self._time(0.0)
        sup.stop_pub = mock.Mock()
        sup.cancel_planner_pub = mock.Mock()
        sup.trigger_pub = mock.Mock()
        sup._setpoint = mock.Mock()
        sup._healthy = mock.Mock(return_value=True)
        sup._health_failure = mock.Mock(return_value=None)
        sup._flight_inputs_failure = mock.Mock(return_value=None)
        sup._perception_servers_failure = mock.Mock(return_value=None)
        sup._planner_ready = mock.Mock(return_value=True)

        def set_mode(*args, **kwargs):
            sup.fcu.mode = kwargs["custom_mode"]
            return type("Response", (), {"mode_sent": True})()

        def arm(value):
            sup.fcu.armed = value
            return type("Response", (), {"success": True})()

        sup.set_mode = set_mode
        sup.arm = arm

        def tick(seconds):
            with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(seconds)):
                sup._tick(None)

        tick(1.0)
        self.assertEqual(sup.state, MissionState.PRESTREAM)
        self.assertAlmostEqual(sup.ground_z, -0.2)
        self.assertAlmostEqual(sup.cruise_z, 0.8)
        tick(3.1)
        self.assertEqual(sup.state, MissionState.ARM)
        tick(4.2)
        self.assertEqual(sup.fcu.mode, "OFFBOARD")
        tick(5.3)
        self.assertTrue(sup.fcu.armed)
        tick(5.4)
        self.assertEqual(sup.state, MissionState.OFFBOARD_TAKEOFF)

        sup.odom.pose.pose.position.z = 0.8
        tick(5.5)
        tick(7.6)
        self.assertEqual(sup.state, MissionState.HOLD_READY)
        self.assertTrue(sup.navigation_enabled)
        self.assertTrue(sup.mapping_enabled)
        tick(7.7)
        self.assertTrue(sup.mapping_enabled)
        self.assertTrue(sup.navigation_enabled)
        sup.trigger_pub.publish.assert_called_once()
        sup.planner_state = Supervisor.PLANNER_ACTIVE_STATES[0]
        tick(7.8)
        self.assertEqual(sup.state, MissionState.AUTO)

        sup.last_cmd = self._time(7.8)
        tick(7.9)
        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(8.0)):
            sup._result_cb(Int32(data=4))
        self.assertEqual(sup.state, MissionState.HOLD)
        self.assertFalse(sup.navigation_enabled)
        self.assertFalse(sup.mapping_enabled)
        tick(10.1)
        self.assertEqual(sup.state, MissionState.LAND)

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(18.0)):
            sup._extended_cb(ExtendedState(
                landed_state=ExtendedState.LANDED_STATE_ON_GROUND))
        tick(18.1)
        self.assertFalse(sup.fcu.armed)
        self.assertEqual(sup.fcu.mode, "OFFBOARD")
        tick(19.2)
        self.assertEqual(sup.state, MissionState.DISARMED)
        self.assertFalse(sup.requested)


if __name__ == "__main__":
    unittest.main()
