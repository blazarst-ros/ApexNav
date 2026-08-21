#!/usr/bin/env python3

import csv
import os
import threading
import time
from datetime import datetime

import rospy
from gazebo_msgs.msg import ModelStates
from mavros_msgs.msg import ExtendedState, State, StatusText
from mavros_msgs.msg import PositionTarget
from nav_msgs.msg import Odometry
from plan_env.msg import MissionState
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, Int32, String


MISSION_NAMES = {
    MissionState.WAIT_FCU: "WAIT_FCU",
    MissionState.PRESTREAM: "PRESTREAM",
    MissionState.ARM: "ARM",
    MissionState.OFFBOARD_TAKEOFF: "OFFBOARD_TAKEOFF",
    MissionState.HOLD_READY: "HOLD_READY",
    MissionState.AUTO: "AUTO",
    MissionState.HOLD: "HOLD",
    MissionState.LAND: "LAND",
    MissionState.DISARMED: "DISARMED",
    MissionState.FAULT: "FAULT",
}
PLANNER_NAMES = {
    0: "INIT", 1: "WAIT_TRIGGER", 2: "PLAN_TRAJ",
    3: "EXEC_TRAJ", 4: "REPLAN", 5: "FINISH",
}
RESULT_NAMES = {
    0: "EXPLORE", 1: "SEARCH_OBJECT", 2: "STUCKING",
    3: "NO_FRONTIER", 4: "REACH_OBJECT", 5: "MANUAL_STOP",
}
LANDED_NAMES = {
    ExtendedState.LANDED_STATE_UNDEFINED: "UNDEFINED",
    ExtendedState.LANDED_STATE_ON_GROUND: "ON_GROUND",
    ExtendedState.LANDED_STATE_IN_AIR: "IN_AIR",
    ExtendedState.LANDED_STATE_TAKEOFF: "TAKEOFF",
    ExtendedState.LANDED_STATE_LANDING: "LANDING",
}


class ApexNavStateLogger:
    FIELDS = (
        "wall_time", "ros_time", "source_stamp", "event",
        "fcu_connected", "armed", "guided", "px4_mode", "system_status",
        "landed_state", "landed_state_name", "fcu_status_severity", "fcu_status_text",
        "battery_voltage", "battery_current", "battery_percentage",
        "battery_power_supply_status", "battery_present",
        "mission_state", "mission_state_name", "target_label", "detail",
        "altitude", "planner_active", "mapping_enabled", "navigation_enabled",
        "z_feedback_source", "estimator_z", "estimator_ground_z", "estimator_agl",
        "estimator_vz", "gazebo_model_name", "gazebo_z", "gazebo_ground_z",
        "gazebo_agl", "gazebo_minus_estimator_agl", "setpoint_z", "setpoint_vz",
        "setpoint_type_mask", "z_position_command_active", "z_velocity_command_active",
        "planner_state", "planner_state_name",
        "exploration_result", "exploration_result_name",
    )

    def __init__(self):
        rospy.init_node("apexnav_state_logger")
        default_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..", "RuntimeData", "logs"))
        log_dir = os.path.abspath(os.path.expanduser(
            rospy.get_param("~log_directory", default_dir)))
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%z")
        self.log_path = os.path.join(
            log_dir, "apexnav_state_{}_{}.csv".format(timestamp, os.getpid()))
        self.lock = threading.Lock()
        self.closed = False
        self.last_signatures = {}
        self.last_write_monotonic = {}
        self.mission_sample_period = float(
            rospy.get_param("~mission_sample_period_sec", 1.0))
        self.state_heartbeat = float(
            rospy.get_param("~state_heartbeat_sec", 5.0))
        self.gazebo_model_name = rospy.get_param("~gazebo_model_name", "iris")
        self.selected_gazebo_model = ""
        self.estimator_ground_z = None
        self.gazebo_ground_z = None
        self.values = {field: "" for field in self.FIELDS}
        self.stream = open(self.log_path, "x", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.stream, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.stream.flush()

        self.path_pub = rospy.Publisher(
            "/apexnav/logging/state_log_file", String, queue_size=1, latch=True)
        self.path_pub.publish(String(data=self.log_path))
        rospy.Subscriber("/mavros/state", State, self._fcu_cb, queue_size=10)
        rospy.Subscriber("/mavros/extended_state", ExtendedState,
                         self._extended_cb, queue_size=10)
        rospy.Subscriber("/mavros/statustext/recv", StatusText,
                         self._status_text_cb, queue_size=20)
        rospy.Subscriber("/mavros/battery", BatteryState,
                         self._battery_cb, queue_size=10)
        rospy.Subscriber("/mavros/local_position/odom", Odometry,
                         self._odom_cb, queue_size=20)
        rospy.Subscriber("/mavros/setpoint_raw/local", PositionTarget,
                         self._setpoint_cb, queue_size=20)
        rospy.Subscriber("/gazebo/model_states", ModelStates,
                         self._gazebo_models_cb, queue_size=10)
        rospy.Subscriber("/apexnav/mission/state", MissionState,
                         self._mission_cb, queue_size=10)
        rospy.Subscriber("/apexnav/mission/navigation_enabled", Bool,
                         self._navigation_cb, queue_size=10)
        rospy.Subscriber("/apexnav/mission/mapping_enabled", Bool,
                         self._mapping_cb, queue_size=10)
        rospy.Subscriber("/ros/state", Int32, self._planner_cb, queue_size=10)
        rospy.Subscriber("/ros/expl_result", Int32, self._result_cb, queue_size=10)
        rospy.on_shutdown(self.close)
        self._write("logger_started", force=True)
        rospy.loginfo("ApexNav state CSV log: %s", self.log_path)

    @staticmethod
    def _name(mapping, value):
        return mapping.get(value, "UNKNOWN_{}".format(value))

    def _write(self, event, source_stamp="", signature=None, repeat_sec=None, force=False):
        with self.lock:
            if self.closed:
                return
            now_monotonic = time.monotonic()
            if not force and signature is not None:
                unchanged = self.last_signatures.get(event) == signature
                elapsed = now_monotonic - self.last_write_monotonic.get(event, 0.0)
                if unchanged and (repeat_sec is None or elapsed < repeat_sec):
                    return
                self.last_signatures[event] = signature
                self.last_write_monotonic[event] = now_monotonic
            row = dict(self.values)
            row["wall_time"] = datetime.now().astimezone().isoformat(timespec="milliseconds")
            row["ros_time"] = "{:.9f}".format(rospy.Time.now().to_sec())
            row["source_stamp"] = source_stamp
            row["event"] = event
            self.writer.writerow(row)
            self.stream.flush()

    def _fcu_cb(self, msg):
        self.values.update({
            "fcu_connected": msg.connected,
            "armed": msg.armed,
            "guided": msg.guided,
            "px4_mode": msg.mode,
            "system_status": msg.system_status,
        })
        self._write("mavros_state", signature=(
            msg.connected, msg.armed, msg.guided, msg.mode, msg.system_status),
            repeat_sec=self.state_heartbeat)

    def _extended_cb(self, msg):
        self.values.update({
            "landed_state": msg.landed_state,
            "landed_state_name": self._name(LANDED_NAMES, msg.landed_state),
        })
        self._write("mavros_extended_state", signature=(msg.landed_state,))

    def _status_text_cb(self, msg):
        self.values.update({
            "fcu_status_severity": msg.severity,
            "fcu_status_text": msg.text,
        })
        self._write("mavros_status_text", "{:.9f}".format(msg.header.stamp.to_sec()),
                    signature=(msg.severity, msg.text, msg.header.stamp.to_nsec()))

    def _battery_cb(self, msg):
        self.values.update({
            "battery_voltage": self._format_float(msg.voltage),
            "battery_current": self._format_float(msg.current),
            "battery_percentage": self._format_float(msg.percentage),
            "battery_power_supply_status": msg.power_supply_status,
            "battery_present": msg.present,
        })
        self._write(
            "mavros_battery", "{:.9f}".format(msg.header.stamp.to_sec()),
            signature=(round(msg.voltage, 3), round(msg.current, 3),
                       round(msg.percentage, 3), msg.power_supply_status,
                       msg.present),
            repeat_sec=self.mission_sample_period)

    @staticmethod
    def _format_float(value):
        return "{:.6f}".format(value)

    def _update_height_difference(self):
        estimator_agl = self.values.get("estimator_agl", "")
        gazebo_agl = self.values.get("gazebo_agl", "")
        if estimator_agl != "" and gazebo_agl != "":
            self.values["gazebo_minus_estimator_agl"] = self._format_float(
                float(gazebo_agl) - float(estimator_agl))

    def _odom_cb(self, msg):
        z = msg.pose.pose.position.z
        if self.estimator_ground_z is None:
            self.estimator_ground_z = z
        self.values.update({
            "z_feedback_source": "/mavros/local_position/odom",
            "estimator_z": self._format_float(z),
            "estimator_ground_z": self._format_float(self.estimator_ground_z),
            "estimator_agl": self._format_float(z - self.estimator_ground_z),
            "estimator_vz": self._format_float(msg.twist.twist.linear.z),
        })
        self._update_height_difference()
        stamp = msg.header.stamp.to_sec()
        self._write("local_odom", "{:.9f}".format(stamp),
                    signature=(round(z, 4), round(msg.twist.twist.linear.z, 4)),
                    repeat_sec=self.mission_sample_period)

    def _select_gazebo_model(self, names):
        if self.selected_gazebo_model in names:
            return self.selected_gazebo_model
        if self.gazebo_model_name in names:
            return self.gazebo_model_name
        matches = [name for name in names if self.gazebo_model_name in name]
        self.selected_gazebo_model = matches[0] if matches else ""
        return self.selected_gazebo_model

    def _gazebo_models_cb(self, msg):
        model = self._select_gazebo_model(msg.name)
        if not model:
            return
        index = msg.name.index(model)
        if index >= len(msg.pose):
            return
        z = msg.pose[index].position.z
        if self.gazebo_ground_z is None:
            self.gazebo_ground_z = z
        self.values.update({
            "gazebo_model_name": model,
            "gazebo_z": self._format_float(z),
            "gazebo_ground_z": self._format_float(self.gazebo_ground_z),
            "gazebo_agl": self._format_float(z - self.gazebo_ground_z),
        })
        self._update_height_difference()
        self._write("gazebo_model_state", signature=(model, round(z, 4)),
                    repeat_sec=self.mission_sample_period)

    def _setpoint_cb(self, msg):
        self.values.update({
            "setpoint_z": self._format_float(msg.position.z),
            "setpoint_vz": self._format_float(msg.velocity.z),
            "setpoint_type_mask": msg.type_mask,
            "z_position_command_active": not bool(msg.type_mask & PositionTarget.IGNORE_PZ),
            "z_velocity_command_active": not bool(msg.type_mask & PositionTarget.IGNORE_VZ),
        })
        self._write("local_setpoint", "{:.9f}".format(msg.header.stamp.to_sec()),
                    signature=(round(msg.position.z, 4), round(msg.velocity.z, 4),
                               msg.type_mask), repeat_sec=self.mission_sample_period)

    def _mission_cb(self, msg):
        self.values.update({
            "mission_state": msg.state,
            "mission_state_name": self._name(MISSION_NAMES, msg.state),
            "target_label": msg.target_label,
            "detail": msg.detail,
            "altitude": "{:.4f}".format(msg.altitude),
            "planner_active": msg.planner_active,
        })
        self._write("mission_state", "{:.9f}".format(msg.header.stamp.to_sec()),
                    signature=(msg.state, msg.target_label, msg.detail, msg.fcu_connected,
                               msg.armed, msg.px4_mode, msg.planner_active),
                    repeat_sec=self.mission_sample_period)

    def _navigation_cb(self, msg):
        self.values["navigation_enabled"] = msg.data
        self._write("navigation_enabled", signature=(msg.data,))

    def _mapping_cb(self, msg):
        self.values["mapping_enabled"] = msg.data
        self._write("mapping_enabled", signature=(msg.data,))

    def _planner_cb(self, msg):
        self.values.update({
            "planner_state": msg.data,
            "planner_state_name": self._name(PLANNER_NAMES, msg.data),
        })
        self._write("planner_state", signature=(msg.data,), repeat_sec=self.state_heartbeat)

    def _result_cb(self, msg):
        self.values.update({
            "exploration_result": msg.data,
            "exploration_result_name": self._name(RESULT_NAMES, msg.data),
        })
        self._write("exploration_result", signature=(msg.data,))

    def close(self):
        with self.lock:
            if self.closed:
                return
            row = dict(self.values)
            row["wall_time"] = datetime.now().astimezone().isoformat(timespec="milliseconds")
            row["ros_time"] = "{:.9f}".format(rospy.Time.now().to_sec())
            row["event"] = "logger_stopped"
            self.writer.writerow(row)
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()
            self.closed = True


if __name__ == "__main__":
    ApexNavStateLogger()
    rospy.spin()
