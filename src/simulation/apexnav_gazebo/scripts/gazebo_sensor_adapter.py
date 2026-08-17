#!/usr/bin/env python3
"""Publish a timestamp-coherent HM3D-v2 virtual RGB-D camera for Gazebo."""

from collections import deque
import math
import threading

import cv2
import message_filters
import numpy as np
import rospy
import tf.transformations as tft
from cv_bridge import CvBridge, CvBridgeError
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image


class GazeboSensorAdapter:
    def __init__(self):
        rospy.init_node("apexnav_gazebo_sensor_adapter")
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.odom = deque()
        self.source_info = None
        self.source_info_key = None
        self.remap_key = None
        self.maps = None
        self.last_stamp = rospy.Time(0)
        self.drop_counts = {}

        self.world_frame = rospy.get_param("~world_frame", "map")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.optical_frame = rospy.get_param("~camera/optical_frame", "iris_camera_optical_frame")
        self.width = int(rospy.get_param("~camera/width", 640))
        self.height = int(rospy.get_param("~camera/height", 480))
        self.fx = float(rospy.get_param("~camera/fx", 388.1910413097385))
        self.fy = float(rospy.get_param("~camera/fy", 422.0475153598262))
        self.cx = float(rospy.get_param("~camera/cx", 320.0))
        self.cy = float(rospy.get_param("~camera/cy", 240.0))
        self.slop = float(rospy.get_param("~sync/rgb_depth_slop", 0.005))
        self.history_sec = float(rospy.get_param("~sync/odom_history_sec", 5.0))
        self.max_bracket = float(rospy.get_param("~sync/max_odom_bracket_sec", 0.10))
        self.max_future = float(rospy.get_param("~sync/max_future_extrapolation_sec", 0.02))
        self.max_frame_age = float(rospy.get_param("~sync/max_frame_age_sec", 0.50))
        translation = rospy.get_param("~extrinsic/translation", [0.10, 0.0, 0.035])
        rpy = rospy.get_param("~extrinsic/rpy", [-math.pi / 2.0, 0.0, -math.pi / 2.0])
        self.t_base_camera = tft.euler_matrix(*[float(x) for x in rpy])
        self.t_base_camera[:3, 3] = np.asarray(translation, dtype=np.float64)

        self.rgb_pub = rospy.Publisher("/apexnav/camera/rgb/image_raw", Image, queue_size=2)
        self.depth_pub = rospy.Publisher("/apexnav/camera/depth/image_raw", Image, queue_size=2)
        self.info_pub = rospy.Publisher("/apexnav/camera/camera_info", CameraInfo, queue_size=2)
        self.pose_pub = rospy.Publisher("/apexnav/camera/pose", Odometry, queue_size=10)
        self.diag_pub = rospy.Publisher("/apexnav/sensors/diagnostics", DiagnosticArray, queue_size=2)

        info_topic = rospy.get_param("~source_camera_info", "/iris_depth_camera/camera/rgb/camera_info")
        odom_topic = rospy.get_param("~odom_topic", "/mavros/local_position/odom")
        rospy.Subscriber(info_topic, CameraInfo, self._info_cb, queue_size=1)
        rospy.Subscriber(odom_topic, Odometry, self._odom_cb, queue_size=100)
        rgb = message_filters.Subscriber(
            rospy.get_param("~source_rgb", "/iris_depth_camera/camera/rgb/image_raw"), Image)
        depth = message_filters.Subscriber(
            rospy.get_param("~source_depth", "/iris_depth_camera/camera/depth/image_raw"), Image)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb, depth], int(rospy.get_param("~sync/queue_size", 20)), self.slop,
            allow_headerless=False)
        self.sync.registerCallback(self._rgb_depth_cb)
        rospy.Timer(rospy.Duration(1.0), self._diagnostics)

    def _drop(self, reason):
        self.drop_counts[reason] = self.drop_counts.get(reason, 0) + 1
        rospy.logwarn_throttle(2.0, "Gazebo sensor frame dropped: %s", reason)

    def _info_cb(self, msg):
        if msg.width <= 0 or msg.height <= 0 or msg.K[0] <= 0.0 or msg.K[4] <= 0.0:
            self._drop("invalid_camera_info")
            return
        with self.lock:
            key = (msg.width, msg.height, tuple(msg.K), tuple(msg.D))
            if self.source_info_key is not None and key != self.source_info_key:
                self._drop("camera_calibration_changed_restart_required")
                return
            self.source_info_key = key
            self.source_info = msg

    def _odom_cb(self, msg):
        stamp = msg.header.stamp
        if stamp.is_zero():
            self._drop("zero_odom_stamp")
            return
        with self.lock:
            if self.odom and stamp <= self.odom[-1].header.stamp:
                self._drop("odom_time_regression")
                return
            self.odom.append(msg)
            cutoff = stamp - rospy.Duration(self.history_sec)
            while len(self.odom) > 2 and self.odom[1].header.stamp < cutoff:
                self.odom.popleft()

    @staticmethod
    def _pose_matrix(msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        matrix = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
        matrix[:3, 3] = [p.x, p.y, p.z]
        return matrix

    def _interpolated_pose(self, stamp):
        with self.lock:
            samples = list(self.odom)
        if not samples:
            raise ValueError("no_odom")
        before = None
        after = None
        for sample in samples:
            if sample.header.stamp <= stamp:
                before = sample
            if sample.header.stamp >= stamp:
                after = sample
                break
        if before is None:
            raise ValueError("sensor_precedes_odom_history")
        if after is None:
            age = (stamp - before.header.stamp).to_sec()
            if age < 0.0 or age > self.max_future:
                raise ValueError("future_extrapolation")
            after = before
        gap = (after.header.stamp - before.header.stamp).to_sec()
        if gap > self.max_bracket:
            raise ValueError("odom_bracket_too_wide")
        if gap <= 1e-9:
            base = self._pose_matrix(before)
        else:
            alpha = (stamp - before.header.stamp).to_sec() / gap
            bp, ap = before.pose.pose.position, after.pose.pose.position
            bq, aq = before.pose.pose.orientation, after.pose.pose.orientation
            q = tft.quaternion_slerp(
                [bq.x, bq.y, bq.z, bq.w], [aq.x, aq.y, aq.z, aq.w], alpha)
            base = tft.quaternion_matrix(q)
            base[:3, 3] = [
                bp.x + alpha * (ap.x - bp.x),
                bp.y + alpha * (ap.y - bp.y),
                bp.z + alpha * (ap.z - bp.z),
            ]
        return np.matmul(base, self.t_base_camera)

    def _target_info(self, stamp):
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = self.optical_frame
        msg.width, msg.height = self.width, self.height
        msg.distortion_model = "plumb_bob"
        msg.D = [0.0] * 5
        msg.K = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
        msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.P = [self.fx, 0.0, self.cx, 0.0, 0.0, self.fy, self.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return msg

    def _ensure_maps(self, info):
        key = (info.width, info.height, tuple(info.K), tuple(info.D),
               self.width, self.height, self.fx, self.fy, self.cx, self.cy)
        if key == self.remap_key:
            return
        src_k = np.asarray(info.K, dtype=np.float64).reshape(3, 3)
        source_hfov = 2.0 * math.atan(float(info.width) / (2.0 * src_k[0, 0]))
        target_hfov = 2.0 * math.atan(float(self.width) / (2.0 * self.fx))
        if source_hfov + math.radians(0.5) < target_hfov:
            raise ValueError("source_fov_smaller_than_hm3dv2")
        target_k = np.asarray([self.fx, 0.0, self.cx, 0.0, self.fy, self.cy,
                               0.0, 0.0, 1.0], dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(info.D or [0.0] * 5, dtype=np.float64)
        self.maps = cv2.initUndistortRectifyMap(
            src_k, distortion, np.eye(3), target_k, (self.width, self.height), cv2.CV_32FC1)
        self.remap_key = key

    def _rgb_depth_cb(self, rgb_msg, depth_msg):
        stamp = depth_msg.header.stamp
        if stamp.is_zero():
            self._drop("zero_sensor_stamp")
            return
        if abs((rgb_msg.header.stamp - stamp).to_sec()) > self.slop:
            self._drop("rgb_depth_delta")
            return
        if not self.last_stamp.is_zero() and stamp <= self.last_stamp:
            self._drop("sensor_time_regression")
            return
        if rospy.Time.now().is_zero():
            self._drop("sim_clock_not_ready")
            return
        age = (rospy.Time.now() - stamp).to_sec()
        if age < -self.max_future or age > self.max_frame_age:
            self._drop("sensor_frame_age")
            return
        with self.lock:
            info = self.source_info
        if info is None:
            self._drop("missing_camera_info")
            return
        if (rgb_msg.width != info.width or rgb_msg.height != info.height or
                depth_msg.width != info.width or depth_msg.height != info.height):
            self._drop("source_dimensions_differ_from_camera_info")
            return
        try:
            pose = self._interpolated_pose(stamp)
            self._ensure_maps(info)
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            rgb_out = cv2.remap(rgb, self.maps[0], self.maps[1], cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
            depth_out = cv2.remap(depth, self.maps[0], self.maps[1], cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            out_rgb = self.bridge.cv2_to_imgmsg(rgb_out, "rgb8")
            out_depth = self.bridge.cv2_to_imgmsg(depth_out, depth_msg.encoding)
        except (ValueError, CvBridgeError, cv2.error) as exc:
            self._drop(str(exc))
            return
        for msg in (out_rgb, out_depth):
            msg.header.stamp = stamp
            msg.header.frame_id = self.optical_frame
        pose_msg = Odometry()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self.world_frame
        pose_msg.child_frame_id = self.optical_frame
        pose_msg.pose.pose.position.x = float(pose[0, 3])
        pose_msg.pose.pose.position.y = float(pose[1, 3])
        pose_msg.pose.pose.position.z = float(pose[2, 3])
        q = tft.quaternion_from_matrix(pose)
        pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y = q[0], q[1]
        pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w = q[2], q[3]
        self.rgb_pub.publish(out_rgb)
        self.depth_pub.publish(out_depth)
        self.info_pub.publish(self._target_info(stamp))
        self.pose_pub.publish(pose_msg)
        self.last_stamp = stamp

    def _diagnostics(self, _event):
        status = DiagnosticStatus()
        status.name = "apexnav/gazebo_sensor_adapter"
        status.hardware_id = "px4_iris_depth_camera"
        healthy = self.source_info is not None and bool(self.odom) and not self.last_stamp.is_zero()
        status.level = DiagnosticStatus.OK if healthy else DiagnosticStatus.WARN
        status.message = "HM3D-v2 virtual camera ready" if healthy else "waiting for coherent inputs"
        status.values = [KeyValue("profile", "hm3dv2"), KeyValue("resolution", "640x480")]
        status.values.extend(KeyValue("dropped_" + k, str(v)) for k, v in sorted(self.drop_counts.items()))
        array = DiagnosticArray()
        array.header.stamp = rospy.Time.now()
        array.status = [status]
        self.diag_pub.publish(array)


if __name__ == "__main__":
    GazeboSensorAdapter()
    rospy.spin()
