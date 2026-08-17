#!/usr/bin/env python3
"""Timestamp-safe Lite YOLOE + CLIPITM bridge for real and Gazebo RGB-D."""

from copy import deepcopy
from pathlib import Path
import sys
import threading
import time

import hydra
from omegaconf import DictConfig
import message_filters
import numpy as np
import requests
import rospy
import tf.transformations as tft
from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import Float64, Header, String
from visualization_msgs.msg import Marker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from basic_utils.path_utils import resolve_existing_path
from llm.answer_reader.answer_reader import read_answer
from plan_env.msg import MultipleMasksWithConfidence, SemanticObservation
from vlm.utils.get_itm_message import get_itm_message_cosine
from vlm.utils.get_object_utils import get_object


class RealWorldNode:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        rospy.init_node("apexnav_real_world_perception", anonymous=False)
        self.bridge = CvBridge()
        self.data_lock = threading.Lock()
        self.condition = threading.Condition()
        self.pending = None
        self.camera_info = None
        self.calibration_changed = False
        self.label = str(cfg.target_label).strip() or None
        self.label_generation = 0
        self.llm_answer, self.room = [], "everywhere"
        self.world_frame = str(cfg.ros.world_frame)
        self.last_result_stamp = rospy.Time(0)
        self.last_yolo_ms = float("nan")
        self.last_clip_ms = float("nan")
        self.last_error = "waiting for synchronized input"
        self.dropped_busy = 0
        self.dropped_stale = 0
        self.accepted = 0
        self.sync_received = 0
        self.input_count = {"rgb": 0, "depth": 0, "pose": 0}
        self.last_input = {"rgb": rospy.Time(0), "depth": rospy.Time(0), "pose": rospy.Time(0)}
        self.server_health = {"YOLOE": False, "CLIPITM": False}

        self.semantic_pub = rospy.Publisher(
            "/apexnav/vlm/semantic_observation", SemanticObservation, queue_size=3)
        self.image_pub = rospy.Publisher("/apexnav/vlm/annotated_image", Image, queue_size=2)
        self.diag_pub = rospy.Publisher("/apexnav/vlm/diagnostics", DiagnosticArray, queue_size=2)
        self.marker_pub = rospy.Publisher("/apexnav/vlm/status_marker", Marker, queue_size=1)
        self.threshold_pub = rospy.Publisher(
            "/detector/confidence_threshold", Float64, queue_size=1, latch=True)
        self.legacy = bool(cfg.ros.legacy_topics)
        if self.legacy:
            self.legacy_cloud_pub = rospy.Publisher(
                "/detector/clouds_with_scores", MultipleMasksWithConfidence, queue_size=3)
            self.legacy_itm_pub = rospy.Publisher("/clip/cosine_score", Float64, queue_size=3)

        rospy.Subscriber(str(cfg.ros.camera_info_topic), CameraInfo, self._camera_info_cb, queue_size=2)
        rospy.Subscriber("/detector/label", String, self._label_cb, queue_size=1)
        rgb_sub = message_filters.Subscriber(str(cfg.ros.rgb_topic), Image)
        depth_sub = message_filters.Subscriber(str(cfg.ros.depth_topic), Image)
        pose_sub = message_filters.Subscriber(str(cfg.ros.camera_pose_topic), Odometry)
        # Independent observers make synchronization failures visible in diagnostics.
        rospy.Subscriber(str(cfg.ros.rgb_topic), Image, self._input_cb, "rgb", queue_size=2)
        rospy.Subscriber(str(cfg.ros.depth_topic), Image, self._input_cb, "depth", queue_size=2)
        rospy.Subscriber(str(cfg.ros.camera_pose_topic), Odometry, self._input_cb, "pose", queue_size=2)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, pose_sub], queue_size=int(cfg.ros.sync_queue),
            slop=float(cfg.ros.sync_slop), allow_headerless=False)
        self.sync.registerCallback(self._sync_cb)
        self.worker = threading.Thread(target=self._worker, name="vlm-latest-frame", daemon=True)
        self.worker.start()
        rospy.Timer(rospy.Duration(1.0), self._status_timer)
        rospy.loginfo("Lite perception uses atomic YOLOE+CLIPITM observations in %s", self.world_frame)

    def _camera_info_cb(self, msg):
        if msg.width <= 0 or msg.height <= 0 or msg.K[0] <= 0 or msg.K[4] <= 0:
            self.last_error = "invalid CameraInfo"
            return
        with self.data_lock:
            if self.camera_info is not None:
                old = self.camera_info
                changed = (old.width != msg.width or old.height != msg.height or
                           any(abs(a - b) > 1e-6 for a, b in zip(old.K, msg.K)))
                if changed:
                    self.calibration_changed = True
                    self.last_error = "CameraInfo changed; restart mission"
                    return
            self.camera_info = deepcopy(msg)

    def _label_cb(self, msg):
        label = msg.data.strip()
        if not label:
            return
        with self.data_lock:
            if label == self.label:
                return
            self.label = label
            self.label_generation += 1
            generation = self.label_generation
            self.llm_answer, self.room = [], "everywhere"
        with self.condition:
            self.pending = None
        if bool(self.cfg.use_llm):
            try:
                answer_path = resolve_existing_path(self.cfg.llm.llm_answer_path)
                response_path = resolve_existing_path(self.cfg.llm.llm_response_path)
                answer, room, _ = read_answer(
                    answer_path, response_path, label, self.cfg.llm.llm_client.llm_client)
                with self.data_lock:
                    if generation == self.label_generation:
                        self.llm_answer, self.room = answer, room
            except Exception as exc:
                rospy.logwarn("LLM expansion unavailable; target only: %s", exc)
        rospy.loginfo("ApexNav target generation %d: %s", generation, label)

    def _input_cb(self, msg, stream):
        self.input_count[stream] += 1
        self.last_input[stream] = rospy.Time.now()

    def _sync_cb(self, rgb_msg, depth_msg, pose_msg):
        self.sync_received += 1
        stamp = depth_msg.header.stamp
        if self.calibration_changed:
            return
        if stamp.is_zero():
            self.last_error = "zero source timestamp"
            return
        deltas = [abs((rgb_msg.header.stamp - stamp).to_sec()),
                  abs((pose_msg.header.stamp - stamp).to_sec())]
        if max(deltas) > float(self.cfg.ros.sync_slop):
            self.last_error = "source timestamp delta exceeded"
            return
        age = (rospy.Time.now() - stamp).to_sec()
        if age < -0.02 or age > float(self.cfg.ros.max_frame_age):
            self.dropped_stale += 1
            self.last_error = "stale or future source frame"
            return
        with self.data_lock:
            info = deepcopy(self.camera_info)
            label = self.label
            generation = self.label_generation
            answers = list(self.llm_answer)
            room = self.room
        if info is None or label is None:
            self.last_error = "waiting for CameraInfo and target label"
            return
        if (info.width != rgb_msg.width or info.height != rgb_msg.height or
                depth_msg.width != rgb_msg.width or depth_msg.height != rgb_msg.height):
            self.last_error = "RGB/depth/CameraInfo dimensions differ"
            return
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8").copy()
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough").copy()
        except Exception as exc:
            self.last_error = "image conversion: %s" % exc
            return
        item = (stamp, rgb_msg.header.frame_id, rgb, depth, pose_msg, info,
                label, generation, answers, room)
        with self.condition:
            if self.pending is not None:
                self.dropped_busy += 1
            self.pending = item
            self.condition.notify()

    def _worker(self):
        while not rospy.is_shutdown():
            with self.condition:
                has_work = self.condition.wait_for(
                    lambda: self.pending is not None or rospy.is_shutdown(), 0.5)
                if rospy.is_shutdown():
                    return
                # A timeout is normal while waiting for the first target/frame.
                # The label callback can also intentionally clear `pending`.
                if not has_work or self.pending is None:
                    continue
                item, self.pending = self.pending, None
            self._infer(item)

    def _infer(self, item):
        stamp, camera_frame, rgb, depth, pose, info, label, generation, answers, room = item
        image = rgb.copy()
        scores, masks, label_ids = [], [], []
        yolo_valid = clip_valid = False
        clip_score = float("nan")
        yolo_ms = clip_ms = float("nan")
        try:
            image, scores, masks, label_ids, stats = get_object(
                label, rgb, self.cfg.detector, answers, return_stats=True)
            yolo_ms = float(stats.get("yoloe_latency_ms", float("nan")))
            yolo_valid = True
        except Exception as exc:
            self.last_error = "YOLOE failed: %s" % exc
        try:
            start = time.perf_counter()
            clip_score, stats = get_itm_message_cosine(
                rgb, label, room or "everywhere", return_stats=True)
            clip_ms = float(stats.get("model_ms", (time.perf_counter() - start) * 1000.0))
            clip_valid = True
        except Exception as exc:
            self.last_error = "CLIPITM failed: %s" % exc
        with self.data_lock:
            current = generation == self.label_generation and label == self.label
        if not current or (rospy.Time.now() - stamp).to_sec() > float(self.cfg.vlm.stale_result_sec):
            self.dropped_stale += 1
            self.last_error = "discarded stale VLM generation"
            return
        observation = SemanticObservation()
        observation.header = Header(stamp=stamp, frame_id=self.world_frame)
        observation.confidence_scores = scores
        observation.label_indices = label_ids
        observation.point_clouds = [
            self._mask_cloud(mask, depth, pose, info, stamp) for mask in masks]
        observation.clip_score = clip_score
        observation.target_label = label
        observation.yolo_latency_ms = yolo_ms
        observation.clip_latency_ms = clip_ms
        observation.yolo_valid = yolo_valid
        observation.clip_valid = clip_valid
        self.semantic_pub.publish(observation)
        out = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")
        out.header = Header(stamp=stamp, frame_id=camera_frame)
        self.image_pub.publish(out)
        if self.legacy:
            legacy = MultipleMasksWithConfidence()
            legacy.point_clouds = observation.point_clouds
            legacy.confidence_scores = scores
            legacy.label_indices = label_ids
            self.legacy_cloud_pub.publish(legacy)
            if clip_valid:
                self.legacy_itm_pub.publish(Float64(data=clip_score))
        self.last_result_stamp = stamp
        self.last_yolo_ms, self.last_clip_ms = yolo_ms, clip_ms
        self.accepted += 1
        if yolo_valid and clip_valid:
            self.last_error = "ready"

    def _mask_cloud(self, mask, depth, pose, info, stamp):
        depth_m = np.asarray(depth, dtype=np.float32) * float(self.cfg.ros.depth_unit_scale)
        valid = (np.asarray(mask, dtype=bool) & np.isfinite(depth_m) &
                 (depth_m >= float(self.cfg.ros.min_depth)) &
                 (depth_m <= float(self.cfg.ros.max_depth)))
        stride = max(1, int(self.cfg.ros.object_cloud_stride))
        v, u = np.where(valid)
        v, u = v[::stride], u[::stride]
        header = Header(stamp=stamp, frame_id=self.world_frame)
        if not len(u):
            return point_cloud2.create_cloud_xyz32(header, [])
        z = depth_m[v, u]
        k = info.K
        points = np.column_stack(((u - k[2]) * z / k[0], (v - k[5]) * z / k[4], z))
        p = pose.pose.pose.position
        q = pose.pose.pose.orientation
        matrix = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
        matrix[:3, 3] = [p.x, p.y, p.z]
        points = (matrix @ np.column_stack((points, np.ones(len(points)))).T).T[:, :3]
        return point_cloud2.create_cloud_xyz32(header, points.tolist())

    def _health(self, name, url):
        try:
            response = requests.get(str(url), timeout=float(self.cfg.vlm.health_timeout))
            data = response.json() if response.ok else {}
            return response.ok and bool(data.get("ready")) and data.get("name") == name
        except (requests.RequestException, ValueError):
            return False

    def _status_timer(self, _event):
        self.threshold_pub.publish(Float64(data=float(self.cfg.detector.yoloe.confidence_threshold)))
        self.server_health["YOLOE"] = self._health("yoloe", self.cfg.vlm.yoloe_health_url)
        self.server_health["CLIPITM"] = self._health("clipitm", self.cfg.vlm.clipitm_health_url)
        status = DiagnosticStatus()
        status.name = "apexnav/vlm"
        status.hardware_id = "YOLOE+CLIPITM"
        age = -1.0 if self.last_result_stamp.is_zero() else (rospy.Time.now() - self.last_result_stamp).to_sec()
        servers_ok = all(self.server_health.values())
        result_fresh = self.accepted > 0 and 0.0 <= age <= float(self.cfg.vlm.stale_result_sec)
        ok = servers_ok and result_fresh
        status.level = DiagnosticStatus.OK if ok else (DiagnosticStatus.WARN if servers_ok else DiagnosticStatus.ERROR)
        status.message = "YOLOE + CLIPITM inference ready" if ok else self.last_error
        status.values = [
            KeyValue("models", "YOLOE+CLIPITM"),
            KeyValue("target", self.label or ""),
            KeyValue("yoloe_ready", str(self.server_health["YOLOE"])),
            KeyValue("clipitm_ready", str(self.server_health["CLIPITM"])),
            KeyValue("result_age_sec", "%.3f" % age),
            KeyValue("yolo_latency_ms", str(self.last_yolo_ms)),
            KeyValue("clip_latency_ms", str(self.last_clip_ms)),
            KeyValue("accepted", str(self.accepted)),
            KeyValue("dropped_busy", str(self.dropped_busy)),
            KeyValue("dropped_stale", str(self.dropped_stale)),
            KeyValue("sync_received", str(self.sync_received)),
            KeyValue("rgb_received", str(self.input_count["rgb"])),
            KeyValue("depth_received", str(self.input_count["depth"])),
            KeyValue("pose_received", str(self.input_count["pose"])),
        ]
        array = DiagnosticArray()
        array.header.stamp = rospy.Time.now()
        array.status = [status]
        self.diag_pub.publish(array)
        marker = Marker()
        marker.header.stamp = array.header.stamp
        marker.header.frame_id = self.world_frame
        marker.ns = "apexnav_vlm"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.z = 2.0
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.35
        marker.color.r = 0.1 if ok else 1.0
        marker.color.g = 1.0 if ok else 0.1
        marker.color.b = 0.1
        marker.color.a = 1.0
        marker.text = "YOLOE+CLIPITM | target=%s | age=%.2fs | %s" % (
            self.label or "unset", age, self.last_error)
        self.marker_pub.publish(marker)

    def run(self):
        rospy.spin()


@hydra.main(version_base=None, config_path="config", config_name="real_world_test")
def main(cfg: DictConfig):
    RealWorldNode(cfg).run()


if __name__ == "__main__":
    main()
