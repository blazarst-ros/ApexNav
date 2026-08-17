#!/usr/bin/env python3
"""Publish world->camera pose at each depth image's source timestamp."""

import rospy
import tf2_ros
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image


class TimestampedCameraPose:
    def __init__(self):
        rospy.init_node("apexnav_camera_pose")
        self.world_frame = rospy.get_param("~world_frame", "odom")
        self.camera_frame = rospy.get_param("~camera_frame", "camera_color_optical_frame")
        output = rospy.get_param("~output_topic", "/apexnav/camera/pose")
        depth = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.timeout = float(rospy.get_param("~lookup_timeout", 0.08))
        self.last_stamp = rospy.Time(0)
        self.publisher = rospy.Publisher(output, Odometry, queue_size=10)
        self.buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.listener = tf2_ros.TransformListener(self.buffer)
        rospy.Subscriber(depth, Image, self.depth_callback, queue_size=20)

    def depth_callback(self, image):
        stamp = image.header.stamp
        if stamp.is_zero() or (not self.last_stamp.is_zero() and stamp <= self.last_stamp):
            rospy.logwarn_throttle(2.0, "Rejecting zero/duplicate/regressing depth timestamp")
            return
        try:
            transform = self.buffer.lookup_transform(
                self.world_frame, self.camera_frame, stamp, rospy.Duration(self.timeout))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(
                2.0, "No exact %s -> %s TF at %.6f: %s",
                self.world_frame, self.camera_frame, stamp.to_sec(), exc)
            return
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self.world_frame
        msg.child_frame_id = self.camera_frame
        msg.pose.pose.position.x = transform.transform.translation.x
        msg.pose.pose.position.y = transform.transform.translation.y
        msg.pose.pose.position.z = transform.transform.translation.z
        msg.pose.pose.orientation = transform.transform.rotation
        self.publisher.publish(msg)
        self.last_stamp = stamp


if __name__ == "__main__":
    TimestampedCameraPose()
    rospy.spin()
