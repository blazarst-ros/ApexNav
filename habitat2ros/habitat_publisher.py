import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Quaternion, Point, PoseStamped
from tf.transformations import quaternion_from_euler
from habitat.core.simulator import Observations
import numpy as np
from copy import deepcopy


class ROSPublisher:
    def __init__(self, agent_name: str = "agent_0", camera_height: float = 0.88):
        """
        Create ROS publishers with namespaced topics for a specific agent.

        Args:
            agent_name: Name of the agent (e.g., "agent_0", "agent_1", "agent_2").
                        This becomes the namespace between /habitat/ and the topic name.
        """
        ns = agent_name
        # Create ROS publishers (namespaced by agent)
        self.depth_pub = rospy.Publisher(f"/habitat/{ns}/camera_depth", Image, queue_size=30)
        self.rgb_pub = rospy.Publisher(f"/habitat/{ns}/camera_rgb", Image, queue_size=30)
        self.odom_pub = rospy.Publisher(f"/habitat/{ns}/odom", Odometry, queue_size=30)
        self.pose_pub = rospy.Publisher(f"/habitat/{ns}/sensor_pose", Odometry, queue_size=30)
        # Create cv_bridge object
        self.bridge = CvBridge()
        self.agent_name = ns
        self.camera_height = float(camera_height)
        rospy.set_param(f"/habitat/{ns}/camera_height", self.camera_height)

    def publish_depth(self, ros_time, depth_image):
        depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="passthrough")
        depth_msg.header.stamp = ros_time
        depth_msg.header.frame_id = "world"
        self.depth_pub.publish(depth_msg)

    def publish_rgb(self, ros_time, rgb_image):
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
        rgb_msg.header.stamp = ros_time
        rgb_msg.header.frame_id = "world"
        self.rgb_pub.publish(rgb_msg)

    def publish_robot_odom(self, ros_time, gps, compass):
        copy_compass = deepcopy(compass)
        odom = Odometry()
        odom.header.stamp = ros_time
        odom.header.frame_id = "world"
        odom.child_frame_id = "base_link"
        odom.pose.pose = Pose(
            position=Point(-gps[2], -gps[0], gps[1]),
            orientation=Quaternion(*quaternion_from_euler(0, 0, copy_compass)),
        )
        self.odom_pub.publish(odom)

    def publish_camera_odom(self, ros_time, gps, compass, pitch):
        copy_compass = deepcopy(compass)
        copy_pitch = deepcopy(pitch)
        sensor_pose = Odometry()
        sensor_pose.header.stamp = ros_time
        sensor_pose.header.frame_id = "world"
        sensor_pose.child_frame_id = "base_link"
        sensor_pose.pose.pose = Pose(
            position=Point(-gps[2], -gps[0], gps[1] + self.camera_height),
            orientation=Quaternion(
                *quaternion_from_euler(
                    copy_pitch + np.pi / 2.0, np.pi, copy_compass + np.pi / 2.0
                )
            ),
        )
        self.pose_pub.publish(sensor_pose)

    def habitat_publish_ros_topic(self, observations):
        """
        Publish all habitat observations to namespaced ROS topics.

        Args:
            observations: Dictionary containing "depth", "rgb", "gps", "compass", "camera_pitch".
                          For multi-agent, call this method once per agent with that agent's
                          observations dict.
        """
        depth_image = observations["depth"]
        rgb_image = observations["rgb"]
        gps = observations["gps"]
        compass = observations["compass"]
        camera_pitch = observations["camera_pitch"]
        ros_time = rospy.Time.now()
        self.publish_depth(ros_time, depth_image)
        self.publish_camera_odom(ros_time, gps, compass, camera_pitch)
        self.publish_rgb(ros_time, rgb_image)
        self.publish_robot_odom(ros_time, gps, compass)
