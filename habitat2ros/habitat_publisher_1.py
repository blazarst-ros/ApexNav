import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Quaternion, Point
from tf.transformations import quaternion_from_euler
from copy import deepcopy
import numpy as np

class ROSPublisher:
    def __init__(self, robot_name):
        """
        初始化时传入 robot_name (如 'robot_1'), 用于隔离话题和坐标系
        """
        self.robot_name = robot_name
        self.bridge = CvBridge()

        # 1. 话题名称去全局化，带上机器人前缀
        # 例如: /robot_1/habitat/camera_depth
        prefix = f"/{robot_name}"
        self.depth_pub = rospy.Publisher(f"{prefix}/habitat/camera_depth", Image, queue_size=10)
        self.rgb_pub = rospy.Publisher(f"{prefix}/habitat/camera_rgb", Image, queue_size=10)
        self.odom_pub = rospy.Publisher(f"{prefix}/habitat/odom", Odometry, queue_size=10)
        self.pose_pub = rospy.Publisher(f"{prefix}/habitat/sensor_pose", Odometry, queue_size=10)

    def publish_depth(self, ros_time, depth_image):
        depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="passthrough")
        depth_msg.header.stamp = ros_time
        depth_msg.header.frame_id = "world" # 深度图通常相对于世界或相机系，这里保持world即可
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
        
        # 2. 关键修改：子坐标系必须带前缀，否则RViz里两个机器人会重叠
        # 例如: robot_1/base_link
        odom.child_frame_id = f"{self.robot_name}/base_link"
        
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
        
        # 同上，防止冲突
        sensor_pose.child_frame_id = f"{self.robot_name}/base_link"
        
        sensor_pose.pose.pose = Pose(
            position=Point(-gps[2], -gps[0], gps[1] + 0.88), # 加上相机高度
            orientation=Quaternion(
                *quaternion_from_euler(
                    copy_pitch + np.pi / 2.0, np.pi, copy_compass + np.pi / 2.0
                )
            ),
        )
        self.pose_pub.publish(sensor_pose)

    def habitat_publish_ros_topic(self, observations):
        """
        调用此函数时，传入的 observations 必须已经提取为该 Agent 独有的数据
        """
        # 注意：这里假设传入的 observations 已经是处理过（去除 agent_x_ 前缀）的字典
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