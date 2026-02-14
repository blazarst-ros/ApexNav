import rospy
from sensor_msgs.msg import Image, PointCloud2 # 增加 PointCloud2 支持
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Quaternion, Point
from tf.transformations import quaternion_from_euler
import numpy as np
from copy import deepcopy

class ROSPublisher:
    def __init__(self, robot_name):
        """
        :param robot_name: 机器人名称 (如 'robot_1'), 用于构建命名空间话题和 TF 框架
        """
        self.robot_name = robot_name
        self.bridge = CvBridge()
        
        # 1. 话题命名空间化 (Namespace Isolation)
        # 将原先的全局话题修改为 /{robot_name}/...
        ns = f"/{robot_name}"
        
        # 基础感知话题
        self.depth_pub = rospy.Publisher(f"{ns}/habitat/camera_depth", Image, queue_size=10)
        self.rgb_pub = rospy.Publisher(f"{ns}/habitat/camera_rgb", Image, queue_size=10)
        self.odom_pub = rospy.Publisher(f"{ns}/habitat/odom", Odometry, queue_size=10)
        self.pose_pub = rospy.Publisher(f"{ns}/habitat/sensor_pose", Odometry, queue_size=10)
        
        # 物体点云话题 (对应原先讨论的 obj_point_cloud_pub)
        # 格式通常为 PointCloud2，如果你的系统使用自定义格式请调整
        self.object_pub = rospy.Publisher(f"{ns}/detector/object_point_cloud", PointCloud2, queue_size=10)

    def publish_depth(self, ros_time, depth_image):
        depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="passthrough")
        depth_msg.header.stamp = ros_time
        depth_msg.header.frame_id = "world" 
        self.depth_pub.publish(depth_msg)

    def publish_rgb(self, ros_time, rgb_image):
        # 注意：此处假设传入的 rgb_image 已经在外部转为了 BGR
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding="bgr8")
        rgb_msg.header.stamp = ros_time
        rgb_msg.header.frame_id = "world"
        self.rgb_pub.publish(rgb_msg)

    def publish_odom(self, ros_time, gps, compass):
        """发布里程计，解决 TF 冲突的关键"""
        copy_compass = deepcopy(compass)
        odom = Odometry()
        odom.header.stamp = ros_time
        odom.header.frame_id = "world"
        
        # 【关键修改】child_frame_id 必须唯一，否则 RViz 中的多台机器会重叠闪烁
        odom.child_frame_id = f"{self.robot_name}/base_link"
        
        odom.pose.pose = Pose(
            position=Point(-gps[2], -gps[0], gps[1]), # Habitat 到 ROS 坐标系转换
            orientation=Quaternion(*quaternion_from_euler(0, 0, copy_compass)),
        )
        self.odom_pub.publish(odom)

    def publish_camera_odom(self, ros_time, gps, compass, pitch):
        """发布相机位姿"""
        copy_compass = deepcopy(compass)
        copy_pitch = deepcopy(pitch)
        sensor_pose = Odometry()
        sensor_pose.header.stamp = ros_time
        sensor_pose.header.frame_id = "world"
        
        # 同样使用带前缀的坐标系名称
        sensor_pose.child_frame_id = f"{self.robot_name}/camera_link"
        
        sensor_pose.pose.pose = Pose(
            position=Point(-gps[2], -gps[0], gps[1] + 0.88),
            orientation=Quaternion(
                *quaternion_from_euler(
                    copy_pitch + np.pi / 2.0, np.pi, copy_compass + np.pi / 2.0
                )
            ),
        )
        self.pose_pub.publish(sensor_pose)

    def habitat_publish_ros_topic(self, observations):
        """
        主发布接口：接收 RobotBridge 提取后的观测字典并分发
        """
        # 1. 基础数据提取
        depth_image = observations.get("depth")
        rgb_image = observations.get("rgb")
        gps = observations.get("gps")
        compass = observations.get("compass")
        camera_pitch = observations.get("camera_pitch", 0.0)
        
        ros_time = rospy.Time.now()

        # 2. 调用具体的发布函数
        if depth_image is not None:
            self.publish_depth(ros_time, depth_image)
        if rgb_image is not None:
            self.publish_rgb(ros_time, rgb_image)
        if gps is not None and compass is not None:
            self.publish_odom(ros_time, gps, compass)
            self.publish_camera_odom(ros_time, gps, compass, camera_pitch)
            
        # 3. 额外数据处理：物体点云 (obj_point_cloud)
        # 如果字典中包含点云数据，则发布
        if "obj_point_cloud" in observations:
            pc_data = observations["obj_point_cloud"]
            if isinstance(pc_data, PointCloud2):
                pc_data.header.stamp = ros_time
                pc_data.header.frame_id = "world"
                self.object_pub.publish(pc_data)