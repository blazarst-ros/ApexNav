# MultipleMasksWithConfidence.msg


# Observation context shared by all detections in this frame
std_msgs/Header header
string scene_id
string episode_id
uint32 step_index
geometry_msgs/Point camera_position
float64 camera_yaw

# Contains multiple projected object clouds
sensor_msgs/PointCloud2[] point_clouds

# Corresponding confidence scores
float32[] confidence_scores

# List of label indices
int32[] label_indices
