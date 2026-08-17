# One atomic YOLOE + CLIPITM result for a single source RGB frame.
# The point clouds are expressed in header.frame_id and all fields belong to
# header.stamp. Consumers must never combine this message with newer frames.
std_msgs/Header header
sensor_msgs/PointCloud2[] point_clouds
float32[] confidence_scores
int32[] label_indices
float32 clip_score
string target_label
float32 yolo_latency_ms
float32 clip_latency_ms
bool yolo_valid
bool clip_valid
