# MultipleMasksWithConfidence.msg


# Contains multiple masks; each mask is a boolean array
sensor_msgs/PointCloud2[] point_clouds

# Corresponding confidence scores
float32[] confidence_scores

# List of label indices
int32[] label_indices

# Complete label dictionary for this episode.  Index 0 is always the target;
# indices 1..N are confusion/similar-object labels.
string[] class_names
