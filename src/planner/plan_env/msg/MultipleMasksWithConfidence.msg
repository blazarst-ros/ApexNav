# MultipleMasksWithConfidence.msg


# Contains multiple masks; each mask is a boolean array
sensor_msgs/PointCloud2[] point_clouds

# Corresponding confidence scores
float32[] confidence_scores

# List of label indices
int32[] label_indices

# Corresponding mask area ratios in the source image
float32[] mask_scales
