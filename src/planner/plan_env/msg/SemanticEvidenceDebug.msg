# Published once per accepted object detection after it is inserted/merged into ObjectMap2D.
# Contains the observation and fused cluster values needed to recompute:
# s_tilde = rho * confidence * (1 - exp(-beta * observation_num))

std_msgs/Header header

int32 agent_id
int32 cluster_id
int32 label
int32 best_label
string stage
string status
string reason
bool accepted

bool use_semantic_observability

# Innovation1 runtime parameters
float64 lambda_d
float64 r0
float64 mask_sigmoid_k
float64 beta
float64 min_semantic_evidence
int32 min_observation_num

# Observation-level values
float64 camera_height
float64 mu_v
float64 sigma_v
float64 distance
float64 view_angle
float64 mask_scale
float64 raw_confidence

# Fused cluster/category values after map update
float64 fused_confidence
int32 observation_num
int32 observation_cloud_sum
float64 observability
float64 quality_evidence

# Useful cluster state for failure analysis
float64 target_quality_evidence
float64 target_fused_confidence
int32 target_observation_num
bool target_passes_threshold
bool target_is_best_label

# Cluster-level top-two semantic evidence check.
# These values compare the largest and second-largest per-label s_tilde in
# the cluster after the current map update. The absolute difference is useful
# as a confidence margin against mixed-label false positives.
int32 top_label
float64 top_quality_evidence
float64 top_fused_confidence
int32 top_observation_num
int32 top_observation_cloud_sum
float64 top_observability
float64 top_distance
float64 top_view_angle
float64 top_mask_scale
bool top_passes_threshold

int32 second_label
float64 second_quality_evidence
float64 second_fused_confidence
int32 second_observation_num
int32 second_observation_cloud_sum
float64 second_observability
float64 second_distance
float64 second_view_angle
float64 second_mask_scale
bool second_passes_threshold

float64 top_second_abs_diff
bool target_is_top_label
