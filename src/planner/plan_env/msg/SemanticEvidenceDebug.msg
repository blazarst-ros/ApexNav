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
