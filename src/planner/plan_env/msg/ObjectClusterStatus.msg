uint8 TARGET=0
uint8 CONFUSION=1
uint8 UNCERTAIN=2

int32 cluster_id
geometry_msgs/Point centroid
uint8 state
int32 best_label_index
string best_label_name
plan_env/ObjectLabelStatus[] labels
