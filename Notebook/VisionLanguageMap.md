# ApexNav Vision-Language Map Workflow

This note documents how ApexNav extracts semantic information, projects it into map space, fuses it into semantic maps, and uses it for navigation.

## High-Level Flow

```text
Task target label / LLM answer
        |
        v
RGB image + depth image + camera pose
        |
        +--> Object semantic branch
        |       |
        |       +--> YOLOv7 / GroundingDINO detection
        |       +--> MobileSAM mask extraction
        |       +--> Mask + depth projection to world point cloud
        |       +--> ObjectMap2D cluster and confidence fusion
        |       +--> Direct object navigation
        |
        +--> Image-text value branch
                |
                +--> BLIP2 image-text cosine score
                +--> Project score over visible free-space grids
                +--> ValueMap2D confidence-weighted fusion
                +--> Semantic frontier selection
```

There are two semantic products:

1. **Object evidence**: detected target/related objects, represented as world-frame point clouds and object clusters.
2. **Semantic value**: a grid value estimating how relevant the current view is to the target, based on BLIP2 image-text matching.

## Runtime Topics

The Python perception node publishes per-agent semantic outputs:

```text
/blip2/agent_X/cosine_score
/detector/agent_X/clouds_with_scores
```

The C++ mapping layer subscribes to these in `MapROS::init()`:

- `src/planner/plan_env/src/map_ros.cpp`
- `/blip2/agent_X/cosine_score` updates each agent's latest ITM/cosine score.
- `/detector/agent_X/clouds_with_scores` provides object point clouds, detector confidence scores, and label indices.

The object message type is:

```text
src/planner/plan_env/msg/MultipleMasksWithConfidence.msg

sensor_msgs/PointCloud2[] point_clouds
float32[] confidence_scores
int32[] label_indices
```

## Semantic Extraction

### Target And Related Labels

The target label is read from the navigation task. ApexNav also uses LLM-derived related labels and room hints.

Key file:

```text
real_world_test_example/real_world_test_habitat.py
```

The per-agent perception class runs two synchronized callbacks:

- `sync_detect_callback()` extracts object masks and object point clouds.
- `sync_value_callback()` extracts the BLIP2 image-text score.

### Object Detection And Segmentation

Object extraction starts in:

```text
vlm/utils/get_object_utils.py
```

Main function:

```python
get_object(right_label, img, cfg, similar_answer)
```

The function splits labels into two groups:

- COCO labels use YOLOv7.
- Open-vocabulary or non-COCO labels use GroundingDINO.

Model clients:

```text
YOLOv7Client       -> port 12184
GroundingDINOClient -> port 12181
MobileSAMClient   -> port 12183
```

For each detection, ApexNav calls:

```python
get_segmentation(...)
```

This converts the detection box into a MobileSAM mask:

```python
object_mask = sam_segmentor.segment_bbox(img, bbox_denorm.tolist())
```

The output of object extraction is:

```text
segmented image
detector confidence scores
binary object masks
label indices
```

Label index convention:

```text
0  -> target label
>0 -> LLM-related or similar labels
```

### Image-Text Matching Score

Image-text relevance is computed in:

```text
vlm/utils/get_itm_message.py
```

Main function:

```python
get_itm_message_cosine(rgb_image, label, room)
```

If a room hint exists, the prompt is:

```text
Seems like there is a {room} or a {label} ahead?
```

Otherwise:

```text
Seems like there is a {label} ahead?
```

The BLIP2 client returns a cosine score. ApexNav publishes that score as:

```text
/blip2/agent_X/cosine_score
```

## Projection Into World Space

Object masks are projected through depth into 3D point clouds in:

```text
basic_utils/object_point_cloud_utils/object_point_cloud.py
```

Main function:

```python
get_object_point_cloud(cfg, observations, object_masks_list, agent_name)
```

For each binary object mask:

1. `extract_object_cloud()` keeps only depth pixels inside the mask.
2. Camera intrinsics convert masked pixels into local camera-frame 3D points.
3. `xyz_yaw_to_tf_matrix()` builds the camera-to-world transform.
4. `transform_points()` transforms local object points into world coordinates.
5. `convert_to_pointcloud2()` converts the result into ROS `PointCloud2`.

The local point conversion is in:

```text
basic_utils/object_point_cloud_utils/geometry_utils.py
```

The core pixel-to-point logic is:

```python
v, u = np.where(mask)
z = depth_image[v, u]
x = (u - depth_image.shape[1] // 2) * z / fx
y = (v - depth_image.shape[0] // 2) * z / fy
cloud = np.stack((z, -x, -y), axis=-1)
```

The projected object point clouds are then published through `MultipleMasksWithConfidence`.

## C++ Semantic Map Fusion

### ROS Input Layer

The main C++ entry point is:

```text
src/planner/plan_env/src/map_ros.cpp
```

Important callbacks:

```cpp
MapROS::itmScoreCallback(...)
MapROS::detectedObjectCloudCallback(...)
MapROS::depthPoseCallback(...)
```

`itmScoreCallback()` stores the latest BLIP2 cosine score per agent:

```cpp
agents_[agent_id].itm_score_ = msg->data;
```

`detectedObjectCloudCallback()` processes object detections:

1. Converts ROS `PointCloud2` messages into PCL clouds.
2. Downsamples with voxel filtering.
3. Removes points beyond depth range.
4. Stores over-depth target evidence for special handling.
5. Uses Euclidean clustering as DBSCAN-like filtering.
6. Creates `DetectedObject` records.
7. Sends objects to the shared map:

```cpp
map_->inputObjectCloud2D(detected_objects, detected_object_cluster_ids);
```

It also calls:

```cpp
getObservationObjectsCloud(agent_id, detected_object_cluster_ids);
```

This provides negative/observation evidence for objects that are expected in the current view but not detected.

### ObjectMap2D

Object fusion is implemented in:

```text
src/planner/plan_env/src/object_map2d.cpp
src/planner/plan_env/include/plan_env/object_map2d.h
```

The object map stores `ObjectCluster` records. Each cluster contains:

```text
2D object cells
3D point clouds per semantic class
confidence scores per semantic class
observation counts
2D and 3D bounding boxes
best semantic label
high-confidence good cells
```

Object insertion starts from:

```cpp
SDFMap2D::inputObjectCloud2D(...)
```

in:

```text
src/planner/plan_env/src/sdf_map2d.cpp
```

For each `DetectedObject`, ApexNav calls:

```cpp
object_map2d_->searchSingleObjectCluster(detected_object);
```

`ObjectMap2D::searchSingleObjectCluster()`:

1. Projects object points to 2D grid cells.
2. Checks whether those cells satisfy object-map constraints.
3. Searches nearby cells for an existing object cluster.
4. Merges into an existing cluster or creates a new cluster.
5. Updates the cluster's best semantic label.

Confidence fusion happens in:

```cpp
ObjectMap2D::fusionConfidenceScore(...)
```

The default fusion mode uses weighted averaging by accumulated observation point counts:

```cpp
w_last = (sum - n_now) / sum;
w_now = n_now / sum;
final_score = w_last * c_last + w_now * c_now;
```

Observation evidence is handled by:

```cpp
ObjectMap2D::inputObservationObjectsCloud(...)
```

If an existing object region is observed but not positively detected, confidence can be reduced. For the primary target label, the BLIP2 ITM score is used as context weighting.

### ValueMap2D

Semantic value fusion is implemented in:

```text
src/planner/plan_env/src/value_map2d.cpp
src/planner/plan_env/include/plan_env/value_map2d.h
```

The value map stores two grid buffers:

```cpp
vector<double> value_buffer_;
vector<double> confidence_buffer_;
```

During each depth/pose update, `MapROS::depthPoseCallback()` first updates occupancy and obtains visible free grids:

```cpp
map_->inputDepthCloud2D(agent.filtered_depth_cloud2d_, agent.camera_pos_, free_grids);
```

If an ITM score is available, the visible free grids receive semantic value:

```cpp
map_->value_map_->updateValueMap(camera_pos, camera_yaw, free_grids, agent.itm_score_);
```

`ValueMap::updateValueMap()` then fuses the current score into every visible free grid:

```cpp
double now_confidence = getFovConfidence(sensor_pos, sensor_yaw, pos);
double now_value = itm_score;

value_buffer_[adr] =
    (now_confidence * now_value + last_confidence * last_value) /
    (now_confidence + last_confidence);
```

The confidence is based on field-of-view angle:

```cpp
double value = std::cos(relative_angle / (fov_angle / 2) * (M_PI / 2));
return value * value;
```

This gives high confidence to cells near the camera center and lower confidence near the view edge.

## Navigation Usage

Semantic information is consumed in:

```text
src/planner/exploration_manager/src/exploration_manager.cpp
```

Main planner function:

```cpp
ExplorationManager::planNextBestPoint(...)
```

### Priority 1: Navigate To High-Confidence Objects

The planner first asks the object map for target-like object clouds:

```cpp
sdf_map_->object_map2d_->getTopConfidenceObjectCloud(object_clouds);
```

If high-confidence target objects exist, ApexNav plans directly to the nearest reachable object point:

```cpp
searchObjectPath(pos, object_cloud, out_next_pos, out_next_best_path)
```

This is the highest-priority semantic behavior. Object evidence can stop general exploration and trigger direct target navigation.

### Priority 2: Navigate To Over-Depth Object Evidence

If a target object appears beyond reliable depth range, ApexNav stores it in:

```cpp
object_map2d_->over_depth_object_cloud_
```

The planner tries this after high-confidence object navigation:

```cpp
searchObjectPath(pos, object_map2d_->over_depth_object_cloud_, ...)
```

This lets ApexNav move toward a suspected target even when the exact object geometry is partly outside the valid depth range.

### Priority 3: Semantic Frontier Selection

If no object target is reachable, ApexNav uses frontier exploration. Depending on policy, frontiers can be chosen by distance, semantic value, hybrid logic, or TSP distance.

Policy dispatch:

```cpp
ExplorationManager::chooseExplorationPolicy(...)
```

Semantic-only policy:

```cpp
ExplorationManager::findHighestSemanticsFrontierPolicy(...)
```

For each frontier:

1. Convert frontier position to grid index.
2. Read `value_map_` value near the frontier.
3. Use the max semantic value in a local neighborhood.
4. Sort frontiers by semantic value descending.
5. Break ties by distance.
6. Select the first reachable frontier.

Hybrid policy:

```cpp
ExplorationManager::hybridExplorePolicy(...)
```

Hybrid mode computes sorted semantic frontiers:

```cpp
getSortedSemanticFrontiers(cur_pos, frontiers, sem_frontiers);
```

It then calculates:

```text
mean semantic value
semantic standard deviation
max-to-mean ratio
```

If semantic contrast is strong enough, ApexNav exploits the semantic signal and plans over high-value frontiers with TSP. Otherwise, it falls back to closest-frontier exploration.

## Map Visualization

`MapROS::visCallback()` publishes semantic and geometric map outputs.

Useful visualization topics include:

```text
/grid_map/value_map
/grid_map/occupancy_object
/grid_map/all_object_cloud
/grid_map/filtered_object_cloud
/grid_map/over_depth_object_cloud
/grid_map/occupied
/grid_map/free
/grid_map/unknown
/grid_map/esdf
```

The semantic value map is published in:

```cpp
MapROS::publishValueMap()
```

It publishes `pcl::PointXYZI`, where intensity represents semantic value.

The object occupancy map is published in:

```cpp
MapROS::publishObjectMap()
```

## End-To-End Summary

```text
1. Receive target label.
2. LLM provides related labels and optional room context.
3. For each agent, synchronize RGB, depth, and sensor pose.
4. Object branch:
   - Detect target/related objects with YOLOv7 or GroundingDINO.
   - Segment detections with MobileSAM.
   - Project masks through depth into world-frame point clouds.
   - Publish clouds, detector scores, and label indices.
5. Value branch:
   - Query BLIP2 with a target/room prompt.
   - Publish image-text cosine score.
6. C++ map layer:
   - Fuse object clouds into ObjectMap2D.
   - Fuse BLIP2 score into ValueMap2D over visible free grids.
7. Planner:
   - First tries high-confidence target object navigation.
   - Then tries over-depth target evidence.
   - Then chooses exploration frontiers using distance, semantic value, hybrid logic, or TSP.
```

## Important Files

```text
real_world_test_example/real_world_test_habitat.py
vlm/utils/get_object_utils.py
vlm/utils/get_itm_message.py
basic_utils/object_point_cloud_utils/object_point_cloud.py
basic_utils/object_point_cloud_utils/geometry_utils.py
src/planner/plan_env/msg/MultipleMasksWithConfidence.msg
src/planner/plan_env/src/map_ros.cpp
src/planner/plan_env/src/sdf_map2d.cpp
src/planner/plan_env/src/object_map2d.cpp
src/planner/plan_env/src/value_map2d.cpp
src/planner/exploration_manager/src/exploration_manager.cpp
```

## Navigation Map Layers

ApexNav navigation is not driven by one single map. It uses several map layers built from different evidence sources.

### 1. Occupancy / SDF Map

Main files:

```text
src/planner/plan_env/src/sdf_map2d.cpp
src/planner/plan_env/include/plan_env/sdf_map2d.h
```

Source evidence:

```text
Depth image + camera pose
```

How it is built:

1. `MapROS::processDepthImage()` converts depth pixels into temporary world-frame 3D points.
2. `MapROS::filterPointCloudToXY()` filters points by height and projects valid obstacle points to XY.
3. `SDFMap2D::inputDepthCloud2D()` raycasts from the camera position to each XY endpoint.
4. Ray endpoints update occupied cells.
5. Cells along rays update free cells.
6. Untouched cells remain unknown.
7. Occupied cells are inflated for robot safety.
8. ESDF is updated for distance-to-obstacle queries.

What it contains:

```text
occupancy_buffer_          occupied / free / unknown log-odds state
occupancy_buffer_inflate_  inflated obstacle cells
distance_buffer_           ESDF distance values
virtual_ground_buffer_     virtual ground cells
local/global update bounds
```

Navigation function:

```text
Answers: where is safe to move?
```

Used for:

- collision checking,
- A* path search,
- frontier extraction,
- trajectory safety,
- ESDF distance queries,
- obstacle inflation.

Published visualization topics include:

```text
/grid_map/occupied
/grid_map/occupied_inflate
/grid_map/free
/grid_map/unknown
/grid_map/esdf
```

Important note:

```text
ApexNav does not need a persistent 3D obstacle map for normal borders/walls.
It computes temporary 3D depth points mainly to get valid ray endpoints and height filtering,
then inserts the result into a 2D occupancy/SDF map.
```

### 2. Object Map

Main files:

```text
src/planner/plan_env/src/object_map2d.cpp
src/planner/plan_env/include/plan_env/object_map2d.h
```

Source evidence:

```text
RGB object detections + SAM masks + depth projection + detector confidence
```

How it is built:

1. Python detects target/related objects with YOLOv7 or GroundingDINO.
2. MobileSAM converts boxes into masks.
3. Masked depth pixels are projected into world-frame object point clouds.
4. C++ receives the object clouds through `MultipleMasksWithConfidence`.
5. `MapROS::detectedObjectCloudCallback()` downsamples, filters, and clusters the clouds.
6. `SDFMap2D::inputObjectCloud2D()` forwards each object to `ObjectMap2D`.
7. `ObjectMap2D::searchSingleObjectCluster()` projects object cloud points into XY cells and merges them into object clusters.

What it contains:

```text
object_buffer_              grid cells marked as object cells
object_indexs_              grid cell -> object cluster id
objects_                    semantic object clusters
all_object_clouds_          latest filtered object cloud visualization
over_depth_object_cloud_    target evidence near/beyond valid depth range
```

Each `ObjectCluster` contains:

```text
2D object cells
3D point clouds per semantic label
confidence scores per label
observation counts per label
2D bounding box
3D bounding box
best semantic label
high-confidence good cells
```

Navigation function:

```text
Answers: has the target object been detected, and where should the robot go to reach it?
```

Used for:

- high-confidence target-object navigation,
- suspicious object fallback,
- over-depth target navigation,
- object confidence fusion across observations,
- rejecting/weakening object hypotheses with negative observations.

Planner usage:

```cpp
sdf_map_->object_map2d_->getTopConfidenceObjectCloud(object_clouds);
searchObjectPath(pos, object_cloud, out_next_pos, out_next_best_path);
```

Published visualization topics include:

```text
/grid_map/occupancy_object
/grid_map/all_object_cloud
/grid_map/filtered_object_cloud
/grid_map/over_depth_object_cloud
/object/clouds
```

Important note:

```text
Object clouds are 3D in world coordinates, but navigation mainly uses their XY projection.
The Z dimension is still kept for filtering, 3D bounding boxes, and visualization.
```

### 3. Value Map

Main files:

```text
src/planner/plan_env/src/value_map2d.cpp
src/planner/plan_env/include/plan_env/value_map2d.h
```

Source evidence:

```text
BLIP2 image-text cosine score + visible free-space grids
```

How it is built:

1. Python asks BLIP2 whether the current RGB image seems related to the target/room.
2. The cosine score is published to `/blip2/agent_X/cosine_score`.
3. C++ stores the latest score per agent.
4. During each depth update, occupancy raycasting returns visible free grids.
5. `ValueMap::updateValueMap()` assigns the current BLIP2 score to those visible grids.
6. Fusion is weighted by field-of-view confidence.

What it contains:

```text
value_buffer_       semantic relevance value per grid cell
confidence_buffer_  observation confidence per grid cell
```

Navigation function:

```text
Answers: which visible/explored area appears semantically related to the target?
```

Used for:

- semantic frontier ranking,
- hybrid exploration/exploitation decision,
- choosing high-value frontiers for TSP-style planning.

Planner usage:

```cpp
sdf_map_->value_map_->getValue(idx);
findHighestSemanticsFrontierPolicy(...);
hybridExplorePolicy(...);
```

Published visualization topic:

```text
/grid_map/value_map
```

Important note:

```text
The value map does not localize an object directly.
It stores target relevance over visible free cells, so frontiers near high-value regions can be prioritized.
```

### 4. Frontier Map

Main files:

```text
src/planner/plan_env/src/frontier_map2d.cpp
src/planner/plan_env/include/plan_env/frontier_map2d.h
```

Source evidence:

```text
Occupancy / free / unknown boundaries from the SDF map
```

What it contains:

```text
active frontier clusters
dormant frontier clusters
frontier average positions
frontier claim state for multi-agent coordination
```

Navigation function:

```text
Answers: where are the useful exploration candidates?
```

Used for:

- frontier extraction,
- frontier filtering,
- frontier clustering,
- multi-agent frontier claiming,
- candidate generation for distance, semantic, hybrid, or TSP policies.

Important note:

```text
The frontier map is a derived planning layer, not a raw sensor-fusion map.
It depends on the occupancy/SDF map to know where free space meets unknown space.
```

## Map Layer Summary

```text
Occupancy / SDF Map
  Input: depth + pose
  Contains: occupied, free, unknown, inflated obstacles, ESDF
  Function: safe navigation and collision checking

Object Map
  Input: object masks + depth-projected clouds + detector confidence
  Contains: object cells, object clusters, labels, confidence, over-depth target evidence
  Function: direct target-object navigation

Value Map
  Input: BLIP2 image-text cosine score + visible free grids
  Contains: semantic relevance value and confidence per grid cell
  Function: semantic frontier prioritization

Frontier Map
  Input: occupancy/free/unknown boundary
  Contains: frontier clusters, averages, dormant frontiers, multi-agent claims
  Function: exploration candidate management
```

The planner's practical priority order is:

```text
1. Use ObjectMap2D to navigate directly to high-confidence target objects.
2. Use over-depth object evidence if target evidence is seen near the depth limit.
3. Use FrontierMap2D to get exploration candidates.
4. Use ValueMap2D to rank or filter frontiers in semantic/hybrid policies.
5. Use Occupancy/SDF map for all path feasibility and safety checks.
```
