# Over-Depth Object Cloud Cache

This branch extends the temporary cache for `over_depth_object_cloud_`.

When an over-depth target object is detected, `MapROS::objectCallback()` still clears and rebuilds
`object_map2d_->over_depth_object_cloud_` on each object callback. The short consistency cache now
keeps the previous over-depth cloud for up to 15 callback cycles instead of 4.

This makes `SEARCH_OVER_DEPTH_OBJECT` less likely to drop immediately when the target is near the
depth sensor limit or temporarily disappears from one frame.

Relevant runtime signal:

```bash
rostopic echo /ros/agent_0/exploration_strategy
rostopic echo /grid_map/over_depth_object_cloud
```

# Sparse Object Cloud Tuning

This branch relaxes object detection and object-cloud clustering so sparse detections, such as
GroundingDINO cabinet detections, are less likely to disappear before entering
`/grid_map/filtered_object_cloud`.

GroundingDINO thresholds are set to:

```yaml
yolo:
  confidence_threshold_yolo: 0.20

groundingDINO:
  confidence_threshold_dino: 0.30
  text_threshold: 0.20
```

The YOLO confidence threshold is reduced from `0.30` to `0.20`. The YOLO IOU threshold remains
unchanged because it controls NMS overlap suppression rather than initial detection confidence.

The object point-cloud DBSCAN step in `MapROS::detectedObjectCloudCallback()` now uses:

```cpp
dbscan(single_object_cloud, 0.15f, 6)
```

This increases the clustering radius from `0.12` to `0.15` and lowers the minimum cluster size from
`10` points to `6` points, making `/grid_map/filtered_object_cloud` less aggressive for sparse
object masks.
