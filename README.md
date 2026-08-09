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
