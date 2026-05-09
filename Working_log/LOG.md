# Working Log

## 2026-05-09

- Refactored semantic observability mask-scale quality from linear clipping `min(mask_scale / r0, 1)` to sigmoid `1 / (1 + exp(-mask_sigmoid_k * (mask_scale - r0)))`.
- Added runtime parameter `object/mask_sigmoid_k`, default/recommended value `300.0`.
- Reinterpreted `object/r0` as the sigmoid midpoint and set launch values to `0.007`.
- Updated semantic evidence debug message to include `mask_sigmoid_k`.
- Verified the ROS workspace with `catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3`; build passed.
