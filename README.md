# Branch: I2S1

**Functions:** Selection of the height geometric indicator function and determination of the coefficients

**Update to:** `049971b613fcd2554b034bd4a853d79cc1e61613`

**Not included:** Experimental data

**Feature:** Multi-agent

**Language prior:** Added

**System:** Height heterogeneous system

## Runtime Exploration Settings

This branch runs the multi-agent evaluation with heterogeneous camera heights:

| Agent | Height | RGB sensor y | Depth sensor y |
| --- | ---: | ---: | ---: |
| `agent_0` | `0.7 m` | `0.7 m` | `0.7 m` |
| `agent_1` | `1.2 m` | `1.2 m` | `1.2 m` |
| `agent_2` | `1.7 m` | `1.7 m` | `1.7 m` |

The evaluation success distance is:

```text
0.5 m
```

Config key:

```yaml
success_distance: 0.5
```

These values are applied consistently in:

- `config/habitat_eval_hm3dv1.yaml`
- `config/habitat_eval_hm3dv2.yaml`
- `config/habitat_eval_mp3d.yaml`

`habitat2ros.ROSPublisher` receives the configured camera height from
`habitat_evaluation.py`, so the ROS sensor pose z-offset matches the Habitat
RGB/depth sensor height.

Stage 1 detection records are published on:

```text
/stage1/detector/detection
```

and written as JSONL files under:

```text
/media/blazarst/Getea/RuntimeData/Stage1_detector
```
