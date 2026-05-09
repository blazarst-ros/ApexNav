# Fuse c Weighted Accepted Data Summary

Source: `Agent_0_info.txt`

Note: despite the filename, the accepted records in this capture are for `agent_id = 1`.

## Policy Context

The current fusion policy fuses confidence `c`, then computes semantic evidence `s`.

```text
c_fused = weighted fusion of historical confidence and current confidence
s_{a,i}^k = rho * c_fused * (1 - exp(-beta * observation_num))
rho = H * D * A * M
```

Runtime parameters:

```text
lambda_d = 0.25
r0 = 0.03
beta = 0.8
strict target gate: s >= 0.15 and observation_num >= 4
```

## Accepted Categories

| Label | Accepted Count | Strict s>=0.15 Count | Target Pass Count | Best Label Match Count | Max s | Max Cluster | Max Best Label | Max Distance | Max Mask Scale | Max c_fused | Max Obs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6 | 2 | 3 | 4 | 0.189764 | 0 | 0 | 1.504 | 0.036709 | 0.314842 | 4 |
| 1 | 5 | 1 | 0 | 3 | 0.154612 | 3 | 1 | 1.735 | 0.177048 | 0.512741 | 9 |
| 3 | 25 | 1 | 2 | 11 | 0.153604 | 6 | 3 | 1.378 | 0.240726 | 0.919181 | 6 |

## Representative Accepted Data

### 1. Strong Target Success

| Field | Value |
|---|---:|
| agent | 1 |
| cluster | 0 |
| label | 0 |
| best_label | 0 |
| s / quality_evidence | 0.189764 |
| observation_num | 4 |
| c_fused | 0.314842 |
| raw_confidence | 0.601074 |
| rho | 0.628341 |
| distance | 1.504 |
| view_angle | 0.172 |
| mask_scale | 0.036709 |
| target_passes_threshold | True |
| target_is_best_label | True |

This is the cleanest target case. It satisfies both the strict evidence gate and the repeated-observation gate.

### 2. Weak Target, Accepted by Old Policy but Rejected by Strict Policy

| Field | Value |
|---|---:|
| agent | 1 |
| cluster | 0 |
| label | 0 |
| best_label | 0 |
| s / quality_evidence | 0.050883 |
| observation_num | 2 |
| c_fused | 0.391631 |
| raw_confidence | 0.418457 |
| rho | 0.162793 |
| distance | 3.019 |
| view_angle | 0.315 |
| mask_scale | 0.012184 |
| target_passes_threshold | True in old log |
| target_is_best_label | True |

This is why the new policy raises thresholds. It barely passes the old `s >= 0.05, n >= 2` gate but is too weak for a false-positive-intolerant setting.

### 3. Strong Non-Target Label 1

| Field | Value |
|---|---:|
| agent | 1 |
| cluster | 3 |
| label | 1 |
| best_label | 1 |
| s / quality_evidence | 0.154612 |
| observation_num | 9 |
| c_fused | 0.512741 |
| raw_confidence | 0.708496 |
| rho | 0.301766 |
| distance | 1.735 |
| view_angle | 0.795 |
| mask_scale | 0.177048 |
| target_passes_threshold | False |
| target_is_best_label | False |

This is strong evidence, but it is not the target label. It should not trigger target success.

### 4. Strong Non-Target Label 3

| Field | Value |
|---|---:|
| agent | 1 |
| cluster | 6 |
| label | 3 |
| best_label | 3 |
| s / quality_evidence | 0.153604 |
| observation_num | 6 |
| c_fused | 0.919181 |
| raw_confidence | 0.941895 |
| rho | 0.168496 |
| distance | 1.378 |
| view_angle | 0.272 |
| mask_scale | 0.240726 |
| target_passes_threshold | False |
| target_is_best_label | False |

This shows why raw confidence alone is not a safe success criterion. A non-target object can have very high detector confidence and high fused confidence.

### 5. Label 3 Distractor on Cluster 0

| Field | Value |
|---|---:|
| agent | 1 |
| cluster | 0 |
| label | 3 |
| best_label | 1 |
| s / quality_evidence | 0.079388 |
| observation_num | 27 |
| c_fused | 0.651242 |
| raw_confidence | 0.829590 |
| rho | 0.121903 |
| distance | 1.781 |
| view_angle | 0.532 |
| mask_scale | 0.164316 |
| target_passes_threshold | False |
| target_is_best_label | False |

This is persistent but still not target-valid. Repetition alone is insufficient without target label agreement and enough evidence.

## Takeaways

1. Label `0` is the only target label in the current policy.
2. Confidence fusion is applied to `c`, not directly to `s`.
3. Non-target labels can produce strong `s`; final target success must require `label == 0`, `best_label == 0`, sufficient `s`, and enough observations.
4. The stricter policy keeps the strongest target case while rejecting weak early target claims.
