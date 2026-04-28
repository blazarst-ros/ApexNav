

# 1. 总体分类

当前 ZSON 方法大体可以分为 5 类：

| 类别                   | 代表方法              | 核心思想                  | 主要问题          |
| -------------------- | ----------------- | --------------------- | ------------- |
| 几何 frontier baseline | CoW               | 只基于几何 frontier 搜索     | 不利用语义，效率低     |
| VLM value map        | VLFM              | 用 VLM 给方向/frontier 打分 | 易受语义误导        |
| LLM semantic map     | L3MVN / TriHelper | 构建语义地图，用 LLM 选目标区域    | 推理强，但依赖语义质量   |
| 3D scene graph + LLM | SG-Nav            | 建 3D 场景图，再用 LLM 推理    | 表达强，但系统复杂     |
| 自适应语义-几何方法           | SemUtil / ApexNav | 语义强时用语义，语义弱时用几何       | 仍是单机为主，多机协同不足 |

ApexNav 论文中也指出：当前 ZSON 方法主要依赖 frontier-based exploration；CoW 只用最近 frontier，L3MVN/TriHelper 构建语义地图并用 LLM 选择 frontier，VLFM 用 VLM 构建 value map，SG-Nav 用 3D scene graph 提示 LLM，而这些方法常用贪心策略选择最高分 frontier。

---

# 2. CoW：几何优先的零样本导航 baseline

## 核心思路

CoW, **Cows on Pasture**，更像一个强 baseline。它使用开放词汇视觉模型或检测器寻找目标，但在探索阶段主要依赖 **nearest frontier**。也就是说，机器人不知道目标在哪，就优先去最近的未知边界。

```text
RGB-D observation
→ open-vocabulary detection
→ if target found: navigate
→ else: nearest frontier exploration
```

## 优点

| 优点     | 说明                 |
| ------ | ------------------ |
| 简单     | 不需要复杂 LLM 推理       |
| 稳定     | 不太会被错误语义强行带偏       |
| 计算量较低  | 适合做 baseline       |
| 泛化性还可以 | 使用开放词汇模型，不依赖目标类别训练 |

## 缺点

| 缺点       | 说明                                   |
| -------- | ------------------------------------ |
| 语义利用不足   | 不知道“厕所更可能在 bathroom”，“床更可能在 bedroom” |
| SPL 容易差  | 可能绕很多无关区域                            |
| 不够智能     | 本质是盲目探索 + 目标检测                       |
| 多机扩展价值有限 | 多机情况下容易重复探索，除非加任务分配                  |

## 对你项目的启发

CoW 可以作为你的 **最低 baseline**：

```text
Multi-agent CoW = 多机器人最近 frontier + 开放词汇检测
```

你的方法要证明比它强，关键在于：

1. 更少重复探索；
2. 更快找到目标；
3. 更少误检；
4. 通信量不要爆炸。

---

# 3. VLFM：Vision-Language Frontier Map

## 核心思路

VLFM 的核心是用 VLM 构建 **value map / frontier map**。简单说，它让视觉语言模型判断当前区域或 frontier 与目标之间的相关性。

例如目标是 `toilet`，VLFM 会更倾向于探索看起来像 bathroom、sink、washing machine 附近的区域。

```text
RGB observation + target text
→ VLM similarity / value prediction
→ build value map
→ choose high-value frontier
```

ApexNav 论文指出，VLFM 使用 VLM 建立 value map，将 frontier 与目标联系起来。

## 优点

| 优点              | 说明                      |
| --------------- | ----------------------- |
| 语义导航能力强         | 能利用 VLM 的视觉-语言常识        |
| 比纯 frontier 更高效 | 不再盲目搜索                  |
| zero-shot 能力好   | 不需要为每类目标训练策略            |
| 易与 ApexNav 结合   | semantic score map 思路相近 |

## 缺点

| 缺点         | 说明                          |
| ---------- | --------------------------- |
| 依赖 VLM 质量  | VLM 对 indoor scene 的判断不一定稳定 |
| 语义弱时会失效    | 白墙、走廊、遮挡场景下，VLM 给不出有效方向     |
| 容易被虚假相关性误导 | 看到 plant、table 等泛化物体时可能误判   |
| 单帧/局部视野限制  | 当前视角看不到上下文时，value map 会偏    |

## 对你项目的启发

你可以把 VLFM 的 value map 升级成：

```text
Multi-agent Semantic-Belief Value Map
```

即每个 agent 不只给自己的 frontier 打分，还把语义价值压缩成节点级信息共享：

```text
frontier_id, target_score, uncertainty, observed_by_agent
```

这样比单机 VLFM 更强。

---

# 4. L3MVN：LLM 辅助的视觉目标导航

## 核心思路

L3MVN 使用 LLM 进行高层语义推理。它通常会利用检测到的物体、房间线索、目标类别，让 LLM 判断下一步应该去哪里。

例如：

```text
Target: microwave
Observed: sofa, TV, table
LLM reasoning: likely not kitchen, continue exploration
Observed: sink, cabinet
LLM reasoning: likely kitchen, prioritize this area
```

ApexNav 论文将 L3MVN 归为“构建语义地图并使用 LLM 识别 target-related frontiers”的方法。

## 优点

| 优点            | 说明            |
| ------------- | ------------- |
| 常识推理强         | LLM 知道物体-房间关系 |
| 可解释性好         | 可以解释为什么去某个区域  |
| zero-shot 能力强 | 不需要目标类别训练数据   |
| 适合长程导航        | LLM 可做高层规划    |

## 缺点

| 缺点               | 说明                      |
| ---------------- | ----------------------- |
| 推理成本高            | LLM 调用慢，在线系统压力大         |
| 不确定性处理弱          | LLM 输出通常不是严格概率          |
| 容易 hallucination | 可能生成不符合当前地图的推理          |
| 与几何约束耦合弱         | LLM 说“去厨房”，但机器人仍要知道可达路径 |
| 多机通信未解决          | 多个 agent 的语义观察如何融合，不是重点 |

## 对你项目的启发

L3MVN 适合做 **高层语义先验模块**，但不能作为系统核心。你应该避免让 LLM 直接决定所有 agent 行为，而是让 LLM 只生成：

```text
target-room prior
similar-object list
object co-occurrence prior
```

然后由你的 **Semantic-Belief MR-DTG** 做规划和分配。这样系统更可控。

---

# 5. TriHelper：动态辅助的 ZSON

## 核心思路

TriHelper 强调动态帮助导航，通常会使用语义地图、LLM/VLM 推理和辅助验证机制来减少错误决策。ApexNav 论文中提到，TriHelper 构建 semantic map 并使用 LLM 选择 target-related frontiers，同时也引入 VLM refinement 来缓解误检测。

## 优点

| 优点                  | 说明                   |
| ------------------- | -------------------- |
| 比普通 LLM frontier 更稳 | 有辅助验证机制              |
| 目标相关区域选择更智能         | 能结合语义地图              |
| 对误检有一定缓解            | VLM refinement 可重新判断 |
| 可解释性较强              | 语义地图 + 推理过程可视化       |

## 缺点

| 缺点         | 说明                 |
| ---------- | ------------------ |
| 仍依赖语义质量    | 语义图错了，后续推理也错       |
| 验证机制通常是单机的 | 没有利用多机互补视角         |
| 可能计算较重     | LLM/VLM 多次调用       |
| 贪心选择问题仍存在  | 高分 frontier 未必全局最优 |

## 对你项目的启发

TriHelper 证明了“验证”重要，但它的验证主要是 **模型内部验证**。你的多机项目可以更进一步：

```text
model verification → multi-agent physical verification
```

即：

```text
一个 agent 发现候选目标
→ 另一个高度/视角不同的 agent 去确认
→ 多视角一致才停止
```

这是你可以超越 TriHelper 的地方。⚙️

---

# 6. SG-Nav：3D Scene Graph + LLM

## 核心思路

SG-Nav 将环境组织为 **3D scene graph**，节点可以是物体、区域、空间关系，再用 LLM 在 scene graph 上推理目标位置。

```text
RGB-D observation
→ object detection / segmentation
→ 3D scene graph construction
→ LLM prompt reasoning
→ frontier / waypoint selection
```

ApexNav 论文指出，SG-Nav 使用 3D scene graph 来 prompt LLM 选择 frontiers，并且结合多帧分数和 3D scene graph 进行验证。

## 优点

| 优点        | 说明                          |
| --------- | --------------------------- |
| 结构化表达强    | 比 2D semantic map 更接近“环境理解” |
| 可解释性强     | 物体、区域、关系都可读                 |
| 适合复杂语义推理  | 如 “chair near dining table” |
| 与 LLM 很匹配 | LLM 擅长处理图结构文本描述             |

## 缺点

| 缺点         | 说明                          |
| ---------- | --------------------------- |
| 构图复杂       | 需要稳定检测、分割、关联、去重             |
| 误检传播严重     | 物体节点错了，图推理会错                |
| 实时性压力大     | 3D graph 更新 + LLM 推理开销高     |
| 多机图合并困难    | 多机器人观察到的 graph node 如何对齐是难点 |
| 对小物体和遮挡仍脆弱 | scene graph 依赖检测质量          |

## 对你项目的启发

SG-Nav 的 scene graph 很强，但直接多机共享完整 scene graph 会很重。你可以采用折中方案：

```text
MR-DTG backbone + semantic object belief nodes
```

也就是：

```text
不是完整 3D Scene Graph
而是任务相关 Semantic-Belief Topological Graph
```

这样既保留语义推理能力，又符合 MR-DTG 的低通信优势。

---

# 7. SemUtil：固定阶段的语义-几何切换

## 核心思路

SemUtil 注意到早期语义引导可能不可靠，所以先进行一段几何探索，再切换到语义探索。ApexNav 论文中提到，SemUtil 采用固定两阶段策略：前 50 steps 用几何探索，之后再切到语义；但这种固定机制缺乏适应性。

## 优点

| 优点         | 说明            |
| ---------- | ------------- |
| 比纯语义更稳     | 早期不盲信语义       |
| 比纯几何更智能    | 后期使用语义        |
| 实现简单       | 固定步数切换        |
| 适合语义早期稀疏场景 | 起步阶段不会被空白视野误导 |

## 缺点

| 缺点        | 说明                       |
| --------- | ------------------------ |
| 固定阈值僵硬    | 50 steps 对不同环境不一定合适      |
| 无法响应语义突变  | 早期看到强语义也可能不用             |
| 无法处理弱语义目标 | 后期语义仍可能不可靠               |
| 不适合多机     | 每个 agent 的语义成熟度不同，不应统一切换 |

## 对你项目的启发

你可以把 SemUtil 的固定切换升级成多机版本：

```text
team-level adaptive semantic readiness
```

例如：

```text
如果全队共享图中 semantic uncertainty 低，进入语义协同；
如果局部区域语义弱，则派 explorer 扩展拓扑；
如果候选目标不确定，则派 verifier 验证。
```

这比固定步数更自然。

---

# 8. ApexNav：自适应探索 + target-centric semantic fusion

## 核心思路

ApexNav 针对两个问题：

1. 语义线索强弱不稳定；
2. 开放词汇检测容易误检。

它构建 frontier map、semantic score map 和 target-centric semantic map。如果发现可靠目标，就导航到目标；否则根据语义分布强弱，在 semantic-based exploration 和 geometry-based exploration 之间切换。

ApexNav 的 semantic score map 使用 BLIP-2 计算图像与文本 prompt 的相似度，并结合视角置信度投影到 2D map；其 adaptive exploration 会根据 frontier semantic score 的 max-to-mean ratio 和标准差判断语义线索是否足够强。

## 优点

| 优点        | 说明                                          |
| --------- | ------------------------------------------- |
| 语义-几何自适应  | 不盲信语义，也不完全盲探                                |
| 抗误检能力更强   | target-centric semantic fusion 维护目标与相似物长期记忆 |
| SPL 较好    | 语义强时可快速靠近目标区域                               |
| 工程完整      | 有 mapping、detection、planning、navigation     |
| 适合你作为基础框架 | 可以直接扩展成多机                                   |

## 缺点

| 缺点           | 说明                             |
| ------------ | ------------------------------ |
| 单机框架         | 没有天然多机协同                       |
| 语义共享缺失       | 其他 agent 无法利用当前 agent 的语义记忆    |
| 目标验证仍局限于单机视角 | 对遮挡、小物体、相似物仍可能失败               |
| 弱语义目标仍困难     | ApexNav 也承认目标与环境语义弱相关时，语义引导会失效 |
| 小物体细粒度探索不足   | 传感器视野内区域被默认充分探索时，小物体可能漏检       |

ApexNav 自身也指出了限制：假设目标可见，隐藏目标需要交互；小物体可能需要更细粒度探索；当目标与周围环境语义相关性弱时，语义引导会失效。

## 对你项目的启发

ApexNav 是你最好的单机 base。你要做的不是推翻它，而是扩展：

```text
ApexNav target-centric fusion
→ Multi-agent target-centric belief fusion

ApexNav adaptive exploration
→ Multi-agent belief-utility task allocation

ApexNav semantic score map
→ Shared semantic-belief topological graph
```

---

# 9. STRM / 语义拓扑图方法对 ZSON 的启发

严格说，STRM 更偏 VLN-CE，不是标准 ObjectNav ZSON。但它对你的项目非常重要，因为它提供了 **语义信息节点化** 的范式。

STRM 通过场景理解辅助任务识别区域和物体，构建空间邻近知识库；导航过程中逐步生成语义拓扑图，并结合指令中的物体、区域信息进行推理定位。

## 优点

| 优点         | 说明                      |
| ---------- | ----------------------- |
| 节点级语义清晰    | 每个节点有区域、物体、视觉特征         |
| 可解释性强      | 可以说明为什么选择某个子目标          |
| 适合拓扑导航     | 比 dense semantic map 更轻 |
| 可结合 MR-DTG | 很适合转成多机共享图结构            |

## 缺点

| 缺点           | 说明                                           |
| ------------ | -------------------------------------------- |
| 主要面向 VLN     | 输入是语言指令，不是单一 object goal                     |
| 多机机制不足       | 没有通信、任务分配、冲突融合                               |
| 依赖训练         | 文本编码、跨模态图编码器等通常需要训练                          |
| 与开放词汇检测结合不充分 | 对 ZSON 的 noisy detection 问题处理不如 ApexNav 针对性强 |

## 对你项目的启发

你可以借鉴 STRM 的节点设计，而不是照搬其训练框架：

```text
STRM semantic node
+ MR-DTG communication-efficient topology
+ ApexNav target-centric fusion
= Multi-agent semantic-belief topological ZSON
```

---

# 10. 横向对比表

| 方法        | 语义来源                       | 地图形式                            | 决策方式             | 优点          | 缺点        | 对你项目价值             |
| --------- | -------------------------- | ------------------------------- | ---------------- | ----------- | --------- | ------------------ |
| CoW       | 开放词汇检测                     | 几何 frontier                     | 最近 frontier      | 简单、稳定       | 语义弱，效率低   | baseline           |
| VLFM      | VLM                        | value/frontier map              | 高 value frontier | 语义导航强       | 语义弱场景失效   | semantic score 可借鉴 |
| L3MVN     | LLM + 视觉语义                 | semantic map                    | LLM 选 frontier   | 常识强，可解释     | 慢，易幻觉     | 高层先验               |
| TriHelper | LLM/VLM                    | semantic map                    | 动态辅助             | 有验证意识       | 单机验证，不够系统 | 主动验证 baseline      |
| SG-Nav    | 3D scene graph + LLM       | 3D scene graph                  | 图推理              | 表达力强        | 构图复杂，实时难  | 语义图节点设计            |
| SemUtil   | VLM/语义分数                   | frontier map                    | 固定阶段切换           | 稳定简单        | 不自适应      | 对比 ApexNav         |
| ApexNav   | VLM + detector + LLM prior | semantic score map + target map | 自适应语义/几何         | 综合强，适合 base | 单机，多机不足   | 核心基础框架             |
| STRM      | 区域/物体识别 + Transformer      | semantic topological graph      | 拓扑推理             | 节点化语义强      | 偏 VLN，需训练 | 语义节点化参考            |

---

# 11. 对你的多机 ZSON 项目的结论

这些方法共同暴露出 4 个缺口：

## 11.1 缺口一：大多数方法是单机

CoW、VLFM、L3MVN、TriHelper、SG-Nav、ApexNav 主要围绕单 agent。它们没有系统解决：

```text
多个 agent 如何共享语义？
如何避免重复探索？
如何处理语义冲突？
谁去验证候选目标？
```

这正是你的多机创新空间。

---

## 11.2 缺口二：语义地图通常不是通信友好的

许多方法使用 dense semantic map、value map 或 3D scene graph。单机可以，但多机共享会带来：

```text
高通信量
节点对齐困难
语义冲突难处理
实时性下降
```

所以你应该采用：

```text
Semantic-Belief MR-DTG
```

而不是完整共享 dense semantic map。

---

## 11.3 缺口三：目标检测误差仍是核心瓶颈

ApexNav 指出，ZSON 成功率直接受目标识别影响；单帧检测容易误判，max-confidence fusion 也容易被偶发高置信误检污染。

因此你的多机系统应该强调：

```text
multi-agent active verification
multi-view target confirmation
heterogeneous viewpoint validation
```

这比单机 fusion 更有说服力。

---

## 11.4 缺口四：语义探索与几何探索仍缺少多机层面的统一决策

ApexNav 解决了单机 semantic/geometry switching。你的工作可以进一步提出：

```text
team-level semantic-geometric coordination
```

例如：

| 情况       | 多机策略                   |
| -------- | ---------------------- |
| 全局语义弱    | 多 agent 分散几何探索         |
| 某区域目标信念高 | 派最近 agent 前往           |
| 候选目标不确定  | 派异构 agent 验证           |
| 语义冲突严重   | 派 best-view agent 重新观测 |
| 通信受限     | 只共享高价值 belief delta    |

---

# 12. 最适合你的论文定位

你不应该把论文写成：

```text
我们把 ApexNav 改成多机版本。
```

更强的说法是：

```text
现有 ZSON 方法大多依赖单机语义推理，缺少面向多机器人协同的语义表示、低通信共享和主动验证机制。本文提出一种基于 Semantic-Belief Topological Graph 的多机 ZSON 框架，在保留 ApexNav 自适应探索优势的基础上，引入 MR-DTG 式低通信拓扑共享、节点化语义信念融合和多视角主动验证，实现更高效、更可靠的目标发现。
```

---

# 13. 推荐你重点对标的 baseline

你的实验里建议重点对比这些：

| Baseline                        | 为什么必须有                               |
| ------------------------------- | ------------------------------------ |
| Single-agent ApexNav            | 证明多机收益                               |
| Multi-agent independent ApexNav | 证明不是简单多机器人数量优势                       |
| Multi-agent CoW                 | 证明语义协同有效                             |
| Multi-agent VLFM-style sharing  | 证明你的 belief graph 优于简单 value sharing |
| MR-DTG-only exploration         | 证明只有拓扑协同不够                           |
| SG-Nav / TriHelper 单机结果         | 作为语义推理强 baseline                     |
| Full map sharing multi-agent    | 证明你低通信仍能接近或超过高通信方案                   |

---

