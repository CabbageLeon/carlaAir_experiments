# Air-Ground Collaborative VLN 实验计划

> 对应论文: *A Training-Free Baseline for Air-Ground Collaborative VLN* (ICRA 2027)

---

## 前置准备

### 环境确认

- [ ] CARLA-Air 能正常启动 (`./carlaAir.sh Town03`)
- [ ] conda 环境 `carlaAir` 可用，`python -c "import carla; import airsim"` 通过
- [ ] VLM API 可用（DashScope / 千问），`OPENAI_API_KEY` 已设
- [ ] 系统代理已关闭（避免 `socks://` 报错）

### 实验目录

所有实验输出放在 `runs/paper/` 下：
```
runs/paper/
├── smoke/          # 冒烟测试
├── single_uav/     # UAV-only 基线
├── single_ugv/     # UGV-only 基线  
├── collab_l0/      # L0 无通信
├── collab_l1/      # L1 坐标
├── collab_l2/      # L2 坐标+置信度
├── collab_l3/      # L3 坐标+语义
├── collab_l4/      # L4 坐标+图块
├── collab_l5/      # L5 坐标+全图
├── feedback_abl/   # 反馈消融
├── backbone_abl/   # VLM 主干消融
└── robustness/     # 鲁棒性测试
```

---

## Phase 0 — Episode 数据准备

**目标:** 定义评估用的导航 episodes。

### 0.1 设计 episode 格式

每个 episode 包含：

| 字段 | 说明 | 示例 |
|------|------|------|
| `id` | 唯一标识 | `"town03_001"` |
| `map` | CARLA 地图名 | `"Town03"` |
| `instruction` | 自然语言指令 | `"到桥边的红色建筑"` |
| `goal` | 目标点世界坐标 | `(125.0, 84.3, 0.0)` |
| `uav_spawn` | UAV 出生点 + 高度 | `(100, 50, 50)` |
| `ugv_spawn` | UGV 出生点 | `Carla 出生点索引` |
| `time_budget_s` | 时间限制 | `180` |
| `tags` | 场景标签 | `["urban", "cross_intersection"]` |

### 0.2 编写 episodes

1. 打开 CARLA-Air (`./carlaAir.sh`)，手动飞 UAV 查看场景
2. 选取有代表性的起点-目标点对，确保需要 UAV 鸟瞰指引 UGV（如跨街区、有遮挡）
3. 为每对写一条自然语言指令（中文或英文）
4. 先手写 **10-15 个 episode**（覆盖 Town01-03），保存为 `episodes/*.json`
5. 目标：最终扩展到 **115 episodes × 11 环境**（对应 AirGroundBench 协议）

### 0.3 验证 episodes

- [ ] Goal 在可导航地面上（不在建筑内部）
- [ ] UGV spawn 和 goal 之间有可通行路径
- [ ] UAV 从 spawn 高度能看到 goal 区域
- [ ] 单个 UGV 无法凭局部视角直接看到 goal（这样才能体现协作价值）

---

## Phase 1 — Single-Agent 基线

**目标:** 获取 Table 1 上半部分数据——UAV 单独导航和 UGV 单独导航的表现，作为协作增益的比较基准。

### 1.1 UAV-only 基线

**任务:** UAV 从空中独立完成导航，不依赖 UGV。

**流程:**
1. UAV 在 episode 指定的高度悬停，**相机改为向下**（模拟鸟瞰视角）
2. 每 2 秒执行一次 VLM 推理：
   - 输入: 当前鸟瞰图 + instruction
   - 输出: 2D waypoint 像素 + 深度标签
3. 将 waypoint 投影到地面 3D 坐标
4. UAV 飞向该坐标
5. 180 秒内到达 goal 3m 内 → 成功；超时或碰撞 → 失败

**运行参数:**
| 参数 | 值 |
|------|-----|
| VLM | qwen-vl-max |
| 决策频率 | 0.5 Hz（~2s 推理延迟） |
| 每 episode 时间 | 180s |
| 重复次数 | 每 episode × 3 seeds（11, 22, 33） |
| 指标 | SR, SPL, NE |

**输出:** 每个 episode 的轨迹数据 + `summary.json`

### 1.2 UGV-only 基线

**任务:** UGV 从地面独立完成导航。

**流程:**
1. UGV 在 episode spawn 点出生，相机前向
2. 每步执行地面 SPF：
   - 输入: 前视图 + instruction
   - 输出: 局部 waypoint
3. UGV 沿路行驶到 waypoint
4. 遇到死胡同/遮挡时自主绕行（无 UAV 帮助）

**运行参数:** 同 1.1

**输出:** 同 1.1

### 1.3 结果计算

对每个 episode 计算：`CG = SR_collab - max(SR_uav, SR_ugv)`

Paper 预期: UAV-only 和 UGV-only 的 SR 都较低（各自有盲区），协作后应明显提升。

---

## Phase 2 — 空地协同系统

**目标:** 实现完整的 UAV-UGV 协同导航，获取 Table 1 下半部分数据。

### 2.1 通信协议定义

UAV → UGV 消息内容（按通信级别）：

| Level | 内容 | 单条大小 | 说明 |
|-------|------|----------|------|
| L0 | 无 | 0 B | 各自独立导航，不通信 |
| L1 | 目标 XYZ 坐标 | 12 B | 3×float32 |
| L2 | L1 + 置信度 | 16 B | 多 1×float32 |
| L3 | L2 + 语义描述 | ≤80 B | ≤64 字符 UTF-8 |
| L4 | L3 + 目标区域截图 | ~8 KB | 96×96 JPEG |
| L5 | L4 + 完整鸟瞰图 | ~300 KB | 1080p JPEG |

UGV → UAV 反馈消息（仅闭环模式）：

| 内容 | 大小 | 说明 |
|------|------|------|
| status (1B) + UGV pose (12B) + blocked_bearing (4B) + 填充 | 20 B | 固定长度 |

### 2.2 UAV 全局推理模块

每轮决策（~2s 一次）：

1. **See:** 拍摄向下鸟瞰图
2. **Point:** VLM 在图上标注 waypoint 像素 `(u, v)` + 深度标签 `{near,medium,far}`
3. **Scale:** 将深度标签映射为调整距离 `d_adj`
4. **Project:** 射线投影到地面平面，得到 3D 世界坐标 `p_world`
   - 公式: `p = c + λ·r`, 其中 `λ = -c_z / r_z`
   - 需要: UAV 相机内参 K、位姿 (R, c)
5. **Transmit:** 按当前通信级别编码消息，发送给 UGV

### 2.3 UGV 本地执行模块

收到 UAV 目标后：

1. **Verify:** 将 `p_world` 转为 UGV 坐标系（方位角 θ、距离 ρ），问 VLM：
   > *"Location at bearing θ°, distance ρm — reachable?"*
   
   返回: `reachable` / `occluded` / `out_of_view`

2. **Navigate (if reachable):**
   - 地面 SPF 导航到 `p_world`，步进控制
   - 检查是否到达 goal 3m 内 → 成功

3. **Feedback (if occluded/out_of_view + 闭环模式):**
   - 发送反馈消息给 UAV
   - UAV 收到后立即重新推理（2s 内），带上失败上下文

4. **Stall detection:**
   - 如果 10s 内没有明显位移 → 触发 replan

### 2.4 协同主循环

```
每 episode:
  while t < 180s:
    if 该 UAV 决策了:
      UAV 拍鸟瞰 → VLM 推理 → 投影 → 发消息给 UGV

    if UGV 有消息:
      UGV 验证 → 导航 or 反馈

    if UGV 反馈了 and 闭环模式:
      UAV 立刻 replan

    if 任一 agent 在 goal 3m 内:
      episode 成功

    t += Δt
```

### 2.5 运行配置矩阵

**实验 2.5a: 主结果 (L2, 闭环)**

```bash
# 默认配置：L2 级别 + 闭环反馈
for map in Town01 Town02 Town03 Town04 Town05 Town10HD:
    python runner_collab.py --map $map --level L2 --feedback
```

对比：
- UAV-only（Phase 1.1）
- UGV-only（Phase 1.2）
- 语义提示消融：不加坐标，只发文本描述（模拟 CARLA-Air 的 C1 模式）
- 无验证消融：UGV 收到坐标直接去，不验证可达性

---

## Phase 3 — 通信频谱消融

**目标:** 确定"多少通信就够了"——填 Table 2。

### 3.1 通信级别扫描

对 L0 ~ L5 每个级别：

1. **配置:** 固定用同一批 episodes
2. **运行:** 每 level × 3 seeds × 50 episodes
3. **记录:**
   - 导航性能: SR, SPL, NE
   - 通信开销: 每轮字节数 × 总轮数 = 每 episode 总流量

### 3.2 预期结果

- L0 → L1: 最大跳跃（从 0 到坐标，几何信号决定性作用）
- L1 → L2: 小幅提升（置信度帮助 UGV 判断）
- L2 → L3: 微弱提升（语义标注边缘情况）
- L3 → L4/L5: 平台期或略微下降（额外图像不提供有效增量信息）

### 3.3 图表

绘制 SR vs Bytes/episode 的 Pareto 曲线，标注 L2 为 sweet spot。

---

## Phase 4 — 反馈消融

**目标:** 量化 UGV 验证 + 闭环 replan 的价值——填 Table 3。

### 4.1 4 种配置（固定 L2 级别）

| # | 配置 | 验证 | 反馈 | 说明 |
|---|------|------|------|------|
| A | 单程无验证 | ❌ | ❌ | UAV 发坐标，UGV 直接去 |
| B | 单程有验证 | ✅ | ❌ | UGV 验证但失败了就停 |
| C | 闭环无验证 | ❌ | ✅ | 不验证但阻塞了 replan |
| D | 闭环有验证 | ✅ | ✅ | **默认配置** |

### 4.2 每配置运行

- 同一批 episodes
- 记录: SR, SPL, 每 episode 平均 replan 次数
- 额外分析: 哪类失败被验证捕获（栅栏、路沿、垂直遮蔽）

---

## Phase 5 — VLM Backbone 消融

**目标:** 验证方法不依赖特定 VLM——填 Table 4。

### 5.1 替换 VLM（固定 L2 闭环）

| VLM | API | 说明 |
|-----|-----|------|
| qwen-vl-max | DashScope | 当前使用 |
| qwen-vl-plus | DashScope | 更快的变体 |
| gpt-4o | OpenAI | 对比闭源最强 |
| gemini-2.5-pro | Google | 备选（需代理） |

### 5.2 每模型运行

- 同一批 episodes × 1 seed
- 记录: SR, SPL, 平均推理延迟

---

## Phase 6 — 鲁棒性测试

**目标:** 验证坐标级通信的抗干扰能力。

### 6.1 注入延迟

在 UAV→UGV 消息传输中人为加入延迟：`{0, 100, 500, 1000}ms`

预期：坐标消息对陈旧性容忍度高（目标点不动），不像速度耦合那样会发散。

### 6.2 注入丢包

以 `{0, 10, 30, 50}%` 概率丢弃 UAV→UGV 消息，无重传。

预期：16 bytes/条，50% 丢包带宽开销依然微不足道。

### 6.3 UAV 高度扫描

UAV 在 `{20, 30, 50, 80}m` 高度飞行。

预期：投影误差 ∝ 高度/焦距，80m 时 ~0.4m 依然在 ε=3m 内；但 VLM 的鸟瞰图细节随高度下降。

---

## Phase 7 — 失败归因

**目标:** 标注每个失败 episode 的根因——填 Table 5。

### 7.1 归因分类

| 类别 | 定义 | 示例 |
|------|------|------|
| Global grounding | UAV 的 VLM 标错了鸟瞰位置 | 点到旁边建筑而非目标 |
| Projection | 像素对但投影偏了 | 坡道/桥梁等高程异常 |
| Verification | UGV 的 VLM 判断错 | 低树丛误判为不可达 |
| Local execution | UGV 局部导航失败 | 最后的拐角过不去 |
| Reference binding | 坐标对但目标歧义 | 两辆相同卡车，不知是哪辆 |

### 7.2 流程

1. 对每个失败 episode，回放 debug 帧（UAV+UGV 视图）
2. 人工判断第一个出错的阶段
3. 统计各类占比

---

## Phase 8 — 汇总输出

### 8.1 数据聚合

```python
# 自动读取所有 summary.json
python scripts/aggregate.py --runs-dir runs/paper/ --output tables/
```

### 8.2 输出表格

| 表格 | 对应 paper | 来源 |
|------|-----------|------|
| `table_main.tex` | Table 1 — 主结果 | Phase 2 |
| `table_comm.tex` | Table 2 — 通信频谱 | Phase 3 |
| `table_feedback.tex` | Table 3 — 反馈消融 | Phase 4 |
| `table_backbone.tex` | Table 4 — VLM 消融 | Phase 5 |
| `table_failures.tex` | Table 5 — 失败归因 | Phase 7 |

### 8.3 输出图表

| 图 | 内容 |
|----|------|
| Figure 1 | 系统概览（UAV 鸟瞰 + 投影 + UGV 导航） |
| Figure 2 | 通信频谱柱状图（SR vs bytes） |
| Figure 3 | 定性 episode 对比（成功 / 失败） |

---

## 快速检查清单

每周跑之前确认：

- [ ] CARLA-Air 启动正常
- [ ] VLM API quota 充足
- [ ] 磁盘剩余 > 5GB（debug 帧积累）
- [ ] `--debug` 模式跑通一个 episode，确认帧输出正确
- [ ] 代理已关（DashScope 直连，不需要代理）

---

## 时间估算

| Phase | 内容 | 预计 |
|-------|------|------|
| 0 | Episode 准备 | 1-2 天 |
| 1 | Single-Agent 基线 | 1 天（跑实验） |
| 2 | 协同系统运行 | 2-3 天（调试+运行） |
| 3 | 通信频谱 | 1 天 |
| 4 | 反馈消融 | 1 天 |
| 5 | VLM Backbone | 1 天 |
| 6 | 鲁棒性 | 1 天 |
| 7 | 失败归因 | 1-2 天 |
| 8 | 汇总输出 | 1 天 |
| **总计** | | **10-13 天** |
