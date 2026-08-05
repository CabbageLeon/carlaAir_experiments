# Collab VLN — 空地协同 VLN 实验

## 环境要求

- CARLA-Air 已安装（`./CarlaAir.sh` 可启动）
- conda 环境 `carlaAir` 已配置
- OpenCV 窗口可用（有显示器）

## 终端布局

```
┌──────────────┬──────────────────┬──────────────────┐
│  终端 1      │  终端 2          │  终端 5          │
│  CARLA-Air   │  无人机双视角     │  UAV 自动飞行     │
│  仿真引擎     │  show_drone_cam  │  auto_fly.py     │
├──────────────┼──────────────────┼──────────────────┤
│  终端 3      │  终端 4          │  终端 6          │
│  场景初始化   │  操控命令行       │  UGV 自动驾驶     │
│  + 俯视+追拍  │  control.py     │  auto_drive.py   │
└──────────────┴──────────────────┴──────────────────┘
```

---

## 启动流程

### 终端 1 — 启动仿真引擎

```bash
cd /home/zsn/VLN/carlaAir_experiments
conda activate carlaAir
./CarlaAir.sh
```

等待 CARLA-Air 窗口出现、地图加载完成。

### 终端 2 — 无人机双视角

```bash
cd /home/zsn/VLN/carlaAir_experiments
conda activate carlaAir
python show_drone_cam.py
```

显示前视 + 下视拼接画面。按键：
- **V** — 切换视角（前视 / 下视 / 拼接）
- **M** — 开关小地图
- **S** — 截图
- **Q / ESC** — 退出

### 终端 3 — 场景初始化 + 监控窗口

```bash
cd /home/zsn/VLN/carlaAir_experiments
conda activate carlaAir
python collab_vln/scripts/setup_scene.py
```

交互式选择 episode，自动：
- 起点生成 Mini Cooper
- 目标点生成 HGV 大卡车
- 无人机 60m 悬停在起点上方

弹出两个监控窗口：

| 窗口 | 内容 |
|------|------|
| `Overhead` | 俯视小地图（START=蓝圈，GOAL=红圈，黄线=路径） |
| `Chase [HGV@GOAL]` | 目标 HGV 卡车后方 12m 追拍 |

关闭窗口或 **Ctrl+C** 退出。

### 终端 4 — 操控

```bash
cd /home/zsn/VLN/carlaAir_experiments
conda activate carlaAir
python collab_vln/scripts/control.py
```

#### 无人机操控（本体坐标系）

| 命令 | 说明 | 示例 |
|------|------|------|
| `d fwd right down [yaw_rate] [dur]` | 速度飞行 | `d 5 0 0` = 前进 5m/s 持续 1s |
| | | `d 0 3 -2` = 右移 3m/s + 下降 2m/s |
| | | `d 0 0 0 30 2` = 原地右转 30°/s 持续 2s |
| `h` | 悬停 | |

- `fwd` — 机头前方（m/s）
- `right` — 机身右侧（m/s）
- `down` — 下方（m/s，NED down 为正）
- `yaw_rate` — 偏航角速度（°/s，正=右转）
- `dur` — 持续时间（秒，默认 1s）

飞行结束后自动悬停。

#### 车辆操控

| 命令 | 说明 | 示例 |
|------|------|------|
| `c throttle steer` | 驾驶 | `c 0.5 0.2` = 50%油门 右转 |
| `s` | 刹车 | |
| `C spawn_idx` | 瞬移到出生点 | `C 15` = 跳到 spawn #15 |

#### 状态查询

| 命令 | 说明 |
|------|------|
| `w` | 查看无人机、车、目标位置 |
| `q` | 退出 |

### 终端 5 — 自动飞行到目标

```bash
cd /home/zsn/VLN/carlaAir_experiments
conda activate carlaAir
python collab_vln/scripts/auto_fly.py
```

自动执行两阶段飞行：

| 阶段 | 动作 | 说明 |
|------|------|------|
| 1 | 原地旋转 | 计算目标方位角，偏航对准 goal |
| 2 | 前进 | 沿机头方向飞行到 goal 正上方 |

选项：
- `--altitude 80` — 指定目标高度（默认保持当前高度）
- `--loop` — 测试模式，来回飞行

飞行过程打印实时位置、距离、误差。

### 终端 6 — UGV 自动驾驶到目标

```bash
cd /home/zsn/VLN/carlaAir_experiments
conda activate carlaAir
python collab_vln/scripts/auto_drive.py
```

沿 CARLA 道路网络自动行驶到 goal HGV 卡车位置。

弹出两个窗口：

| 窗口 | 内容 |
|------|------|
| `Chase Cam` | 车后方 8m 追拍，显示实时距离和速度 |
| `Route Map` | 俯视路线图：绿线=规划路径，蓝点=车，红点=目标 |

特性：
- 闭环纯追踪控制（lookahead 8m）
- 自动减速入弯 + 接近目标时缓行
- 到达 3m 内自动刹车
- ESC 随时终止

---

## Episode 文件

Episode 模板在 `collab_vln/episodes/` 下：

```
collab_vln/episodes/
├── town10hd_templates.json   # 自动生成的 15 个 episode
└── ...
```

### 生成新 episode

```bash
cd /home/zsn/VLN/carlaAir_experiments
conda activate carlaAir
python collab_vln/scripts/generate_episodes.py --map Town10HD --num 15 --min-dist 80 --max-dist 300
```

### 查看 episode 地图

```bash
python collab_vln/scripts/plot_episodes.py \
    --input collab_vln/episodes/town10hd_templates.json \
    --output collab_vln/episodes/town10hd_map.png
```

---

## Episode JSON 格式

```json
{
  "id": "town10hd_001",
  "map": "Town10HD",
  "instruction": "导航指令（待填写）",
  "goal": {"x": -106.6, "y": -17.1, "z": 0.6},
  "ugv_spawn": {"index": 28, "x": -15.1, "y": 69.7, "z": 0.6, "yaw": 0.1},
  "uav_spawn": {"x": -60.8, "y": 26.3, "z": 50.0, "yaw": 0.0},
  "distance_m": 126.1,
  "time_budget_s": 180,
  "tags": []
}
```

---

## 手动测试流程

1. 启动 4 个终端（按上方流程）
2. 终端 3 选一个 episode 加载
3. 终端 2 观察无人机视角，终端 4 操控无人机飞到目标上方
4. 判断：
   - 起点到目标的路径是否合理（有路可走）
   - 地面视角是否能看到目标（是否被建筑遮挡）
   - 是否需要 UAV 鸟瞰才能找到路径（体现协作价值）
5. 为合理的 episode 写 `instruction`，如 "沿主路直行到路口右转，目标在右侧红色建筑前"
6. 不合理或无法到达的 episode 标记 `tags: ["rejected"]` 或删除

---

## 目录结构

```
collab_vln/
├── README.md                          # 本文件
├── episodes/                          # episode 数据
│   ├── town10hd_templates.json
│   ├── town10hd_map.png
│   └── ...
├── scripts/                           # 工具脚本
│   ├── generate_episodes.py           # 从 CARLA 地图生成 episode 模板
│   ├── plot_episodes.py               # 2D 可视化 episode 分布
│   ├── birdview_goals.py              # 俯拍 goal 截图
│   ├── setup_scene.py                 # 终端 3 — 场景初始化
│   ├── control.py                     # 终端 4 — 操控界面
│   └── preview_episode.py             # 单窗口交互预览
├── birdviews/                         # 俯拍截图输出
└── captures/                          # 终端截图输出
```

---

## 常见问题

### 无人机乱飞 / 漂移

终端 4 执行 `h` 悬停，或重新运行终端 3 初始化场景。

### AirSim 超时

终端 1（CARLA-Air）可能卡住了，重启 `./CarlaAir.sh`。

### 相机角度没变化

修改 `~/Documents/AirSim/settings.json`（不是项目里那个），重启 CARLA-Air。
