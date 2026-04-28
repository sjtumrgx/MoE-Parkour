# MoE-Parkour：Go2 跑酷训练与部署文档

English documentation: [README.md](README.md)

### 1. 项目概述

MoE-Parkour 是一个面向 **Unitree Go2** 的四足跑酷项目，核心技术栈包括：

- **Isaac Gym** 仿真
- 基于 **legged_gym / extreme-parkour** 的改造环境
- 自定义强化学习库 `rl_lib`（支持 **Mixture-of-Experts, MoE** 策略）
- 面向真实 Go2 机器人的 ROS2 + 深度相机部署脚本

仓库同时覆盖仿真训练与真机运行代码；真机部分属于高风险操作，请务必谨慎。

---

### 2. 主要特性

- **两阶段训练流程**
  1. 不使用深度相机的基础策略训练
  2. 使用深度观测的蒸馏/微调训练（`--use_camera`）
- **MoE 策略网络支持**（在自定义 Actor 中实现）
- **Go2 专用配置**（`task=go2`）
- **仿真回放能力**（加载 checkpoint）
- **真机控制脚本**（低层控制发布 + 深度图处理）

---

### 3. 仓库结构

```text
.
├── README.md                   # 英文详细文档
├── README_zh.md                # 中文文档（本文件）
├── docker/
│   ├── Dockerfile              # Isaac Gym + 训练环境镜像
│   └── python/                 # Isaac Gym python 包骨架
├── extreme-parkour/            # 上游 fork（含 legged_gym + rsl_rl）
├── rl_lib/
│   ├── rl_lib/train.py         # 训练入口
│   ├── rl_lib/play.py          # 仿真回放入口
│   ├── rl_lib/rl_algo.py       # PPO / 深度 / MoE 核心逻辑
│   └── tests/                  # 单元测试
├── robot_firmware/
│   ├── get_algo_wo_isaac_gym.py # 策略加载与控制器逻辑
│   ├── make_obs.py              # 真机 ROS2 低层控制主循环
│   ├── see_robot_cam.py         # 相机流连通性检查
│   ├── setup.sh                 # 真机网卡 ROS2 环境变量
│   └── setup_sim.sh             # 回环/仿真 ROS2 环境变量
└── go2_cam_mount.stl            # 相机安装结构件
```

---

### 4. 环境要求

### 4.1 主机侧（训练/仿真）

- Linux + NVIDIA GPU
- Docker + NVIDIA Container Toolkit
- 若需图形化回放，需支持 X11 转发

### 4.2 训练容器内依赖

由 `docker/Dockerfile` 构建，主要包含：

- CUDA 版本 PyTorch
- Isaac Gym Python 包（`docker/python`）
- `extreme-parkour/rsl_rl` 与 `extreme-parkour/legged_gym`
- `wandb`、`opencv-python`、`pyfqmr` 等 Python 依赖

### 4.3 真机运行依赖

- Unitree Go2
- ROS2 Humble
- Intel RealSense 深度流（`/camera/depth/image_rect_raw`）
- 与 `robot_firmware/setup.sh` 一致的网络接口配置

---

### 5. 构建训练环境

### 5.1 Docker Compose（推荐）

仓库已提供 `docker-compose.yml`，可直接运行 Isaac Gym 训练/仿真容器：

```bash
cd MoE-Parkour
mkdir -p docker_mount/logs
docker compose build isaacgym
docker compose run --rm isaacgym
```

容器内建议设置：

```bash
export PYTHONPATH=$PYTHONPATH:/home/gymuser/rl_lib
export PYTHONPATH=$PYTHONPATH:/home/gymuser/robot_firmware
```

### 5.2 直接 docker run（等价旧流程）

```bash
export REPO_DIR=$HOME
cd "$REPO_DIR"
# 先克隆仓库，再进入目录
cd MoE-Parkour

docker build -f docker/Dockerfile -t isaacgym .
mkdir -p "$REPO_DIR/MoE-Parkour/docker_mount/logs"

# 启动训练容器
docker run -it --rm \
  -v "$REPO_DIR/MoE-Parkour/docker_mount:/docker_mount" \
  -v "$REPO_DIR/MoE-Parkour/rl_lib:/home/gymuser/rl_lib" \
  --ipc=host --network=host --gpus=all \
  isaacgym /bin/bash
```

容器内设置：

```bash
export PYTHONPATH=$PYTHONPATH:/home/gymuser/rl_lib
```

---

### 6. 两阶段训练流程

### 6.1 第一阶段：基础策略（无相机）

```bash
export WANDB_API_KEY=your_key  # 使用 wandb 时可设置
python rl_lib/rl_lib/train.py \
  --task go2 \
  --exptid 555-55-moe-top4-16 \
  --device cuda:0 \
  --num_envs 6000 \
  --max_iterations 15000 \
  --no_wandb \
  2>&1 | tee -a /docker_mount/logs/555-55-moe-top4-16.txt
```

### 6.2 第二阶段：加入深度观测蒸馏/微调

```bash
python rl_lib/rl_lib/train.py \
  --task go2 \
  --exptid 555-55-moe-top4-16-cam \
  --device cuda:0 \
  --resume --resumeid 555-55-moe-top4-16 \
  --delay --use_camera --no_wandb \
  2>&1 | tee -a /docker_mount/logs/555-55-moe-top4-16-cam.txt
```

### 6.3 训练产物

默认保存路径：

```text
/docker_mount/logs/<proj_name>/<exptid>/
├── snapshot_<iter>.pt
├── env_cfg.yaml
└── train_cfg.yaml
```

默认 `proj_name` 为 `parkour_new`。

### 6.4 常用命令行参数

| 参数 | 含义 |
| --- | --- |
| `--task` | `legged_gym` 中注册的任务名（本项目通常用 `go2`）。 |
| `--exptid` | 实验/运行 ID（也用于日志目录命名）。 |
| `--resume --resumeid <id>` | 从已有 run id 继续训练。 |
| `--use_camera` | 启用基于深度相机的训练/推理路径。 |
| `--delay` | 启用跑酷训练中使用的动作延迟设置。 |
| `--num_envs` | 并行仿真环境数量。 |
| `--max_iterations` | 训练最大迭代次数。 |
| `--device` | 计算/仿真设备（如 `cuda:0`）。 |
| `--no_wandb` | 禁用 Weights & Biases 日志。 |

---

### 7. 仿真回放

`rl_lib/rl_lib/play.py` 会读取：

- `robot_firmware/get_algo_wo_isaac_gym.py`
- 关键常量：`SNAP_PATH`、`ACTOR_TYPE`、`ACTOR_KWARGS`、`USE_CAMERA`

请先将 `SNAP_PATH` 改为你的 checkpoint 路径。

示例：

```bash
docker compose run --rm isaacgym

# 或使用旧版 docker run：
export REPO_DIR=$HOME
docker run -it --rm \
  -v "$REPO_DIR/MoE-Parkour/docker_mount:/docker_mount" \
  -v "$REPO_DIR/MoE-Parkour/rl_lib:/home/gymuser/rl_lib" \
  -v "$REPO_DIR/MoE-Parkour/robot_firmware:/home/gymuser/robot_firmware" \
  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY \
  --ipc=host --network=host --gpus=all \
  isaacgym /bin/bash

export PYTHONPATH=$PYTHONPATH:/home/gymuser/rl_lib
export PYTHONPATH=$PYTHONPATH:/home/gymuser/robot_firmware
python3 rl_lib/rl_lib/play.py --task go2 --delay --use_camera --no_wandb
```

---

### 8. 真机部署（高风险）

> ⚠️ 机器人损坏、人身风险由使用者自行承担。该代码可能产生较激进动作。

建议最少安全措施：

- 顶吊/支撑保护
- 紧急停机方案
- 人员远离机器人工作区
- 先在仿真充分验证策略

### 8.1 真机准备

- 主机静态 IP：`192.168.123.222`
- 连接机器人：`unitree@192.168.123.18`（密码见旧 README）
- 机器人侧启动 RealSense：

```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_depth:=true enable_color:=true \
  depth_module.profile:=640x480x15 \
  rgb_camera.profile:=640x480x30
```

### 8.2 构建并运行真机容器

```bash
export REPO_DIR=$HOME
cd "$REPO_DIR/MoE-Parkour/robot_firmware"
docker build -f Dockerfile -t go2-parkour .

docker run --rm -it --network host \
  -v "$REPO_DIR/MoE-Parkour/docker_mount:/docker_mount" \
  -v "$REPO_DIR/MoE-Parkour/rl_lib:/home/gymuser/rl_lib" \
  -v "$REPO_DIR/MoE-Parkour/robot_firmware:/home/gymuser/robot_firmware" \
  --gpus device=0 \
  go2-parkour
```

容器内：

```bash
export PYTHONPATH=$PYTHONPATH:/home/developer/rl_lib
export PYTHONPATH=$PYTHONPATH:/tmp/unitree_sdk2/python
pip install "numpy<2.0.0"
cd /home/developer/robot_firmware
source setup.sh
python3 see_robot_cam.py
```

### 8.3 低层控制执行

先关闭 sports mode，再启动控制循环：

```bash
cd ~/unitree_sdk2/build
./bin/go2_stand_example enp13s0f1
# sports mode 关闭后 Ctrl+C

cd ~/robot_firmware
python3 make_obs.py
```

---

### 9. 关键运行开关

在 `robot_firmware/get_algo_wo_isaac_gym.py` 中：

- `SNAP_PATH`：checkpoint 路径
- `ACTOR_TYPE`：策略类型（如 `MIX_OF_EXPERTS`）
- `ACTOR_KWARGS`：MoE 结构参数（expert 数、top-k 等）
- `USE_CAMERA`：是否依赖深度相机
- `DO_ENCODING_TRICKS`：是否启用编码技巧/状态估计路径

这些配置若与训练时不一致，推理可能失败。

---

## 10. 本地测试

仓库根目录执行：

```bash
PYTHONPATH=rl_lib python -m unittest rl_lib/tests/test_replay_buffer.py
```

---

## 11. 常见问题排查

- **找不到 checkpoint**
  - 检查 `SNAP_PATH` 与 `docker_mount` 挂载路径是否正确。
- **模型加载维度不匹配**
  - 检查 `ACTOR_TYPE` 与 MoE 参数是否与训练时一致。
- **真机模式没有深度图**
  - 确认 ROS2 话题 `/camera/depth/image_rect_raw` 正常发布。
- **`robot_firmware` 构建时报缺文件**
  - 当前 Dockerfile 引用了 `realsense.py`、`visualize_realsense.py`、`go2_fastlio.yaml`，需补齐文件或按部署需要移除对应 `COPY` 行。

---

## 12. 致谢

本项目基于并改造了以下开源工作：

- [Extreme Parkour](https://github.com/chengxuxin/extreme-parkour)
- [legged_gym](https://github.com/leggedrobotics/legged_gym)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)

---

## 13. 许可证

根目录 `LICENSE.txt` 为 GPLv3 文本；子目录中上游代码分别保留其原始许可证。
