# MoE-Parkour: Go2 Parkour Training & Deployment

中文文档请见: [README_zh.md](README_zh.md)

### 1. Overview

MoE-Parkour is a quadruped locomotion project for **Unitree Go2**, focused on parkour-like terrain traversal using:

- **Isaac Gym** simulation
- a forked **legged_gym / extreme-parkour** stack
- a custom RL library (`rl_lib`) with **Mixture-of-Experts (MoE)** actor support
- deployment scripts for **real Go2 robot** control with ROS2 + depth camera

This repository includes both simulation training and real-robot runtime code, but the real-robot branch is safety-critical and should be used with extreme caution.

---

### 2. Key Features

- **Two-stage training pipeline**
  1. Base policy training without depth camera
  2. Distillation/fine-tuning with depth observations (`--use_camera`)
- **MoE policy support** in the custom actor implementation
- **Go2-specific legged_gym configuration** (`task=go2`)
- **Simulation playback** using saved checkpoints
- **Real-robot runtime scripts** for low-level command publishing and depth processing

---

### 3. Repository Structure

```text
.
├── README.md                   # English documentation (this file)
├── README_zh.md                # Chinese documentation
├── docker/
│   ├── Dockerfile              # Isaac Gym + training environment image
│   └── python/                 # Isaac Gym python package scaffold
├── extreme-parkour/            # Forked upstream codebase (legged_gym + rsl_rl)
├── rl_lib/
│   ├── rl_lib/train.py         # Training entry
│   ├── rl_lib/play.py          # Simulation playback entry
│   ├── rl_lib/rl_algo.py       # Core PPO/depth/MoE runner
│   └── tests/                  # Unit tests
├── robot_firmware/
│   ├── get_algo_wo_isaac_gym.py # Policy loading + controller logic
│   ├── make_obs.py              # ROS2 low-level control loop (real robot)
│   ├── see_robot_cam.py         # Camera stream sanity check
│   ├── setup.sh                 # ROS2 network env setup (robot NIC)
│   └── setup_sim.sh             # ROS2 setup for loopback/sim
└── go2_cam_mount.stl            # Camera mount model
```

---

### 4. Requirements

### 4.1 Host machine (training/sim)

- Linux host with NVIDIA GPU
- Docker + NVIDIA Container Toolkit
- X11 forwarding if you want graphical playback

### 4.2 Training runtime (inside training image)

Built by `docker/Dockerfile`, including:

- PyTorch (CUDA)
- Isaac Gym python package (`docker/python`)
- `extreme-parkour/rsl_rl` and `extreme-parkour/legged_gym`
- Python deps such as `wandb`, `opencv-python`, `pyfqmr`

### 4.3 Real robot runtime

- Unitree Go2
- ROS2 Humble environment
- Intel RealSense depth stream (`/camera/depth/image_rect_raw`)
- Network configuration matching `robot_firmware/setup.sh`

---

### 5. Build Training Environment

### 5.1 Docker Compose (recommended)

This repository includes `docker-compose.yml` for Isaac Gym training/simulation:

```bash
cd MoE-Parkour
mkdir -p docker_mount/logs
docker compose build isaacgym
docker compose run --rm isaacgym
```

Inside container:

```bash
export PYTHONPATH=$PYTHONPATH:/home/gymuser/rl_lib
export PYTHONPATH=$PYTHONPATH:/home/gymuser/robot_firmware
```

### 5.2 Direct Docker run (legacy equivalent)

```bash
export REPO_DIR=$HOME
cd "$REPO_DIR"
# Clone your repository first, then:
cd MoE-Parkour

docker build -f docker/Dockerfile -t isaacgym .
mkdir -p "$REPO_DIR/MoE-Parkour/docker_mount/logs"

# Launch training container
docker run -it --rm \
  -v "$REPO_DIR/MoE-Parkour/docker_mount:/docker_mount" \
  -v "$REPO_DIR/MoE-Parkour/rl_lib:/home/gymuser/rl_lib" \
  --ipc=host --network=host --gpus=all \
  isaacgym /bin/bash
```

Inside container:

```bash
export PYTHONPATH=$PYTHONPATH:/home/gymuser/rl_lib
```

---

### 6. Two-Stage Training

### 6.1 Stage-1: base policy (no camera)

```bash
export WANDB_API_KEY=your_key  # optional when using wandb
python rl_lib/rl_lib/train.py \
  --task go2 \
  --exptid 555-55-moe-top4-16 \
  --device cuda:0 \
  --num_envs 6000 \
  --max_iterations 15000 \
  --no_wandb \
  2>&1 | tee -a /docker_mount/logs/555-55-moe-top4-16.txt
```

### 6.2 Stage-2: depth-enabled training/distillation

```bash
python rl_lib/rl_lib/train.py \
  --task go2 \
  --exptid 555-55-moe-top4-16-cam \
  --device cuda:0 \
  --resume --resumeid 555-55-moe-top4-16 \
  --delay --use_camera --no_wandb \
  2>&1 | tee -a /docker_mount/logs/555-55-moe-top4-16-cam.txt
```

### 6.3 Output artifacts

Checkpoints and configs are saved under:

```text
/docker_mount/logs/<proj_name>/<exptid>/
├── snapshot_<iter>.pt
├── env_cfg.yaml
└── train_cfg.yaml
```

Default `proj_name` is `parkour_new`.

### 6.4 Common CLI arguments

| Argument | Meaning |
| --- | --- |
| `--task` | Task name registered in `legged_gym` (`go2` is used here). |
| `--exptid` | Experiment/run identifier (used in log folder names). |
| `--resume --resumeid <id>` | Resume from an existing run id. |
| `--use_camera` | Enable depth-based training/inference path. |
| `--delay` | Enable action delay settings used in parkour training. |
| `--num_envs` | Number of parallel simulation environments. |
| `--max_iterations` | Training iteration limit. |
| `--device` | Compute/sim device (e.g. `cuda:0`). |
| `--no_wandb` | Disable Weights & Biases logging. |

---

### 7. Simulation Playback

`rl_lib/rl_lib/play.py` loads policy/controller settings via:

- `robot_firmware/get_algo_wo_isaac_gym.py`
- especially `SNAP_PATH`, `ACTOR_TYPE`, `ACTOR_KWARGS`, `USE_CAMERA`

Before playback, edit `SNAP_PATH` to your trained checkpoint.

Example:

```bash
docker compose run --rm isaacgym

# or direct docker run:
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

### 8. Real Robot Deployment (High Risk)

> ⚠️ You are fully responsible for hardware risks. This code can generate aggressive motions.

Recommended safety minimum:

- overhead support/crane
- emergency stop strategy
- keep people away from robot workspace
- test policy in simulation first

### 8.1 Robot-side preparation

- Configure host static IP: `192.168.123.222`
- Connect to robot: `unitree@192.168.123.18` (password in legacy README)
- On robot, launch RealSense depth pipeline:

```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_depth:=true enable_color:=true \
  depth_module.profile:=640x480x15 \
  rgb_camera.profile:=640x480x30
```

### 8.2 Build and run robot container

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

Inside container:

```bash
export PYTHONPATH=$PYTHONPATH:/home/developer/rl_lib
export PYTHONPATH=$PYTHONPATH:/tmp/unitree_sdk2/python
pip install "numpy<2.0.0"
cd /home/developer/robot_firmware
source setup.sh
python3 see_robot_cam.py
```

### 8.3 Low-level control run

Disable sports mode first, then run controller:

```bash
cd ~/unitree_sdk2/build
./bin/go2_stand_example enp13s0f1
# Ctrl+C after sports mode is disabled

cd ~/robot_firmware
python3 make_obs.py
```

---

### 9. Important Runtime Switches

- `robot_firmware/get_algo_wo_isaac_gym.py`
  - `SNAP_PATH`: checkpoint to load
  - `ACTOR_TYPE`: policy family (`MIX_OF_EXPERTS`, etc.)
  - `ACTOR_KWARGS`: MoE topology (e.g., experts, top-k)
  - `USE_CAMERA`: whether depth stream is required
  - `DO_ENCODING_TRICKS`: state-estimation/encoding pathway toggle

Changing these values without matching training config can break inference.

---

## 10. Testing (Repository-local)

From repository root:

```bash
PYTHONPATH=rl_lib python -m unittest rl_lib/tests/test_replay_buffer.py
```

---

## 11. Troubleshooting

- **No checkpoint found at runtime**
  - Verify `SNAP_PATH` and mounted `docker_mount` path.
- **Policy shape mismatch / load failure**
  - Ensure actor type and MoE parameters match the training run.
- **No camera frames in real robot mode**
  - Confirm ROS2 topic `/camera/depth/image_rect_raw` is publishing.
- **Docker build under `robot_firmware` fails on missing files**
  - Current Dockerfile references `realsense.py`, `visualize_realsense.py`, and `go2_fastlio.yaml`. Add or remove those `COPY` lines based on your deployment setup.

---

## 12. Acknowledgements

This project is built upon and adapts:

- [Extreme Parkour](https://github.com/chengxuxin/extreme-parkour)
- [legged_gym](https://github.com/leggedrobotics/legged_gym)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)

---

## 13. License

This repository includes `LICENSE.txt` (GPLv3 text at root) and upstream components with their own licenses under subdirectories.
