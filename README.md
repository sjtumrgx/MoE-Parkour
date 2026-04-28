# MoE-Parkour (Go2 Parkour with MoE Policy)

This repository contains a Go2 parkour training and deployment stack built on top of **Isaac Gym + legged_gym**, with a **Mixture-of-Experts (MoE)** policy pipeline and sim-to-real deployment scripts.

## Documentation

- **English:** [README_en.md](README_en.md)
- **中文文档：** [README_zh.md](README_zh.md)

## Quick Links

- Training environment Docker: [`docker/Dockerfile`](docker/Dockerfile)
- RL training/inference code: [`rl_lib/rl_lib`](rl_lib/rl_lib)
- Legged Gym fork: [`extreme-parkour`](extreme-parkour)
- Real robot scripts: [`robot_firmware`](robot_firmware)
- Safety-critical controller config: [`robot_firmware/get_algo_wo_isaac_gym.py`](robot_firmware/get_algo_wo_isaac_gym.py)

> ⚠️ Real robot operation is high risk. Validate in simulation first and follow strict safety procedures.
