import numpy as np
import os
import sys
from datetime import datetime

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, class_to_dict
import wandb
import yaml

from rl_lib.rl_algo import AlgoRunner
from rl_lib.agent import ActorType

LOG_DIR = "/docker_mount/logs"

# SELECTED_ACTOR_TYPE = ActorType.SINGLE_POLICY
SELECTED_ACTOR_TYPE = ActorType.MIX_OF_EXPERTS
# SELECTED_ACTOR_TYPE = ActorType.POLICY_PER_SKILL


def train(args):
    print("starting training", flush=True)
    print(args, flush=True)
    args.headless = True
    log_pth = "{}/{}/".format(LOG_DIR, args.proj_name) + args.exptid
    try:
        os.makedirs(log_pth)
    except:
        pass

    mode = "online"

    if args.no_wandb:
        mode = "disabled"
    wandb.init(
        project=args.proj_name,
        name=args.exptid,
        entity=args.wandb_entity,
        group=args.exptid[:3],
        mode=mode,
        dir=LOG_DIR,
    )
    # wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot_config.py", policy="now")
    # wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot.py", policy="now")

    print("building env", flush=True)
    print("               building env", flush=True)

    # args.rows = 10
    # args.cols = 6
    print("args", flush=True)
    print(args)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    if args.use_camera:
        env_cfg.domain_rand.action_curr_step = [1, 1]

    print(f"terrain: {env_cfg.terrain.terrain_dict}")

    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    print("built env", flush=True)
    print(f"terrain: {env_cfg.terrain.terrain_dict}")

    env_cfg_dict = class_to_dict(env_cfg)
    train_cfg_dict = class_to_dict(train_cfg)

    with open(os.path.join(log_pth, "env_cfg.yaml"), "w") as file_handle:
        yaml_str = yaml.dump(env_cfg_dict)
        file_handle.write(yaml_str)

    with open(os.path.join(log_pth, "train_cfg.yaml"), "w") as file_handle:
        yaml_str = yaml.dump(train_cfg_dict)
        file_handle.write(yaml_str)

    algo_runner = AlgoRunner(
        log_dir=log_pth,
        env=env,
        args=args,
        env_name=args.task,
        device=args.device,
        actor_type=SELECTED_ACTOR_TYPE,
    )

    if args.resume:
        resume_id = args.resumeid
        resume_log_pth = "{}/{}/".format(LOG_DIR, args.proj_name) + resume_id
        all_snapshots = os.listdir(resume_log_pth)
        all_snapshots = [snap for snap in all_snapshots if "snapshot" in snap]
        max_snap_idx = -1
        max_iter_num = -1
        for i in range(len(all_snapshots)):
            snap = all_snapshots[i]
            snap = snap.replace(".pt", "")
            _, iter_num = snap.split("_")
            iter_num = int(iter_num)
            if iter_num > max_iter_num:
                max_iter_num = iter_num
                max_snap_idx = i
        snap_path = os.path.join(resume_log_pth, all_snapshots[max_snap_idx])
        print(f"policy found, loading: {snap_path}", flush=True)
        algo_runner.load_snapshot(snap_path)
        algo_runner.cur_learning_iteration = 0

    algo_runner.train()


if __name__ == "__main__":
    # Log configs immediately
    print("getting args", flush=True)
    args = get_args()
    print("running train", flush=True)
    train(args)
