import os
from time import time, sleep

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"
from legged_gym import LEGGED_GYM_ROOT_DIR
import code

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.terrain import parkour_step_terrain
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import cv2
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from legged_gym.utils import webviewer
from rl_lib.rl_algo import AlgoRunner
from rl_lib.agent import ActorType
from get_algo_wo_isaac_gym import (
    RobotRLController,
    ENV_DICT,
    DOF_MAP,
    TORQUE_LIMITS,
    DOF_SIGNS,
    SNAP_PATH,
    USE_CAMERA,
    DO_ENCODING_TRICKS,
)

SELECTED_ENV = False
SELECTED_ACTOR_TYPE = ActorType.SINGLE_POLICY


def stepping_stones_terrain_mz(
    terrain, stone_size, stone_distance, max_height, platform_size=1.0, depth=-1
):
    """
    Generate a stepping stones terrain

    Parameters:
        terrain (terrain): the terrain
        stone_size (float): horizontal size of the stepping stones [meters]
        stone_distance (float): distance between stones (i.e size of the holes) [meters]
        max_height (float): maximum height of the stones (positive and negative) [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        depth (float): depth of the holes (default=-10.) [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """

    def get_rand_dis_int(scale):
        return np.random.randint(
            int(-scale / terrain.horizontal_scale + 1),
            int(scale / terrain.horizontal_scale),
        )

    # switch parameters to discrete units
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(-max_height - 1, max_height, step=1)

    start_x = 0
    start_y = 0
    terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)
    if terrain.length >= terrain.width:
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size)
            start_x = np.random.randint(0, stone_size)
            # fill first hole
            stop_x = max(0, start_x - stone_distance - get_rand_dis_int(0.2))
            terrain.height_field_raw[0:stop_x, start_y:stop_y] = np.random.choice(
                height_range
            )
            # fill row
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = (
                    np.random.choice(height_range)
                )
                start_x += stone_size + stone_distance + get_rand_dis_int(0.2)
            start_y += stone_size + stone_distance + get_rand_dis_int(0.2)
    elif terrain.width > terrain.length:
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            # fill first hole
            stop_y = max(0, start_y - stone_distance)
            terrain.height_field_raw[start_x:stop_x, 0:stop_y] = np.random.choice(
                height_range
            )
            # fill column
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size)
                terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = (
                    np.random.choice(height_range)
                )
                start_y += stone_size + stone_distance
            start_x += stone_size + stone_distance

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    terrain.idx = 0

    # 1st dimension: x, 2nd dimension: y
    goals = np.zeros((8, 2))
    mid_y = terrain.length // 2  # length is actually y width

    goals[:, 0] = np.linspace(terrain.width / 8, x1 + platform_size / 2, 8)
    goals[:, 1] = mid_y

    terrain.goals = goals * terrain.horizontal_scale

    return terrain


# KWARGS = {
#     "type": stepping_stones_terrain_mz,
#     "stone_size": 0.1,
#     "stone_distance": 0.01,
#     "max_height": 0.1,
#     "depth": -0.1,
# }

KWARGS = {
    "type": parkour_step_terrain,
    "platform_len": 2.5,
    "platform_height": 0.0,
    "num_stones": 6,
    "x_range": [0.2, 0.4],
    "y_range": [-0.15, 0.15],
    "half_valid_width": [0.45, 0.5],
    "step_height": 0.2,
    "pad_width": 0.1,
    "pad_height": 0.5
}


@torch.no_grad
def play(args):
    # args.headless = True
    # exptid = args.exptid
    log_pth = None  # "/docker_mount/logs/{}/".format(args.proj_name) + args.exptid
    # print(f"log_pth: {log_pth}")

    args.num_envs = 8

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # turn off delay to compare observations
    env_cfg.domain_rand.action_delay = False

    if SELECTED_ENV:
        env_cfg.terrain.selected = (
            True  # select a unique terrain type and pass all arguments
        )
        env_cfg.terrain.terrain_kwargs = (
            KWARGS  # Dict of arguments for selected terrain
        )
        env_cfg.env.num_envs = 1
        env_cfg.terrain.num_rows = 1
        env_cfg.terrain.num_cols = 1

    env_cfg.terrain.terrain_dict = {
        "smooth slope": 0.0,
        "rough slope up": 0.0,
        "rough slope down": 0.0,
        "rough stairs up": 0.0,
        "rough stairs down": 0.0,
        "discrete": 0.0,
        "stepping stones": 0.0,
        "gaps": 0.0,
        "smooth flat": 0,
        "pit": 0.0,
        "wall": 0.0,
        "platform": 0.0,
        "large stairs up": 0.0,
        "large stairs down": 0.0,
        # "parkour": 0.2,
        "parkour_hurdle": 0.2,
        "parkour_flat": 0.2,
        "parkour_step": 0.2,
        "parkour_step": 0.2,
        "parkour_gap": 0.2,
        "demo": 0.0,
    }

    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 4
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.simplify_grid = False
    env_cfg.terrain.max_difficulty = False
    env_cfg.terrain.max_init_terrain_level = -2

    env_cfg.sim.physx.max_gpu_contact_pairs = 2**23
    env_cfg.sim.physx.default_buffer_size_multiplier = 5

    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    algo_runner = AlgoRunner(
        log_dir=log_pth,
        env=env,
        args=args,
        env_name=args.task,
        device=args.device,
        actor_type=SELECTED_ACTOR_TYPE,
        enable_vids=False,
    )

    obs = env.get_observations()
    priv_obs = env.get_privileged_observations()

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    infos = {}
    depth_buf = env.depth_buffer.clone().to(args.device)[env.lookat_id, -1]
    # infos["depth"] = env.depth_buffer.clone().to(args.device)[:, -1] if ppo_runner.if_depth else None
    rl_controller = RobotRLController("cuda:0")

    scan_output_dim = train_cfg.policy.scan_encoder_dims[-1]

    cur_time = 0.0

    prior_action = None

    vx = 0.6
    vy = 0.0
    vtheta = 0.0
    rl_controller.commands[0] = vx
    rl_controller.commands[1] = vy
    rl_controller.commands[2] = vtheta

    show_reset = False
    show_reset_ctr = 0
    num_to_show_post_reset = 10
    n_yaw_est = 2

    inf_time_buffer_len = 1000
    inf_time_ptr = 0
    inf_time_buffer = [None] * inf_time_buffer_len

    for i in range(10 * int(env.max_episode_length)):
        if USE_CAMERA:
            obs_ours = torch.cat(
                (
                    obs[:, : env.cfg.env.n_proprio],
                    obs[
                        :,
                        env.cfg.env.n_proprio : env.cfg.env.n_proprio
                        + env.cfg.env.n_scan,
                    ],
                    obs[:, -env.cfg.env.history_len * env.cfg.env.n_proprio :],
                ),
                dim=1,
            ).clone()
        else:
            obs_ours = torch.cat(
                (
                    obs[:, : env.cfg.env.n_proprio],
                    obs[:, -env.cfg.env.history_len * env.cfg.env.n_proprio :],
                ),
                dim=1,
            ).clone()

        start_cmd_idx = env.cmd_start_idx
        obs_ours[:, start_cmd_idx] = vx
        obs_ours[:, start_cmd_idx + 1] = vy
        obs_ours[:, start_cmd_idx + 2] = vtheta

        for j in range(env.cfg.env.history_len):
            if i == 0:
                break

            target_vx = vx
            target_vy = vy
            target_vtheta = vtheta

            start_obs_idx = env_cfg.env.n_proprio
            if USE_CAMERA:
                start_obs_idx += env.cfg.env.n_scan

            start_obs_idx += env_cfg.env.n_proprio * j
            obs_ours[:, start_obs_idx + start_cmd_idx] = target_vx
            obs_ours[:, start_obs_idx + start_cmd_idx + 1] = target_vy
            obs_ours[:, start_obs_idx + start_cmd_idx + 2] = target_vtheta

        ang_vel_noisy = (
            obs[env.lookat_id, env.ang_vel_start_idx : env.ang_vel_end_idx]
            .detach()
            .cpu()
            .numpy()
            / env.obs_scales.ang_vel
        )
        rpy_noisy = (
            obs[env.lookat_id, env.rotation_start_idx : env.rotation_end_idx]
            .detach()
            .cpu()
            .numpy()
        )
        pos_noisy = (
            obs[env.lookat_id, env.dof_pos_start_idx : env.dof_pos_end_idx]
            .detach()
            .cpu()
            .numpy()
            / env.obs_scales.dof_pos
            + env.default_dof_pos_all[env.lookat_id, :].detach().cpu().numpy()
        )
        vel_noisy = (
            obs[env.lookat_id, env.dof_vel_start_idx : env.dof_vel_end_idx]
            .detach()
            .cpu()
            .numpy()
            / env.obs_scales.dof_vel
        )
        contact_filt_noisy = (
            obs[env.lookat_id, env.contact_filt_start_idx : env.contact_filt_end_idx]
            .detach()
            .cpu()
            .numpy()
        )
        foot_force = (
            torch.norm(env.contact_forces[env.lookat_id, env.feet_indices], dim=-1)
            .detach()
            .cpu()
            .numpy()
        )
        tau = env.torques[env.lookat_id, :].detach().cpu().numpy()
        contact_thresh = 2.0

        start_time_inf = time()

        obs_ours_2, depth_latent_ours = rl_controller.arrs_to_obs(
            ang_vel_noisy,
            rpy_noisy,
            pos_noisy,
            vel_noisy,
            tau,
            foot_force,
            depth_buf,
            contact_thresh,
        )

        actions = torch.zeros(
            (obs.shape[0], 12),
            device="cuda:0",
            dtype=torch.float32,
            requires_grad=False,
        )

        with torch.inference_mode():
            if USE_CAMERA:
                our_actions = rl_controller.obs_latent_to_act(obs_ours_2, None)
            else:
                our_actions, actions_log_prob, mus, sigmas = algo_runner.ac_agent.act(
                    obs_ours
                )
                rl_controller.action_buf[-1, :] = mus[env.lookat_id, :]
            actions[env.lookat_id, :] = our_actions.squeeze()
        detached_actions = actions.detach()

        obs, priv_obs, rews, dones, infos = env.step(detached_actions)

        if infos["depth"] is not None:
            # print(infos["depth"].shape)
            depth_buf = infos["depth"][env.lookat_id]

        print(
            "time:",
            env.episode_length_buf[env.lookat_id].item() / 50,
            "cmd vx",
            obs_ours[env.lookat_id, start_cmd_idx].item(),
            "actual vx",
            env.base_lin_vel[env.lookat_id, 0].item(),
            "actual x",
            env.root_states[env.lookat_id, 0].item(),
        )

        cur_time += env.dt

        if dones[env.lookat_id]:
            rl_controller.action_buf[:, :] = 0.0
            rl_controller.init_obs_history = True
            rl_controller.obs_history_buf[:, :, :] = 0.0
            rl_controller.last_contacts = np.array([False] * 4, dtype=bool)
            init_obs = True


if __name__ == "__main__":
    args = get_args()
    play(args)
