# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import code

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.helpers import get_args, update_cfg_from_args, class_to_dict
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import cv2
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep
from legged_gym.utils import webviewer

# from get_algo_wo_isaac_gym import ENV_DICT, DOF_MAP, TORQUE_LIMITS, DOF_SIGNS

USE_THEIR_POLICY = True

if not USE_THEIR_POLICY:
    from rl_lib.rl_algo import AlgoRunner

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint

def play(args):
    if args.web:
        web_viewer = webviewer.WebViewer()
    faulthandler.enable()
    exptid = args.exptid
    log_pth = "/docker_mount/logs/{}/".format(args.proj_name) + args.exptid
    print(f"log_pth: {log_pth}")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)


    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 16 if not args.save else 64
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {"smooth slope": 0., 
                                    "rough slope up": 0.0,
                                    "rough slope down": 0.0,
                                    "rough stairs up": 0., 
                                    "rough stairs down": 0., 
                                    "discrete": 0., 
                                    "stepping stones": 0.0,
                                    "gaps": 0., 
                                    "smooth flat": 0,
                                    "pit": 0.0,
                                    "wall": 0.0,
                                    "platform": 0.,
                                    "large stairs up": 0.,
                                    "large stairs down": 0.,
                                    "parkour": 0.2,
                                    "parkour_hurdle": 0.2,
                                    "parkour_flat": 0.2,
                                    "parkour_step": 0.2,
                                    "parkour_gap": 0.2, 
                                    "demo": 0.0}
    # env_cfg.terrain.terrain_dict = {"smooth slope": 0., 
    #                                 "rough slope up": 0.0,
    #                                 "rough slope down": 0.0,
    #                                 "rough stairs up": 0., 
    #                                 "rough stairs down": 0., 
    #                                 "discrete": 0., 
    #                                 "stepping stones": 0.0,
    #                                 "gaps": 0., 
    #                                 "smooth flat": 0,
    #                                 "pit": 0.0,
    #                                 "wall": 0.0,
    #                                 "platform": 0.,
    #                                 "large stairs up": 0.,
    #                                 "large stairs down": 0.,
    #                                 "parkour": 0.0,
    #                                 "parkour_hurdle": 0.0,
    #                                 "parkour_flat": 1.0,
    #                                 "parkour_step": 0.0,
    #                                 "parkour_gap": 0.0, 
    #                                 "demo": 0.0}
    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True
    
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False

    depth_latent_buffer = []
    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    print(f"use camera is set to : {env.cfg.depth.use_camera}")

    print(f"vx: {env.commands[env.lookat_id, 0].item()}")

    if args.web:
        web_viewer.setup(env)

    # load policy
    train_cfg.runner.resume = True

    if USE_THEIR_POLICY:
        ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)
        
        print(f"env cfg: \n{class_to_dict(env_cfg)}")
        print(f"train_cfg: \n{class_to_dict(train_cfg)}")
        if args.use_jit:
            path = os.path.join(log_pth, "traced")
            model, checkpoint = get_load_path(root=path, checkpoint=args.checkpoint)
            path = os.path.join(path, model)
            print("Loading jit for policy: ", path)
            policy_jit = torch.jit.load(path, map_location=env.device)
        else:
            policy = ppo_runner.get_inference_policy(device=env.device)
        estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
        if env.cfg.depth.use_camera:
            depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)
    else:
        algo_runner = AlgoRunner(log_dir = log_pth, env=env, args=args, env_name=args.task, device=args.device)
        all_snapshots = os.listdir(log_pth)
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
        snap_path = os.path.join(log_pth, all_snapshots[max_snap_idx])
        algo_runner.load_snapshot(snap_path)

        policy = algo_runner.ac_agent.actor
        policy.eval()
        estimator = algo_runner.estimator
        estimator.eval()

    cur_time = 0.0
    timestamps_states = []
    joint_pos_arr = []
    joint_vel_arr = []
    joint_torque_arr = []
    pos_cmds = []
    pos_cmds_ours = []
    for i in range(12):
        joint_pos_arr.append([])
        joint_vel_arr.append([])
        joint_torque_arr.append([])
        pos_cmds.append([])
        pos_cmds_ours.append([])
    contact_force_data = []
    contact_data = []
    for i in range(4):
        contact_data.append([])
        contact_force_data.append([])

    diffs_obs_proprio = []
    for i in range(env_cfg.env.n_proprio):
        diffs_obs_proprio.append([])

    print(f"pgains: {env.p_gains}, dgains: {env.d_gains}")
    print(f"action_scale: {env.cfg.control.action_scale}")
    print(f"clip_obs: {env.cfg.normalization.clip_observations}")
    print(f"clip_actions: {env.cfg.normalization.clip_actions}")
    print(f"dof_pos scale: {env.obs_scales.dof_pos} dof_vel scale: {env.obs_scales.dof_vel}")

    # rl_controller = RobotRLController("cuda:0")

    with torch.inference_mode():
        actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
        infos = {}
        infos["depth"] = env.depth_buffer.clone().to(args.device)[:, -1] if ppo_runner.if_depth else None

        for step_idx in range(10*int(env.max_episode_length)):
        # for step_idx in range(300):
            if args.use_jit:
                if env.cfg.depth.use_camera:
                    if infos["depth"] is not None:
                        depth_latent = torch.ones((env_cfg.env.num_envs, 32), device=env.device)
                        obs[:, 8:10] = 0.0
                        actions, depth_latent = policy_jit(obs.detach(), True, infos["depth"], depth_latent)
                    else:
                        depth_buffer = torch.ones((env_cfg.env.num_envs, 58, 87), device=env.device)
                        actions, depth_latent = policy_jit(obs.detach(), False, depth_buffer, depth_latent)
                else:
                    obs_jit = torch.cat((obs.detach()[:, :env_cfg.env.n_proprio+env_cfg.env.n_priv], obs.detach()[:, -env_cfg.env.history_len*env_cfg.env.n_proprio:]), dim=1)
                    actions = policy(obs_jit)
            else:
                if env.cfg.depth.use_camera:
                    if infos["depth"] is not None:
                        obs_student = obs[:, :env.cfg.env.n_proprio].clone()
                        obs_student[:, 6:8] = 0
                        print("depth info size")
                        print(infos["depth"].shape)
                        # depth_latent_and_yaw = depth_encoder(torch.zeros_like(infos["depth"]), obs_student)
                        depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)
                        depth_latent = depth_latent_and_yaw[:, :-2]
                        yaw = depth_latent_and_yaw[:, -2:]
                    obs[:, 6:8] = 1.5*yaw
                        
                else:
                    depth_latent = None

                if USE_THEIR_POLICY and hasattr(ppo_runner.alg, "depth_actor"):
                    heights = torch.zeros((env_cfg.env.num_envs, 132), dtype=torch.float32, device=env.device)
                    priv_latent = torch.zeros((env_cfg.env.num_envs, env_cfg.env.n_priv_latent), device=env.device)
                    priv_explicit = torch.zeros((env_cfg.env.num_envs, env_cfg.env.n_priv), device=env.device)
                    # print(f"obs buf shape {obs.shape}, heights: {heights}, priv expl shape: {priv_explicit.shape}, priv_lat_shape: {priv_latent.shape}, hist_buf_shape: {self.obs_history_buf.view(1, -1).shape}")

                    new_obs = torch.cat([obs.detach()[:, :env_cfg.env.n_proprio], heights, priv_explicit, priv_latent, env.obs_history_buf.view(env_cfg.env.num_envs, -1)], dim=-1)
                    # print(f"obs_history_buf size: {self.obs_history_buf.shape}")
                    # print(f"obs buf size: {self.obs_buf.shape}")
                    actions = ppo_runner.alg.depth_actor(new_obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
                else:
                    if USE_THEIR_POLICY:
                        actions = policy(obs, hist_encoding=True, scandots_latent=depth_latent)
                    else:
                        encoded_obs = algo_runner.encode_obs_phase1(obs, hist_encoding=True)
                        dists = policy(encoded_obs.detach())
                        actions = dists.sample()

            dof_pos = env.dof_pos[env.lookat_id, :]
            dof_vel = env.dof_vel[env.lookat_id, :]
            computed_torques = env.torques[env.lookat_id, :]
            contact_forces = torch.norm(env.contact_forces[env.lookat_id, env.feet_indices], dim=-1)
            for i in range(4):
                contact_force_data[i].append(contact_forces[i].detach().cpu().item())
                contact_data[i].append(env.contact_filt[env.lookat_id, i].detach().cpu().item())
        
            timestamps_states.append(cur_time)
            actions_scaled = actions * env.cfg.control.action_scale
            joint_pos_cmds = actions_scaled + env.default_dof_pos_all
            joint_pos_cmd = joint_pos_cmds[env.lookat_id, :]
            for sim_idx in range(12):
                joint_pos = dof_pos[sim_idx].detach().cpu().item()
                vel = dof_vel[sim_idx].detach().cpu().item()
                tau_est = computed_torques[sim_idx].detach().cpu().item()
                cmd = joint_pos_cmd[sim_idx].detach().cpu().item()

                joint_pos_arr[sim_idx].append(joint_pos)
                joint_vel_arr[sim_idx].append(vel)
                joint_torque_arr[sim_idx].append(tau_est)
                pos_cmds[sim_idx].append(cmd)

            # contact_thresh = 2.0
            # if infos["depth"] is not None:
            #     depth_buf = infos["depth"][env.lookat_id].unsqueeze(0)
            # print(f"our depth buf size: {depth_buf.shape}")
            # base_ang_vel = env.base_ang_vel[env.lookat_id].detach().cpu().numpy()
            # rpy = np.array([env.roll[env.lookat_id].detach().cpu().item(), env.pitch[env.lookat_id].detach().cpu().item(), env.yaw[env.lookat_id].detach().cpu().item()], dtype=np.float32)
            # pos = env.dof_pos[env.lookat_id, :].detach().cpu().numpy()
            # vel = env.dof_vel[env.lookat_id, :].detach().cpu().numpy()
            # tau = env.torques[env.lookat_id, :].detach().cpu().numpy()
            # foot_force = contact_forces.detach().cpu().numpy()

            # obs_ours, depth_latent_ours = rl_controller.arrs_to_obs(base_ang_vel, rpy, pos, vel, tau, foot_force, depth_buf, contact_thresh)
            # actions_ours = rl_controller.obs_latent_to_act(obs_ours, depth_latent_ours)
            # actions_ours_scaled = actions_ours * env.cfg.control.action_scale
            # joint_pos_cmds_ours = actions_ours_scaled + env.default_dof_pos_all
            # actions[env.lookat_id, :] = actions_ours[0]
            # for sim_idx in range(12):
            #     cmd = joint_pos_cmds_ours[0][sim_idx].detach().cpu().item()
            #     pos_cmds_ours[sim_idx].append(cmd)

            # our_obs_proprio = obs_ours[0, :env_cfg.env.n_proprio]
            # theirs_obs_proprio = new_obs[env.lookat_id, :env_cfg.env.n_proprio]
            # diffs = our_obs_proprio - theirs_obs_proprio

            # for i in range(env_cfg.env.n_proprio):
            #     diffs_obs_proprio[i].append(diffs[i].detach().cpu().item())

            obs, _, rews, dones, infos = env.step(actions.detach())
            if args.web:
                web_viewer.render(fetch_results=True,
                            step_graphics=True,
                            render_all_camera_sensors=True,
                            wait_for_page_load=True)
            print("time:", env.episode_length_buf[env.lookat_id].item() / 50, 
                "cmd vx", env.commands[env.lookat_id, 0].item(),
                "actual vx", env.base_lin_vel[env.lookat_id, 0].item(), )

            if dones[env.lookat_id].detach().cpu().item():
                break
            
            cur_time += env.dt
    os.makedirs("tmp", exist_ok=True)
    file_dir = f"tmp/sim_plots"
    os.makedirs(file_dir, exist_ok=True)
    for sim_idx in range(12):
        sim_name = env.dof_names[sim_idx]
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8))

        axes[0].plot(timestamps_states, joint_pos_arr[sim_idx], label=f"{sim_name}_pos")
        axes[0].plot(timestamps_states, pos_cmds_ours[sim_idx], linestyle='dashed', label=f"{sim_name}_pos_cmd_ours")
        axes[0].plot(timestamps_states, pos_cmds[sim_idx], linestyle='dashed', label=f"{sim_name}_pos_cmd_env")

        # axes[0].plot(tracked_data[:, 0], default_pos_targ + tracked_data[:, 4]  * action_scale, linestyle='dashed', label=f"{target_joint_name}_goal_pos")
        axes[0].set_ylabel("pos")
        axes[0].legend(loc='lower left')
        axes[1].plot(timestamps_states, joint_vel_arr[sim_idx], label=f"{sim_name}_vel")
        axes[1].set_ylabel("vel")
        axes[1].legend(loc='lower left')
        axes[2].plot(timestamps_states, joint_torque_arr[sim_idx], label=f"{sim_name}_torque")
        axes[2].set_ylabel("torque")
        axes[2].legend(loc='lower left')
        # axes[3].plot(timestamps_states, tracked_data[:, 4] * action_scale, label=f"{sim_name}_pos_diff")
        # axes[3].set_ylabel("pos_diff")
        # axes[3].legend(loc='lower left')
        # axes[3].set_xlabel("time")

        fig.savefig(os.path.join(file_dir, f"sim_joint_traj_{sim_name}.png"))
        plt.close(fig)

    fig, axes = plt.subplots(2, 4, sharex=True, figsize=(10, 14))

    for i in range(4):
        axes[0][i].plot(timestamps_states, contact_force_data[i], label=f"contact_force_{i}")
        axes[1][i].plot(timestamps_states, contact_data[i], label=f"contact_{i}")
        axes[1][i].set_xlabel("time")
        axes[0][i].set_ylabel("contact_force")
        axes[1][i].set_ylabel("contact_bool")
    

    fig.savefig(os.path.join(file_dir, f"sim_contact_plots.png"))

    plt.close(fig)

    diffs_obs_arr = np.array(diffs_obs_proprio)
    sum_abs_diffs = np.sum(np.abs(diffs_obs_arr), axis=1)
    top_n = 6
    top_indices = (-sum_abs_diffs).argsort()[:top_n]


    fig, ax = plt.subplots(figsize=(10, 14))
    for i in range(env_cfg.env.n_proprio):
        label=None
        if i in top_indices:
            label=f"{i}"
        ax.plot(timestamps_states, diffs_obs_proprio[i], label=label)
    ax.set_xlabel("time")
    ax.set_ylabel("obs_diffs")
    ax.legend()
    fig.savefig(os.path.join(file_dir, f"sim_obs_diffs.png"))


if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
