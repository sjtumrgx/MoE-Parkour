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

import matplotlib.pyplot as plt

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
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


def get_control(num_envs: int, cur_time: float):
    pass



def play(args):


    if args.web:
        web_viewer = webviewer.WebViewer()
    faulthandler.enable()
    exptid = args.exptid
    log_pth = "/docker_mount/logs/{}/".format("torque_data")
    print(f"log_pth: {log_pth}")

    # args.task = "go2"
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 16
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
                                    "parkour_flat": 0.,
                                    "parkour_step": 0.2,
                                    "parkour_gap": 0.2, 
                                    "demo": 0.2}
    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True
    
    env_cfg.depth.angle = [0, 1]
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

    target_joint_idx = -1
    target_joint_name = "FL_calf_joint"
    for i in range(env.num_dofs):
        name = env.dof_names[i]
        print(name)
        if name == target_joint_name:
            target_joint_idx = i
    
    print("default pos")
    print(env.default_dof_pos_all[0])

    if args.web:
        web_viewer.setup(env)
    run_time = 5.0
    start_move = 2.0
    move_radians = -np.pi / 8
    time_to_move = 0.5
    time_to_stay = 0.5
  
    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)

    num_iterations = int(run_time / env.dt)

    robot_id = env.lookat_id

    # we track time, pos, and vel
    tracked_data = np.zeros((num_iterations, 5), dtype=np.float32)
    # tracked_data[:, 4] = env.default_dof_pos_all[robot_id, target_joint_idx].cpu().numpy()
    tracked_data[:, 4] = 0.0
    
    increments_to_move = int(time_to_move / env.dt)
    angle_increment = move_radians / increments_to_move

    start_up_move_idx = int(start_move/env.dt)

    start_down_move_idx = int((start_move + time_to_move + time_to_stay)/env.dt)

    # action scale: target angle = actionScale * action + defaultAngle
    action_scale = 0.25
    # moves = env.default_dof_pos_all[robot_id, target_joint_idx].cpu().numpy() + np.arange(increments_to_move) * angle_increment
    default_pos_targ = env.default_dof_pos_all[robot_id, target_joint_idx].cpu().numpy() 
    moves = (1/action_scale) * np.arange(increments_to_move) * angle_increment


    tracked_data[start_up_move_idx: start_up_move_idx + increments_to_move, 4] = moves

    tracked_data[start_up_move_idx + increments_to_move : start_down_move_idx, 4] = moves[-1] # env.default_dof_pos_all[robot_id, target_joint_idx].cpu().numpy() + move_radians

    tracked_data[start_down_move_idx: start_down_move_idx + increments_to_move, 4] = np.flip(moves)

    cur_time = 0.0

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)

    reindex_arr = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

    transformed_idx = reindex_arr[target_joint_idx]

    print(f"orig idx: {target_joint_idx}, their idx: {transformed_idx}")

    cur_time = 0.0

    for i in range(num_iterations):

        tracked_data[i, 0] = cur_time

        dof_pos = env.dof_pos[robot_id, target_joint_idx].item()
        dof_vel = env.dof_vel[robot_id, target_joint_idx].item()
        computed_torques = env.torques[robot_id, target_joint_idx].item()
        tracked_data[i, 1] = dof_pos
        tracked_data[i, 2] = dof_vel
        tracked_data[i, 3] = computed_torques

        actions[robot_id, target_joint_idx] = torch.tensor(tracked_data[i, 4], device=env.device)

        obs, _, rews, dones, infos = env.step(actions.detach())

        print(f"dof pos: {env.dof_pos[robot_id, :]}")
        base_pos = (env.root_states[robot_id, :3]).cpu().numpy()
        print(f"base pos: {base_pos}")

        cur_time += env.dt
        

    fig, axes = plt.subplots(4, 1, sharex=True)

    axes[0].plot(tracked_data[:, 0], tracked_data[:, 1], label=f"{target_joint_name}_pos")
    axes[0].plot(tracked_data[:, 0], default_pos_targ + tracked_data[:, 4]  * action_scale, linestyle='dashed', label=f"{target_joint_name}_goal_pos")
    axes[0].set_ylabel("pos")
    axes[0].legend()
    axes[1].plot(tracked_data[:, 0], tracked_data[:, 2], label=f"{target_joint_name}_vel")
    axes[1].set_ylabel("vel")
    axes[1].legend()
    axes[2].plot(tracked_data[:, 0], tracked_data[:, 3], label=f"{target_joint_name}_torque")
    axes[2].set_ylabel("torque")
    axes[2].legend()
    axes[3].plot(tracked_data[:, 0], tracked_data[:, 4] * action_scale, label=f"{target_joint_name}_pos_diff")
    axes[3].set_ylabel("pos_diff")
    axes[3].set_xlabel("time")
    axes[3].legend()
    
    plt.show()


if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
