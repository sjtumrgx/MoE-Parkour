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

import time
import os
from collections import deque
import statistics
from typing import List

# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim

# import ml_runlog
import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import sys
from copy import copy, deepcopy
import warnings
from rl_lib.agent import load_actor_from_file, load_depth_from_file, ActorType


ENV_DICT = {
    "asset": {
        "angular_damping": 0.0,
        "armature": 0.0,
        "collapse_fixed_joints": True,
        "default_dof_drive_mode": 3,
        "density": 0.001,
        "disable_gravity": False,
        "dof_velocity_o verride": 35.0,
        "file": "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf",
        "fix_base_link": False,
        "flip_visual_attachments": True,
        "foot_name": "foot",
        "front_hi p_names": ["FL_hip_joint", "FR_hip_joint"],
        "linear_damping": 0.0,
        "max_angular_velocity": 1000.0,
        "max_linear_velocity": 1000.0,
        "name": "go2",
        "penalize_contacts_on": [" thigh", "calf"],
        "rear_hip_names": ["RL_hip_joint", "RR_hip_joint"],
        "replace_cylinder_with_capsule": True,
        "sdk_dof_range": {
            "Hip_max": 1.0472,
            "Hip_min": -1.0472,
            "Front _Thigh_max": 3.4907,
            "Front_Thigh_min": -1.5708,
            "Rear_Thingh_max": 4.5379,
            "Rear_Thingh_min": -0.5236,
            "Calf_max": -0.83776,
            "Calf_min": -2.7227,
        },
        "self_collisions": 1,
        " terminate_after_contacts_on": ["base"],
        "thickness": 0.01,
    },
    "commands": {
        "ang_vel_clip": 0.4,
        "crclm_incremnt": {
            "ang_vel_yaw": 0.1,
            "heading": 0.5,
            "lin_vel_x": 0.1,
            "lin _vel_y": 0.1,
        },
        "curriculum": False,
        "heading_command": True,
        "lin_vel_clip": 0.2,
        "max_curriculum": 1.0,
        "max_ranges": {
            "ang_vel_yaw": [0, 0],
            "heading": [-1.6, 1.6],
            "lin _vel_x": [0.3, 0.8],
            "lin_vel_y": [-0.3, 0.3],
        },
        "num_commands": 4,
        "ranges": {
            "ang_vel_yaw": [0, 0],
            "heading": [0, 0],
            "lin_vel_x": [0.0, 1.5],
            "lin_vel_y": [0.0, 0.0],
        },
        "resampling_time": 60,
        "waypoint_delta": 0.7,
    },
    "control": {
        "action_scale": 0.25,
        "computer_clip_torque": False,
        "control_type": "P",
        "damping": {"joint": 1.0},
        "decimation ": 4,
        "motor_clip_torque": True,
        "stiffness": {"joint": 40.0},
    },
    "debug": {"render_vis": False},
    "depth": {
        "angle": [0, 1],
        "buffer_len": 2,
        "gaussian_blur_kernel": 3,
        "gaussian_blur_sigma": (0.1, 2.0),
        "contour_detection_kernel_size": 3,
        "contour_threshold": 3.0,
        "contour_nuke_prob": 0.1,
        "camera_num_envs": 192,
        "camera _terrain_num_cols": 20,
        "camera_terrain_num_rows": 10,
        "dis_noise": 0.0,
        "far_clip": 3.0,
        "horizontal_fov": 88,
        "invert": True,
        "near_clip": 0.15,
        "left_clip": 20,
        "right_clip": 5,
        "bottom_clip": 16,
        "artifact_prob": 0.001,
        "artifact_height_mean_std": (3, 3),
        "artifact_width_mean_std": (3, 3),
        "original": (160, 120),
        "position": [0.24, -0.0175, 0.12],
        "resized": (87, 58),
        "scale": 1,
        "update_interval": 5,
        "use_camera": True,
    },
    "domain_rand": {
        "action_buf_len": 8,
        "action_curr_step": [1, 1],
        "action_curr_step_scratch": [0, 1],
        "action_delay": False,
        "action_delay_view": 1,
        "added_com_range": [-0.2, 0.2],
        "added_mass_range": [0.0, 3.0],
        "delay_update_global_ste ps": 192000,
        "friction_range": [0.6, 2.0],
        "max_push_vel_xy": 0.5,
        "motor_strength_range": [0.8, 1.2],
        "push_interval_s": 6,
        "push_robots": False,
        "randomize_base_com": False,
        "randomize_base_mass": False,
        "randomize_friction": True,
        "randomize_motor": True,
    },
    "env": {
        "contact_buf_len": 100,
        "env_spacing": 3.0,
        "episode_length_s": 60,
        "history_encoding": True,
        "history_len": 10,
        "include_foot_contacts": True,
        "n_priv": 9,
        "n_priv_latent": 29,
        "n_proprio": 48,
        "n_scan": 132,
        "next_goal_threshold": 0.2,
        "num_ac tions": 12,
        "num_envs": 4,
        "num_future_goal_obs": 2,
        "num_observations": 753,
        "num_privileged_obs": None,
        "obs_type": "og",
        "rand_pitch_range": 1.6,
        "rand_y_range": 0.5,
        " rand_yaw_range": 1.2,
        "randomize_start_pitch": False,
        "randomize_start_pos": False,
        "randomize_start_vel": False,
        "randomize_start_y": False,
        "randomize_start_yaw": False,
        "reach_goal_delay": 0.1,
        "reorder_dofs": True,
        "send_timeouts": True,
    },
    "init_member_classes": {},
    "init_state": {
        "ang_vel": [0.0, 0.0, 0.0],
        "default_joint_angles": {
            "FL_ hip_joint": 0.1,
            "FL_thigh_joint": 0.7,
            "FL_calf_joint": -1.5,
            "FR_hip_joint": -0.1,
            "FR_thigh_joint": 0.7,
            "FR_calf_joint": -1.5,
            "RL_hip_joint": 0.1,
            "RL_thigh_joint": 1.0,
            "RL_calf_joint": -1.5,
            "RR_hip_joint": -0.1,
            "RR_thigh_joint": 1.0,
            "RR_calf_joint": -1.5,
        },
        "lin_vel": [0.0, 0.0, 0.0],
        "pos": [0.0, 0.0, 0.5],
        "rot": [0.0, 0.0, 0.0, 1.0],
    },
    "noise": {
        "add_noise": True,
        "noise_level": 1.0,
        "noise_scales": {
            "ang_vel": 0.05,
            "dof_pos": 0.01,
            "dof_vel": 0.05,
            "gravity": 0.02,
            "height_measurements": 0.02,
            " lin_vel": 0.05,
            "rotation": 0.0,
        },
        "quantize_height": True,
    },
    "normalization": {
        "clip_actions": 1.2,
        "clip_observations": 100.0,
        "obs_scales": {
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "height_measurements": 5.0,
            "lin_vel": 2.0,
        },
    },
    "play": {"load_student_config": False, "mask_priv_obs": False},
    "rewards": {
        "base_height_target": 1.0,
        " max_contact_force": 100,
        "only_positive_rewards": True,
        "scales": {
            "action_rate": -0.1,
            "ang_vel_xy": -0.05,
            "collision": -10.0,
            "delta_torques": -1e-07,
            "dof_acc": -2.5e-07,
            "dof_error": -0.04,
            "feet_edge": -1,
            "feet_stumble": -1,
            "hip_pos": -0.5,
            "lin_vel_z": -1.0,
            "orientation": -1.0,
            "torques": -1e-05,
            "tracking_goal_vel": 1.5,
            "trackin g_yaw": 0.5,
        },
        "soft_dof_pos_limit": 0.9,
        "soft_dof_vel_limit": 1,
        "soft_torque_limit": 0.4,
        "tracking_sigma": 0.2,
    },
    "seed": 1,
    "sim": {
        "dt": 0.005,
        "gravity": [0.0, 0.0, -9.81],
        "physx": {
            "bounce_threshold_velocity": 0.5,
            "contact_collection": 2,
            "contact_offset": 0.01,
            "default_buffer_size_multiplier": 5,
            "max_depenetration_velocity": 1.0,
            "max_gpu_contact_pairs": 8388608,
            "num_position_iterations": 4,
            "num_threads": 10,
            "num_velocity_iterations": 0,
            "rest_offset": 0.0,
            "solver_type": 1,
        },
        "substeps": 1,
        "up _axis": 1,
    },
    "terrain": {
        "all_vertical": False,
        "border_size": 5,
        "curriculum": False,
        "downsampled_scale": 0.075,
        "dynamic_friction": 1.0,
        "edge_width_thresh": 0.05,
        "gap_ size": [0.02, 0.1],
        "height": [0.02, 0.02],
        "hf2mesh_method": "grid",
        "horizontal_scale": 0.05,
        "horizontal_scale_camera": 0.1,
        "max_difficulty": True,
        "max_error": 0.1,
        " max_error_camera": 2,
        "max_init_terrain_level": 5,
        "measure_heights": True,
        "measure_horizontal_noise": 0.0,
        "measured_points_x": [
            -0.45,
            -0.3,
            -0.15,
            0,
            0.15,
            0.3,
            0.45,
            0.6,
            0.75,
            0.9,
            1.05,
            1.2,
        ],
        "measured_points_y": [
            -0.75,
            -0.6,
            -0.45,
            -0.3,
            -0.15,
            0.0,
            0.15,
            0.3,
            0.45,
            0.6,
            0.75,
        ],
        "mesh_type": "trimesh",
        "no_flat": True,
        "num_cols": 5,
        "num_goals": 8,
        "num_rows": 5,
        "origin_zero_z": True,
        "restitution": 0.0,
        "selected": False,
        "simplify_grid": False,
        "slope_treshold": 1.5,
        "static_friction": 1.0,
        "step ping_stone_distance": [0.02, 0.08],
        "terrain_dict": {
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
            "parko ur": 0.2,
            "parkour_hurdle": 0.2,
            "parkour_flat": 0.0,
            "parkour_step": 0.2,
            "parkour_gap": 0.2,
            "demo": 0.2,
        },
        "terrain_kwargs": None,
        "terrain_length": 18.0,
        "terrain_propo rtions": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.2,
            0.2,
            0.0,
            0.2,
            0.2,
            0.2,
        ],
        "terrain_width": 4,
        "vertical_scale": 0.005,
        "y_range": [-0.1, 0.1],
    },
    "viewer": {"lookat": [11.0, 5, 3.0], "pos": [10, 0, 6], "ref_env": 0},
}

TRAIN_DICT = {
    "algorithm": {
        "clip_param": 0.2,
        "dagger_update_freq": 20,
        "desired_kl": 0.01,
        "entropy_coef": 0.01,
        "gamma": 0.99,
        "lam": 0.95,
        "learning_rate": 0.0002,
        "max_grad_norm": 1.0,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "priv_reg_coef_schedual": [0, 0.1, 2000, 3000],
        "priv_reg_coef_schedual_resume": [0, 0.1, 0, 1],
        "schedule": "adapti ve",
        "use_clipped_value_loss": True,
        "value_loss_coef": 1.0,
        "moe_n_experts": 3,
        "moe_top_k": 2,
        "moe_loss_coeff": 0.01,
        "moe_noise_mat_init": -0.5,
        "gate_noise_with_x": False,
        "dropout": False,
        "dropout_prob": 0.2,
        "moe_layer_idx": 1,
        "past_obs_for_depth_encoder": 2,
        "delta_yaw_thresh": 0.6,
    },
    "depth_encoder": {
        "buffer_len": 2,
        "depth_shape": (87, 58),
        "hidden_dims": 512,
        "if_depth": True,
        "learning_rate": 0.001,
        "num_steps_per_env": 120,
    },
    "estimator": {
        "hidden_dims": [128, 64],
        "learning_rate": 0.0001,
        "num_prop": 48,
        "num_scan": 132,
        "priv_states_dim": 9,
        "train_with_estimated_states": True,
    },
    "init_member_classes": {},
    "policy": {
        "activation": "elu",
        "actor_hidden_dims": [512, 256, 128],
        "continue_from_last_std": True,
        "critic_hidden_dims": [512, 256, 128],
        "init_noise_std": 1.0,
        "priv_encoder_dims": [64, 20],
        "rnn_hidden_size": 512,
        "rnn_num_layers": 1,
        "rnn_type": "lstm",
        "scan_encoder_dims": [128, 64, 32],
        "tanh_encoder_output": False,
    },
    "runner": {
        "algorithm_class_name": "PPO",
        "checkpoint": -1,
        "experiment_name": "rough_go2",
        "load_run": -1,
        "max_iterations": 50000,
        "num_steps_per_env": 24,
        "policy_class_name": "ActorCritic",
        "resume": True,
        "resume_path": None,
        "run_name": "",
        "save_interval": 100,
    },
    "runner_class_name": "OnPolicyRunner",
    "seed": 1,
}
NUM_ACTIONS = 12
ENV_DICT["env"]["num_actions"] = NUM_ACTIONS

CONTACT_THRESHOLD = 30.0


ACTION_SCALE = ENV_DICT["control"]["action_scale"]
NUM_DOF = 12

POS_STOP = 2.146e9
VEL_STOP = 16000.0

DOF_MAP = [  # from isaacgym simulation joint order to real robot joint order
    3,
    4,
    5,
    0,
    1,
    2,
    9,
    10,
    11,
    6,
    7,
    8,
]

# from real to sim
DOF_MAP_TO_SIM = []

for real_idx in range(len(DOF_MAP)):
    for i in range(len(DOF_MAP)):
        sim_idx = DOF_MAP[i]
        if real_idx == sim_idx:
            DOF_MAP_TO_SIM.append(i)
            break

# in unitree lowstate, foot contact force
# index 3 is back left foot, index 2 is back right foot, index 0 is front rFight,
# index 1 is front left

DOF_NAMES = [  # NOTE: order matters. This list is the order in simulation.
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]


# in sim order
FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
# # from isaacgym simulation joint order to real robot joint order
FOOT_MAP = [1, 0, 3, 2]
# from real to sim map
FOOT_MAP_TO_SIM = []
for real_idx in range(4):
    for i in range(len(FOOT_MAP)):
        sim_idx = FOOT_MAP[i]
        if real_idx == sim_idx:
            FOOT_MAP_TO_SIM.append(i)
            break

DEFAULT_JOINT_ANGLES = {  # 12 joints in the order of simulation
    "FL_hip_joint": 0.1,
    "FL_thigh_joint": 0.7,
    "FL_calf_joint": -1.5,
    "FR_hip_joint": -0.1,
    "FR_thigh_joint": 0.7,
    "FR_calf_joint": -1.5,
    "RL_hip_joint": 0.1,
    "RL_thigh_joint": 1.0,
    "RL_calf_joint": -1.5,
    "RR_hip_joint": -0.1,
    "RR_thigh_joint": 1.0,
    "RR_calf_joint": -1.5,
}


SIT_JOINT_ANGLES = {  # 12 joints in the order of simulation
    "FL_hip_joint": 0.1,
    "FL_thigh_joint": 1.2,
    "FL_calf_joint": -2.3,
    "FR_hip_joint": -0.1,
    "FR_thigh_joint": 1.2,
    "FR_calf_joint": -2.3,
    "RL_hip_joint": 0.1,
    "RL_thigh_joint": 1.8,
    "RL_calf_joint": -2.3,
    "RR_hip_joint": -0.1,
    "RR_thigh_joint": 1.8,
    "RR_calf_joint": -2.3,
}

SIT_JOINT_ANGLES_ARR = [SIT_JOINT_ANGLES[k] for k in DOF_NAMES]

DOF_SIGNS = [1.0] * 12
JOINT_LIMITS_HIGH = torch.tensor(
    [
        1.0472,
        3.4907,
        -0.83776,
        1.0472,
        3.4907,
        -0.83776,
        1.0472,
        4.5379,
        -0.83776,
        1.0472,
        4.5379,
        -0.83776,
    ],
    device="cpu",
    dtype=torch.float32,
)

# JOINT_LIMITS_LOW = torch.tensor([
#     -1.0472, -1.5708, -2.7227,
#     -1.0472, -1.5708, -2.7227,
#     -1.0472, -0.5236, -2.7227,
#     -1.0472, -0.5236, -2.7227,
# ], device= "cpu", dtype= torch.float32)

JOINT_LIMITS_LOW = torch.tensor(
    [
        -1.0472,
        -3,
        -3.3,
        -1.0472,
        -3,
        -3.3,
        -1.0472,
        -1,
        -3.3,
        -1.0472,
        -1,
        -3.3,
    ],
    device="cpu",
    dtype=torch.float32,
)

# TORQUE_LIMITS = torch.tensor([ # from urdf and in simulation order
#             25, 40, 40,
#             25, 40, 40,
#             25, 40, 40,
#             25, 40, 40,
#         ], device= "cpu", dtype= torch.float32)

TORQUE_LIMITS = torch.tensor(
    [  # from urdf and in simulation order
        35,
        65,
        65,
        35,
        65,
        65,
        35,
        65,
        65,
        35,
        65,
        65,
    ],
    device="cpu",
    dtype=torch.float32,
)


TURN_ON_MOTOR_MODE = [0x01] * 12

# 25kp, 0.5kd, is what visiting phd student uses
STIFFNESS = 40.0
DAMPING = 1.0

DOF_SIGNS = [1.0] * 12


def quat_apply_inverse(quat, vec):
    """Apply an inverse quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (4).
        vec: The vector in (x, y, z). Shape is (3).

    Returns:
        The rotated vector in (x, y, z). Shape is (3).
    """
    # extract components from quaternions
    xyz = quat[1:]
    t = np.cross(xyz, vec) * 2
    return vec - quat[0] * t + np.cross(xyz, t)

TRAIN_DICT["policy"]["actor_hidden_dims"] = [512, 256, 256]
SNAP_PATH = "/docker_mount/logs/parkour_new/555-55-moe-top4-16-cam/snapshot_5000.pt"
ACTOR_TYPE = ActorType.MIX_OF_EXPERTS
ACTOR_KWARGS = {
    "moe_loss_coeff": 0.1,
    "moe_n_experts": 16,
    "moe_top_k": 4,
    "moe_noise_mat_init": 0.0,
    "scan_idx": ENV_DICT["env"]["n_proprio"],
    "scan_len": TRAIN_DICT["policy"]["scan_encoder_dims"][-1],
    "gate_noise_with_x": True,
    "moe_layer_idx": 1,
}

DO_ENCODING_TRICKS = True
USE_CAMERA = True


class RobotRLController:

    def __init__(
        self,
        device: str,
        use_camera: bool = False,
        vx: float = 0.4,
    ):

        dt = 0.005
        self.decimation = 4
        self.dt = dt * self.decimation

        self.device = device

        self.next_goal_threshold = 0.2

        self.cur_goal_idx = 0
        self.goals = np.array([[0.5, 0], [1.0, 0], [1.5, 0]])
        self.max_goals_idx = self.goals.shape[0] - 1

        self.global_counter = 0
        self.prior_delta_yaw = None
        self.prior_next_delta_yaw = None

        self.obs_scales = {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "height_measurements": 5.0,
            "clip_observations": 100,
            "clip_actions": 1.2,
        }

        self.ang_vel_clip = 0.4

        self.commands = np.array(
            [vx, 0.0, 0.0, 0.0]
        )  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)

        self.joint_default_angles = DEFAULT_JOINT_ANGLES
        self.dof_names = DOF_NAMES

        self.num_dofs = len(self.joint_default_angles.keys())
        self.default_dof_pos = np.zeros(self.num_dofs)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.joint_default_angles[name]
            self.default_dof_pos[i] = angle

        self.action_buf_len = 8
        self.action_buf = torch.zeros(
            (self.action_buf_len, self.num_dofs),
            device=self.device,
            dtype=torch.float32,
        )

        self.setup()

        self.mass_params_tensor = torch.tensor(
            [0.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32
        )

        self.base_lin_vel = np.zeros(3, dtype=np.float32)

        self.init_obs_history = True
        self.initial_obs = True
        self.obs_history_buf = torch.zeros(
            1,
            ENV_DICT["env"]["history_len"],
            ENV_DICT["env"]["n_proprio"],
            device=self.device,
            dtype=torch.float32,
        )
        self.last_contacts = np.array([False] * 4, dtype=bool)

        # for logging/plotting purposes

        self.timestamps_states = []
        self.joint_pos_arr = []
        self.joint_vel_arr = []
        self.joint_torque_arr = []
        self.pos_cmds_theirs = []
        self.pos_cmds_ours = []
        self.real_idx = 0
        self.noisy_idx = 1
        # we have real, and obs, for some of them
        for j in range(2):
            self.joint_pos_arr.append([])
            self.joint_vel_arr.append([])
            for i in range(12):
                self.joint_pos_arr[j].append([])
                self.joint_vel_arr[j].append([])
        for i in range(12):
            self.pos_cmds_theirs.append([])
            self.pos_cmds_ours.append([])
            self.joint_torque_arr.append([])
        self.contact_force_data = []
        for i in range(4):
            self.contact_force_data.append([])

        self.contact_data = []
        for j in range(2):
            self.contact_data.append([])
            for i in range(4):
                self.contact_data[j].append([])

        self.rpy = []
        self.base_ang_vel = []
        for j in range(2):
            self.rpy.append([])
            self.base_ang_vel.append([])
            for i in range(3):
                self.rpy[j].append([])
                self.base_ang_vel[j].append([])

        if not USE_CAMERA:
            # self.our_agent_path = SNAP_PATH
            # print(f"loading: {self.our_agent_path}")
            # # in dim from proprioception plus history lookback
            # self.our_agent = load_actor_from_file(self.our_agent_path, device="cuda:0", in_dim=ENV_DICT["env"]["n_proprio"] + (ENV_DICT["env"]["history_len"] * ENV_DICT["env"]["n_proprio"]),
            #     out_dim=NUM_DOF, hidden_dims=TRAIN_DICT['policy']['actor_hidden_dims'], activation=nn.ELU)
            # self.our_agent.eval()
            raise NotImplementedError
        else:
            self.our_agent_path = SNAP_PATH
            print(f"loading: {self.our_agent_path}")
            self.scan_output_dim = TRAIN_DICT["policy"]["scan_encoder_dims"][-1]

            self.n_proprio = ENV_DICT["env"]["n_proprio"]
            self.prop_hist_len = ENV_DICT["env"]["history_len"]

            self.num_priv_yaw = 2

            self.priv_encoder_dims = TRAIN_DICT["policy"]["priv_encoder_dims"]
            self.priv_encoder_output_dim = self.priv_encoder_dims[-1]
            self.priv_encoder_dims = self.priv_encoder_dims[:-1]

            self.n_priv_latent = ENV_DICT["env"]["n_priv_latent"]
            self.n_priv_explicit = ENV_DICT["env"]["n_priv"]

            self.past_obs_for_depth_encoder = TRAIN_DICT["algorithm"][
                "past_obs_for_depth_encoder"
            ]

            self.actor_type = ACTOR_TYPE
            self.actor_type_kwargs = ACTOR_KWARGS

            if not DO_ENCODING_TRICKS:
                self.our_agent, self.depth_backbone, self.depth_encoder, extras = (
                    load_depth_from_file(
                        self.our_agent_path,
                        device=self.device,
                        actor_type=self.actor_type,
                        actor_type_kwargs=self.actor_type_kwargs,
                        in_dim=self.n_proprio
                        + self.scan_output_dim
                        + (self.prop_hist_len * self.n_proprio)
                        + self.num_priv_yaw,
                        out_dim=NUM_DOF,
                        hidden_dims=TRAIN_DICT["policy"]["actor_hidden_dims"],
                        activation=nn.ELU,
                        depth_output_size=self.scan_output_dim,
                        n_proprio=ENV_DICT["env"]["n_proprio"],
                        past_obs_for_depth_encoder=self.past_obs_for_depth_encoder,
                        encoding_tricks=DO_ENCODING_TRICKS,
                    )
                )
            else:

                actor_in_dim = (
                    self.n_proprio
                    + self.prop_hist_len * self.n_proprio
                    + self.scan_output_dim
                    + self.num_priv_yaw
                    + self.priv_encoder_output_dim
                    + self.n_priv_explicit
                )
                print(
                    f"{self.n_proprio}, {self.prop_hist_len}, {self.scan_output_dim}, {self.num_priv_yaw}, {self.priv_encoder_output_dim}, {self.n_priv_explicit}"
                )
                print(f"actor in: {actor_in_dim}")
                self.our_agent, self.depth_backbone, self.depth_encoder, extras = (
                    load_depth_from_file(
                        self.our_agent_path,
                        device=self.device,
                        actor_type=self.actor_type,
                        actor_type_kwargs=self.actor_type_kwargs,
                        in_dim=actor_in_dim,
                        out_dim=NUM_DOF,
                        hidden_dims=TRAIN_DICT["policy"]["actor_hidden_dims"],
                        activation=nn.ELU,
                        depth_output_size=self.scan_output_dim,
                        n_proprio=ENV_DICT["env"]["n_proprio"],
                        past_obs_for_depth_encoder=self.past_obs_for_depth_encoder,
                        encoding_tricks=DO_ENCODING_TRICKS,
                        priv_encoder_output_dim=self.priv_encoder_output_dim,
                        n_priv_explicit=self.n_priv_explicit,
                        estimator_hidden_dim=TRAIN_DICT["estimator"]["hidden_dims"],
                        history_len=self.prop_hist_len,
                        dropout=TRAIN_DICT["algorithm"]["dropout"],
                        dropout_prob=TRAIN_DICT["algorithm"]["dropout_prob"],
                    )
                )

                self.estimator, self.history_encoder = extras

        self.resize_transform_output_shape = torchvision.transforms.Resize(
            (ENV_DICT["depth"]["resized"][1], ENV_DICT["depth"]["resized"][0]),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
        )

        self.resize_transform_640to_4th = torchvision.transforms.Resize(
            (ENV_DICT["depth"]["original"][1], ENV_DICT["depth"]["original"][0]),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
        )

        self.gaussian_blur_transform = torchvision.transforms.GaussianBlur(
            ENV_DICT["depth"]["gaussian_blur_kernel"],
            sigma=ENV_DICT["depth"]["gaussian_blur_sigma"],
        )

        self.contour_detection_kernel = torch.zeros(
            (8, 1, 3, 3),
            dtype=torch.float32,
            device=self.device,
        )

        # emperical values to be more sensitive to vertical edges
        self.contour_detection_kernel[0, :, 1, 1] = 0.5
        self.contour_detection_kernel[0, :, 0, 0] = -0.5
        self.contour_detection_kernel[1, :, 1, 1] = 0.1
        self.contour_detection_kernel[1, :, 0, 1] = -0.1
        self.contour_detection_kernel[2, :, 1, 1] = 0.5
        self.contour_detection_kernel[2, :, 0, 2] = -0.5
        self.contour_detection_kernel[3, :, 1, 1] = 1.2
        self.contour_detection_kernel[3, :, 1, 0] = -1.2
        self.contour_detection_kernel[4, :, 1, 1] = 1.2
        self.contour_detection_kernel[4, :, 1, 2] = -1.2
        self.contour_detection_kernel[5, :, 1, 1] = 0.5
        self.contour_detection_kernel[5, :, 2, 0] = -0.5
        self.contour_detection_kernel[6, :, 1, 1] = 0.1
        self.contour_detection_kernel[6, :, 2, 1] = -0.1
        self.contour_detection_kernel[7, :, 1, 1] = 0.5
        self.contour_detection_kernel[7, :, 2, 2] = -0.5

    def log_proprio(
        self,
        idx,
        cur_time,
        dof_pos,
        dof_vel,
        torques,
        contact_forces,
        contact_filt,
        their_cmds,
        our_cmds,
        rpy,
        base_ang_vel,
    ):

        # TODO, don't log if noisy the below:
        # log_proprio(1, cur_time, pos_noisy, vel_noisy, None, None, contact_filt_noisy, None, None, rpy_noisy, ang_vel_noisy)

        for i in range(4):
            if contact_forces is not None:
                self.contact_force_data[i].append(contact_forces[i])
            if contact_filt is not None:
                self.contact_data[idx][i].append(contact_filt[i])

        if cur_time is not None:
            self.timestamps_states.append(cur_time)

        for sim_idx in range(12):
            joint_pos = dof_pos[sim_idx]
            vel = dof_vel[sim_idx]
            if their_cmds is not None:
                cmd_theirs = their_cmds[sim_idx]
                self.pos_cmds_theirs[sim_idx].append(cmd_theirs)
            if our_cmds is not None:
                cmd_ours = our_cmds[sim_idx]
                self.pos_cmds_ours[sim_idx].append(cmd_ours)

            if torques is not None:
                tau_est = torques[sim_idx]
                self.joint_torque_arr[sim_idx].append(tau_est)

            self.joint_pos_arr[idx][sim_idx].append(joint_pos)
            self.joint_vel_arr[idx][sim_idx].append(vel)

        for i in range(2):
            self.rpy[idx][i].append(rpy[i])
        if len(rpy) > 2:
            self.rpy[idx][2].append(rpy[2])
        for i in range(3):
            self.base_ang_vel[idx][i].append(base_ang_vel[i])

    def make_log_plots(self, out_dir: str, prefix: str, only_noisy: bool = False):

        for sim_idx in range(12):
            sim_name = DOF_NAMES[sim_idx]
            fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
            axes[0].plot(
                self.timestamps_states,
                self.joint_pos_arr[1][sim_idx],
                label=f"{sim_name}_pos_noisy",
            )
            axes[0].plot(
                self.timestamps_states,
                self.pos_cmds_ours[sim_idx],
                label=f"{sim_name}_pos_cmd_ours",
            )

            if len(self.pos_cmds_theirs[sim_idx]) > 1:
                axes[0].plot(
                    self.timestamps_states,
                    self.pos_cmds_theirs[sim_idx],
                    linestyle="dashed",
                    label=f"{sim_name}_pos_cmd_env",
                )

            if len(self.joint_pos_arr[0][sim_idx]) > 1:
                axes[0].plot(
                    self.timestamps_states,
                    self.joint_pos_arr[0][sim_idx],
                    label=f"{sim_name}_pos",
                )

            axes[0].set_ylabel("pos")
            axes[0].legend(loc="lower left")
            axes[1].plot(
                self.timestamps_states,
                self.joint_vel_arr[1][sim_idx],
                label=f"{sim_name}_vel_noisy",
            )
            if len(self.joint_vel_arr[0][sim_idx]) > 1:
                axes[1].plot(
                    self.timestamps_states,
                    self.joint_vel_arr[0][sim_idx],
                    label=f"{sim_name}_vel",
                )
            axes[1].set_ylabel("vel")
            axes[1].legend(loc="lower left")
            if len(self.joint_torque_arr[0]) > 1:
                axes[2].plot(
                    self.timestamps_states,
                    self.joint_torque_arr[sim_idx],
                    label=f"{sim_name}_torque",
                )
                axes[2].set_ylabel("torque")
                axes[2].legend(loc="lower left")
            # axes[3].plot(timestamps_states, tracked_data[:, 4] * action_scale, label=f"{sim_name}_pos_diff")
            # axes[3].set_ylabel("pos_diff")
            # axes[3].legend(loc='lower left')
            # axes[3].set_xlabel("time")

            fig.savefig(os.path.join(out_dir, f"{prefix}_joint_traj_{sim_name}.png"))
            plt.close(fig)

        fig, axes = plt.subplots(2, 4, sharex=True, figsize=(10, 14))

        for i in range(4):
            axes[1][i].plot(
                self.timestamps_states,
                self.contact_data[1][i],
                label=f"contact_{i}_noisy",
            )
            if not only_noisy:
                axes[1][i].plot(
                    self.timestamps_states,
                    self.contact_data[0][i],
                    label=f"contact_{i}",
                )
            if len(self.contact_force_data[i]) > 1:
                axes[0][i].plot(
                    self.timestamps_states,
                    self.contact_force_data[i],
                    label=f"contact_force_{i}",
                )
            axes[1][i].set_xlabel("time")
            axes[0][i].set_ylabel("contact_force")
            axes[1][i].set_ylabel("contact_bool")
        fig.savefig(os.path.join(out_dir, f"{prefix}_contact_plots.png"))

        # plot pitch/roll now
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 14))

        for i in range(2):
            axes[i].plot(self.timestamps_states, self.rpy[1][i], label=f"noisy")
            axes[i].set_title(f"rpy: {i}")
            axes[i].set_xlabel("time")
            if not only_noisy:
                axes[i].plot(self.timestamps_states, self.rpy[0][i], label=f"clean")

            axes[i].legend(loc="lower left")

        fig.savefig(os.path.join(out_dir, f"{prefix}_rpy.png"))

        # plot ang vel now

        fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 14))

        for i in range(3):
            axes[i].plot(
                self.timestamps_states, self.base_ang_vel[1][i], label=f"noisy"
            )
            axes[i].set_title(f"rpy: {i}")
            axes[i].set_xlabel("time")
            axes[i].set_ylabel("radians/s")
            axes[i].legend()
            if not only_noisy:
                axes[i].plot(
                    self.timestamps_states, self.base_ang_vel[0][i], label=f"clean"
                )

            axes[i].legend(loc="lower left")

        fig.savefig(os.path.join(out_dir, f"{prefix}_ang_vel.png"))

    @staticmethod
    def make_moe_weight_plot(out_dir: str, prefix: str, times: List[float], moe_weights: List[List[float]], weight_gradients_obs: List[List[List[float]]]):
        moe_weights_arr = np.array(moe_weights)
        n_experts = moe_weights_arr.shape[1]
        expert_utilization_pct = np.sum(np.where(moe_weights_arr>0.0, 1.0, 0.0), axis=0) / moe_weights_arr.shape[0]
        # filter to a subset of experts to make graph more readable. Experts chosen based on full graph and interesting cases like 
        # cyclicality or attention to depth....
        selected_experts = [5, 6, 7, 8]
        n_experts = len(selected_experts)
        
        fig, axes = plt.subplots(n_experts, sharex=True, sharey=True, figsize=(10, 5), constrained_layout=True, tight_layout=True)
        plot_idx = 0
        for i in selected_experts:
            axes[plot_idx].plot(times, moe_weights_arr[:, i], label=f"expert_{str(i).zfill(2)}")
            axes[plot_idx].set_ylabel(f"{str(i).zfill(1)}")
            plot_idx += 1
        axes[-1].set_xlabel("time from start (seconds)")
        axes[0].set_title("Selected Expert Weights vs Time")
        fig.supylabel('Expert Index, Average Expert Weight')
        fig.savefig(os.path.join(out_dir, f"{prefix}_moe_weights.pdf"))
        fig.savefig(os.path.join(out_dir, f"{prefix}_moe_weights.png"))

        # weight_gradients_obs is n_times x num_experts x n_obs
        weight_grads_arr = np.array(weight_gradients_obs)

        print("weights grad shape:")
        print(weight_grads_arr.shape)

        fig_positive_weight_contributors, axes_positive_weight_contributors = plt.subplots(n_experts, sharex=True, sharey=True, figsize=(10, 5))
        fig_positive_negative_contributors, axes_negative_weight_contributors = plt.subplots(n_experts, sharex=True, sharey=True, figsize=(10, 5))
        fig_all_contributors, axes_all_weight_contributors = plt.subplots(n_experts, sharex=True, sharey=True, figsize=(10, 5))
        fig_all_grads, axes_all_grads = plt.subplots(n_experts, sharex=True, sharey=True, figsize=(10, 5), constrained_layout=True, tight_layout=True)

        # first we check the most important elements of input for each expert across all time
        # then we histogram these? we put the top n most important indices and plot their 
        # average contribution across time when non-zero?
        times = np.array(times)
        top_n_for_timestep = 100
        top_n_across_timesteps = 6
        plot_idx = 0
        for i in selected_experts:
            # sometimes one row will be all zeros if the expert was not used or in topk for the MoE
            # let's drop those rows but keep all the columns
            expert_weight_grads = weight_grads_arr[:, i, :]
            non_zero_more_zero_ind = np.nonzero(expert_weight_grads > 0.0)
            non_zero_more_zero_ind = np.unique(non_zero_more_zero_ind[0])
            non_zero_less_zero_ind = np.nonzero(expert_weight_grads < 0.0)
            non_zero_less_zero_ind = np.unique(non_zero_less_zero_ind[0])
            all_contrib_ind = np.nonzero(expert_weight_grads != 0.0)
            all_contrib_ind = np.unique(all_contrib_ind[0])

            positive_contributors = expert_weight_grads[non_zero_more_zero_ind, :]
            negative_contributors = expert_weight_grads[non_zero_less_zero_ind, :]
            all_contributors = expert_weight_grads[all_contrib_ind, :]
            
            positive_times = times[non_zero_more_zero_ind]
            negative_times = times[non_zero_less_zero_ind]
            times_all_contrib = times[all_contrib_ind]

            top_pos_experts = np.zeros((positive_contributors.shape[0], top_n_for_timestep), dtype=np.int32)
            top_pos_expert_weight_grads = np.zeros((positive_contributors.shape[0], top_n_for_timestep), dtype=np.float32)
            top_pos_experts_across = np.zeros((top_n_across_timesteps), dtype=np.int32)

            top_neg_experts = np.zeros((negative_contributors.shape[0], top_n_for_timestep), dtype=np.int32)
            top_neg_expert_weight_grads = np.zeros((negative_contributors.shape[0], top_n_for_timestep), dtype=np.float32)
            top_neg_experts_across = np.zeros((top_n_across_timesteps), dtype=np.int32)

            top_all_experts = np.zeros((all_contributors.shape[0], top_n_for_timestep), dtype=np.int32)
            top_all_expert_weight_grads = np.zeros((all_contributors.shape[0], top_n_for_timestep), dtype=np.float32)
            top_all_experts_across = np.zeros((top_n_across_timesteps), dtype=np.int32)

            for j in range(positive_contributors.shape[0]):
                input_grads = positive_contributors[j]
                max_ind = np.argpartition(input_grads, -top_n_for_timestep)[-top_n_for_timestep:]
                sorted_indices = max_ind[np.argsort(input_grads[max_ind])]
                top_grads = input_grads[sorted_indices]
                top_pos_expert_weight_grads[j, :] = top_grads
                top_pos_experts[j, :] = sorted_indices
            
            for j in range(negative_contributors.shape[0]):
                input_grads = np.abs(negative_contributors[j])
                max_ind = np.argpartition(input_grads, -top_n_for_timestep)[-top_n_for_timestep:]
                sorted_indices = max_ind[np.argsort(input_grads[max_ind])]
                top_grads = input_grads[sorted_indices]
                top_neg_expert_weight_grads[j, :] = -top_grads
                top_neg_experts[j, :] = sorted_indices

            for j in range(all_contributors.shape[0]):
                input_grads = np.abs(all_contributors[j])
                max_ind = np.argpartition(input_grads, -top_n_for_timestep)[-top_n_for_timestep:]
                sorted_indices = max_ind[np.argsort(input_grads[max_ind])]
                top_grads = input_grads[sorted_indices]
                top_all_expert_weight_grads[j, :] = top_grads
                top_all_experts[j, :] = sorted_indices
            
            # now across all of these count the top most common max inputs...
            # np.unique returns sorted by unique val
            unique, counts = np.unique(top_pos_experts.flatten(), return_counts=True)
            axes_positive_weight_contributors[plot_idx].bar(unique, counts)
            count_sort_ind = np.argsort(-counts)
            top_pos_experts_across[:] = unique[count_sort_ind[:top_n_across_timesteps]]
            top_obs = [str(v) for v in top_pos_experts_across]
            top_obs = ",".join(top_obs)
            axes_positive_weight_contributors[plot_idx].set_title(f"top: {top_obs}")

            unique, counts = np.unique(top_neg_experts.flatten(), return_counts=True)
            axes_negative_weight_contributors[plot_idx].bar(unique, counts)
            count_sort_ind = np.argsort(-counts)
            top_neg_experts_across[:] = unique[count_sort_ind[:top_n_across_timesteps]]
            top_obs = [str(v) for v in top_neg_experts_across]
            top_obs = ",".join(top_obs)
            axes_negative_weight_contributors[plot_idx].set_title(f"top: {top_obs}")


            unique, counts = np.unique(top_all_experts.flatten(), return_counts=True)
            axes_all_weight_contributors[plot_idx].bar(unique, counts, color="k")
            axes_all_weight_contributors[plot_idx].set_ylabel(f"{str(i).zfill(1)}")
            # count_sort_ind = np.argsort(-counts)
            # top_neg_experts_across[:] = unique[count_sort_ind[:top_n_across_timesteps]]
            # top_obs = [str(v) for v in top_neg_experts_across]
            # top_obs = ",".join(top_obs)
            # axes_negative_weight_contributors[i].set_title(f"top: {top_obs}")
            avg_abs_grad = np.mean(np.abs(all_contributors), axis=0)
            print("expert yaw importance:")
            print(avg_abs_grad[560:562])
            print(avg_abs_grad[560:562].sum())
            print("phys latent importance:")
            print(avg_abs_grad[571:591])
            print(avg_abs_grad[571:591].sum())
            print("proprio importance:")
            print(avg_abs_grad[0:48])
            print(avg_abs_grad[0:48].sum())
            axes_all_grads[plot_idx].bar(np.arange(all_contributors.shape[1]), avg_abs_grad, color="k")
            axes_all_grads[plot_idx].set_ylabel(f"{str(i).zfill(1)}")
            plot_idx += 1



        fig_positive_weight_contributors.suptitle(f"Count of Most Important Indices of Observation for Positive Gradient on Expert Weight (top {top_n_for_timestep} gradients, across timesteps)")
        fig_positive_weight_contributors.savefig(os.path.join(out_dir, f"{prefix}_moe_weights_grad_obs_positive.png"))


        fig_positive_negative_contributors.suptitle(f"Count of Most Important Indices of Observation for Positive Gradient on Expert Weight (top {top_n_for_timestep} gradients, across timesteps)")
        fig_positive_negative_contributors.savefig(os.path.join(out_dir, f"{prefix}_moe_weights_grad_obs_negative.png"))

        axes_all_weight_contributors[0].set_title(f"Count of Obs. Indices w/ Biggest Abs. Gradient on Expert Weight")
        axes_all_weight_contributors[-1].set_xlabel("Observation Index")

        axes_all_grads[0].set_title(f"Avg. Absolute Gradient of Selected Expert Weights w.r.t. Obs.")
        axes_all_grads[-1].set_xlabel("Observation Index")

        alpha = 0.3
        for i in range(n_experts):

            """
obs: 
ang vel 0:3
orient 3:5
command 5:8
dof_diff_vs_def 8:20
dof_vel 20:32
action_past 32:44
foot: 44:48

depth encoder: 48:80
obs history: 80: 560 from (48 * 10)
yaw_est: 560:562
priv explicit: 562: 571 (9)
priv latent: 571: 591 (20)
            """

            for ax in [axes_all_weight_contributors, axes_all_grads]:
                ylim = ax[0].get_ylim()[1]
                ax[i].fill_between(np.arange(0, 48 + 1), 0.0, ylim, facecolor='red', alpha=alpha)
                ax[i].fill_between(np.arange(48, 80 + 1), 0.0, ylim, facecolor='orange', alpha=alpha)
                ax[i].fill_between(np.arange(80, 560 + 1), 0.0, ylim, facecolor='yellow', alpha=alpha)
                ax[i].fill_between(np.arange(560, 562 + 1), 0.0, ylim, facecolor='green', alpha=alpha)
                ax[i].fill_between(np.arange(562, 571 + 1), 0.0, ylim, facecolor='blue', alpha=alpha)
                ax[i].fill_between(np.arange(571, 591), 0.0, ylim, facecolor='purple', alpha=alpha)

        fig_all_contributors.supylabel('Expert Index')
        fig_all_contributors.savefig(os.path.join(out_dir, f"{prefix}_moe_weights_grad_obs_all.pdf"))
        fig_all_contributors.savefig(os.path.join(out_dir, f"{prefix}_moe_weights_grad_obs_all.png"))
        fig_all_grads.supylabel('Expert Index, Average Absolute Gradient')
        fig_all_grads.savefig(os.path.join(out_dir, f"{prefix}_moe_weights_grad_obs_all.pdf"))
        fig_all_grads.savefig(os.path.join(out_dir, f"{prefix}_moe_weights_grad_obs_all.png"))

        print(f"min expert_utilization_pct: {np.min(expert_utilization_pct)}, max expert_utilization_pct: {np.max(expert_utilization_pct)}")
        print(f"expert_utilization_pct: {expert_utilization_pct}")



    def get_weight_jacobian(self, obs):
        
        def get_weights(obs):

            int_out = self.our_agent.net_to_moe_layer(obs)

            h_x = int_out @ self.our_agent.gate_mat
            ret = torch.topk(h_x, self.our_agent.top_k, dim=1)

            top_k_gates = self.our_agent.soft_max(ret.values)
            zeros = torch.zeros_like(h_x)
            gates = zeros.scatter(1, ret.indices, top_k_gates)
            return gates

        new_obs = obs.clone().detach()
        new_obs.requires_grad = True

        jac = torch.autograd.functional.jacobian(get_weights, new_obs, create_graph=False)
        jac = jac[0][:, 0, :]

        return jac

    def setup(self):
        self.p_gains = [STIFFNESS] * self.num_dofs
        self.d_gains = [DAMPING] * self.num_dofs

        self.p_gains = torch.tensor(
            self.p_gains, device=self.device, dtype=torch.float32
        )
        self.d_gains = torch.tensor(
            self.d_gains, device=self.device, dtype=torch.float32
        )

        self.default_dof_pos = torch.zeros(
            self.num_dofs, device=self.device, dtype=torch.float32
        )
        self.dof_pos_ = torch.empty(
            1, self.num_dofs, device=self.device, dtype=torch.float32
        )
        self.dof_vel_ = torch.empty(
            1, self.num_dofs, device=self.device, dtype=torch.float32
        )
        self.tau_est_ = torch.empty(
            1, self.num_dofs, device=self.device, dtype=torch.float32
        )

        for i in range(self.num_dofs):
            name = self.dof_names[i]
            default_joint_angle = DEFAULT_JOINT_ANGLES[name]
            # in simulation order.
            self.default_dof_pos[i] = default_joint_angle

        self.computer_clip_torque = True
        self.torque_limits = TORQUE_LIMITS.to(self.device)

        # actions
        self.num_actions = NUM_ACTIONS
        self.action_scale = ACTION_SCALE
        self.clip_actions = 10.0
        self.clip_actions_method = "hard"

        # hardware related, in simulation order
        self.joint_limits_high = JOINT_LIMITS_HIGH.to(self.device)
        self.joint_limits_low = JOINT_LIMITS_LOW.to(self.device)

        self.clip_actions_low = (
            (self.joint_limits_low - self.default_dof_pos) * 1 / self.action_scale
        )
        self.clip_actions_high = (
            (self.joint_limits_high - self.default_dof_pos) * 1 / self.action_scale
        )

        self.actions = torch.zeros(
            self.num_actions, device=self.device, dtype=torch.float32
        )

        joint_pos_mid = (self.joint_limits_high + self.joint_limits_low) / 2
        joint_pos_range = (self.joint_limits_high - self.joint_limits_low) / 2
        self.dof_pos_protect_ratio = 1.1
        self.joint_pos_protect_high = (
            joint_pos_mid + joint_pos_range * self.dof_pos_protect_ratio
        )
        self.joint_pos_protect_low = (
            joint_pos_mid - joint_pos_range * self.dof_pos_protect_ratio
        )

    def clip_by_torque_limit(self, actions_scaled):
        """Different from simulation, we reverse the process and clip the actions directly,
        so that the PD controller runs in robot but not our script.
        """
        p_limits_low = (-self.torque_limits) + self.d_gains * self.dof_vel_
        p_limits_high = (self.torque_limits) + self.d_gains * self.dof_vel_
        actions_low = (
            (p_limits_low / self.p_gains) - self.default_dof_pos + self.dof_pos_
        )
        actions_high = (
            (p_limits_high / self.p_gains) - self.default_dof_pos + self.dof_pos_
        )
        return torch.clip(actions_scaled, actions_low, actions_high)

    def clip_action_before_scale(self, action):
        action = torch.clip(action, -self.clip_actions, self.clip_actions)
        action = torch.clip(action, self.clip_actions_low, self.clip_actions_high)
        return action

    def do_action_limits_and_scale(self, actions):
        actions = self.clip_action_before_scale(actions)
        clipped_scaled_action = self.clip_by_torque_limit(actions * self.action_scale)
        return clipped_scaled_action

    def reindex(self, vec):
        return vec[:, DOF_MAP_TO_SIM]

    def reindex_feet(self, vec):
        return vec[:, FOOT_MAP_TO_SIM]

    def obs_latent_to_act(self, obs, depth_latent):

        with torch.no_grad():
            if self.actor_type == ActorType.MIX_OF_EXPERTS:
                # dists, means, modes = self.our_agent.forward_fast(obs)
                dists, means, modes = self.our_agent(obs)
                actions = means
            elif self.actor_type == ActorType.POLICY_PER_SKILL:
                env_types = torch.zeros(
                    (obs.shape[0], 1), dtype=torch.float32, device=self.device
                )
                env_types[:, 0] = 18
                dists, means, modes = self.our_agent(obs, env_types)
                actions = means
            else:
                dists, means, modes = self.our_agent(obs)
            actions = means

        clip_actions = ENV_DICT["normalization"]["clip_actions"] / self.action_scale
        actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.action_buf[self.action_buf_len - 1, :] = actions.flatten()
        return actions

    @torch.no_grad()
    def arrs_to_obs(
        self,
        base_ang_vel,
        rpy,
        pos,
        vel,
        tau,
        foot_force,
        depth_buf,
        contact_thresh: float = CONTACT_THRESHOLD,
    ):

        with torch.inference_mode():

            norm_ang_vel = base_ang_vel * self.obs_scales["ang_vel"]

            roll = rpy[0]
            pitch = rpy[1]

            # we do not use yaw or x/y for target vector except when we train for priviledged data in phase 1
            # later we estimate it using a pre-trained encoder
            # yaw = msg.imu_state.rpy[2]
            # x, y = 0.0, 0.0
            self.delta_yaw = 0.0
            self.prior_delta_yaw = 0.0
            self.prior_next_delta_yaw = 0.0

            # target_vec_norm, target_yaw, reached_goal = self.update_goals(x, y, self.cur_goal_idx)
            # next_target_vec_norm, next_target_yaw, next_reached_goal = self.update_goals(x, y, self.cur_goal_idx + 1)

            # if self.global_counter % 5 == 0 or self.prior_delta_yaw is None:
            #     self.prior_delta_yaw = target_yaw- yaw
            #     self.prior_next_delta_yaw = next_target_yaw - yaw

            yaw = 0.0

            env_class = 17 

            env_class_neq_17 = float(env_class != 17)
            env_class_eq_17 = float(env_class == 17)
            # print(f"pos noisy rl: {pos}")
            # print(f"default_dof_pos_all rl: {self.default_dof_pos}")

            for sim_idx in range(self.num_dofs):
                self.dof_pos_[0, sim_idx] = torch.tensor(
                    pos[sim_idx], dtype=torch.float32, device=self.device
                )
                self.dof_vel_[0, sim_idx] = torch.tensor(
                    vel[sim_idx], dtype=torch.float32, device=self.device
                )
                self.tau_est_[0, sim_idx] = torch.tensor(
                    tau[sim_idx], dtype=torch.float32, device=self.device
                )

            dof_diff_vs_default = (
                self.dof_pos_ - self.default_dof_pos
            ) * self.obs_scales["dof_pos"]

            # print(f"dof diff vs default, scaled, rl: {dof_diff_vs_default}")

            dof_vel = self.dof_vel_ * self.obs_scales["dof_vel"]
            action_buf_obs = self.action_buf[-1, :].reshape(1, -1)

            foot_contact = foot_force > contact_thresh

            new_foot_contact = np.logical_or(self.last_contacts, foot_contact)

            self.last_contacts = foot_contact

            foot_contact_obs = new_foot_contact.astype(np.float32) - 0.5

            obs = torch.tensor(
                (
                    *norm_ang_vel,
                    roll,
                    pitch,
                    *self.commands[0:3],
                    *(dof_diff_vs_default.flatten().cpu().detach().numpy()),
                    *(dof_vel.flatten().cpu().detach().numpy()),
                    *(action_buf_obs.flatten().cpu().detach().numpy()),
                    *foot_contact_obs,
                ),
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(0)

            if USE_CAMERA:
                obs_for_depth_encoder = obs[:, : self.n_proprio].clone()
                if self.past_obs_for_depth_encoder > 0:
                    relevant_hist = self.obs_history_buf[
                        0, -self.past_obs_for_depth_encoder :, :
                    ]
                    obs_for_depth_encoder = torch.cat(
                        (obs_for_depth_encoder, relevant_hist.view(1, -1)), dim=1
                    )
                depth_latent, yaw_est = self.depth_encoder(
                    depth_buf.unsqueeze(0).clone(), obs_for_depth_encoder
                )
            else:
                depth_latent, yaw_est = (None, None)


            if USE_CAMERA:
                self.obs_buf = torch.cat(
                    (
                        obs[:, : self.n_proprio],
                        depth_latent,
                        self.obs_history_buf.view(1, -1),
                        yaw_est,
                    ),
                    dim=1,
                )
                if DO_ENCODING_TRICKS:
                    priv_explicit = self.estimator(
                        self.obs_buf[:, : self.n_proprio]
                    )
                    priv_latent = self.history_encoder(self.obs_history_buf.clone())
                    self.obs_buf = torch.cat(
                        (self.obs_buf, priv_explicit, priv_latent), dim=1
                    )
            else:
                raise NotImplementedError

            # we want to init obs history when we have action, so not right after a reset but rather after we
            # do one action. The env for sim does this behavior.
            if self.initial_obs or (
                self.init_obs_history and torch.all(action_buf_obs != 0)
            ):
                self.obs_history_buf[0] = torch.stack(
                    [obs.squeeze(0)] * ENV_DICT["env"]["history_len"], dim=0
                )
                self.initial_obs = False
                self.init_obs_history = False
            else:
                self.obs_history_buf[0] = torch.cat(
                    [self.obs_history_buf[0, 1:], obs], dim=0
                )

            self.global_counter += 1

            # print(f"took: {time.time() - start_time}")

        return (
            torch.clip(
                self.obs_buf,
                -ENV_DICT["normalization"]["clip_observations"],
                ENV_DICT["normalization"]["clip_observations"],
            ),
            depth_latent,
        )

    def _add_depth_contour(self, depth_images):
        gradients = F.max_pool2d(
            torch.abs(
                F.conv2d(depth_images, self.contour_detection_kernel, padding=1)
            ).max(dim=-3, keepdim=True)[0],
            kernel_size=ENV_DICT["depth"]["contour_detection_kernel_size"],
            stride=1,
            padding=int(ENV_DICT["depth"]["contour_detection_kernel_size"] / 2),
        )
        # print(f"max gradient contour: {torch.max(gradients)}")
        # print(f"min gradient contour: {torch.min(gradients)}")
        # print(f"mean gradient contour: {torch.mean(gradients)}")
        # print(f"90 quantile: {torch.quantile(gradients, 0.9)}")
        mask = gradients > ENV_DICT["depth"]["contour_threshold"]
        rand_flts = torch.rand(depth_images[mask].shape, device=self.device)
        depth_images[mask] = torch.where(
            rand_flts < ENV_DICT["depth"]["contour_nuke_prob"],
            ENV_DICT["depth"]["far_clip"],
            depth_images[mask],
        )
        return depth_images

    @torch.no_grad()
    def form_artifacts(
        self,
        H,
        W,  # image resolution
        tops,
        bottoms,  # artifacts positions (in pixel) shape (n_,)
        lefts,
        rights,
    ):
        """Paste an artifact to the depth image.
        NOTE: Using the paradigm of spatial transformer network to build the artifacts of the
        entire depth image.
        """
        batch_size = tops.shape[0]
        tops, bottoms = tops[:, None, None], bottoms[:, None, None]
        lefts, rights = lefts[:, None, None], rights[:, None, None]

        # build the source patch
        source_patch = torch.zeros((batch_size, 1, 25, 25), device=self.device)
        source_patch[:, :, 1:24, 1:24] = 1.0

        # build the grid
        grid = torch.zeros((batch_size, H, W, 2), device=self.device)
        grid[..., 0] = torch.linspace(-1, 1, W, device=self.device).view(1, 1, W)
        grid[..., 1] = torch.linspace(-1, 1, H, device=self.device).view(1, H, 1)
        grid[..., 0] = (grid[..., 0] * W + W - rights - lefts) / (rights - lefts)
        grid[..., 1] = (grid[..., 1] * H + H - bottoms - tops) / (bottoms - tops)

        # sample using the grid and form the artifacts for the entire depth image
        artifacts = torch.clip(
            F.grid_sample(
                source_patch,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            .sum(dim=0)
            .view(H, W),
            0,
            1,
        )

        return artifacts.bool()

    def _add_depth_artifacts(
        self,
        depth_image,
        artifacts_prob,
        artifacts_height_mean_std,
        artifacts_width_mean_std,
    ):

        # artifacts_height_mean_std is (usual height, std dev of height)
        # artifacts_width_mean_std is (usual_width, std dev of width)

        h, w = depth_image.shape

        def _clip(x, dim):
            return torch.clip(x, 0.0, (h, w)[dim])

        # random patched artifacts
        artifacts_mask = (
            torch.rand(
                (h, w),
                device=self.device,
            )
            < artifacts_prob
        )

        artifacts_mask = (
            torch.rand(
                (ENV_DICT["depth"]["original"][0], ENV_DICT["depth"]["original"][1]),
                device=self.device,
            )
            < artifacts_prob
        )

        artifacts_coord = torch.nonzero(artifacts_mask).to(torch.float32)
        artifacts_size = (
            torch.clip(
                artifacts_height_mean_std[0]
                + torch.randn(
                    (artifacts_coord.shape[0],),
                    device=self.device,
                )
                * artifacts_height_mean_std[1],
                0.0,
                h,
            ),
            torch.clip(
                artifacts_width_mean_std[0]
                + torch.randn(
                    (artifacts_coord.shape[0],),
                    device=self.device,
                )
                * artifacts_width_mean_std[1],
                0.0,
                w,
            ),
        )

        artifacts_top_left = (
            _clip(artifacts_coord[:, 0] - artifacts_size[0] / 2, 0),
            _clip(artifacts_coord[:, 1] - artifacts_size[1] / 2, 1),
        )

        artifacts_bottom_right = (
            _clip(artifacts_coord[:, 0] + artifacts_size[0] / 2, 0),
            _clip(artifacts_coord[:, 1] + artifacts_size[1] / 2, 1),
        )

        art_mask = self.form_artifacts(
            h,
            w,
            artifacts_top_left[0],
            artifacts_bottom_right[0],
            artifacts_top_left[1],
            artifacts_bottom_right[1],
        )
        depth_image[art_mask] = ENV_DICT["depth"]["far_clip"]
        return depth_image

    @torch.no_grad()
    def _process_depth_img(self, img):

        assert img.shape == (480, 640), f"expected img: (480, 640), got {img.shape}"
        depth_image = torch.tensor(img, device=self.device, dtype=torch.float32)
        # convert to meters
        depth_image *= 0.001
        depth_image = self.resize_transform_640to_4th(depth_image[None, :]).squeeze()

        depth_image += ENV_DICT["depth"]["dis_noise"] * torch.randn(
            depth_image.shape, device=self.device
        )

        # These operations are replicatd in sim
        # set everything below near clip to max
        depth_image = torch.where(
            depth_image < ENV_DICT["depth"]["near_clip"],
            ENV_DICT["depth"]["far_clip"],
            depth_image,
        )
        depth_image = torch.where(
            depth_image > ENV_DICT["depth"]["far_clip"],
            ENV_DICT["depth"]["far_clip"],
            depth_image,
        )
        # for real runs we skip noising it with depth contours
        # depth_image = self._add_depth_contour(depth_image.unsqueeze(0)).squeeze()
        depth_image = depth_image[
            : -ENV_DICT["depth"]["bottom_clip"],
            ENV_DICT["depth"]["left_clip"] : -ENV_DICT["depth"]["right_clip"],
        ]

        # for real runs we skip noising it with depth artifacts
        # depth_image = self._add_depth_artifacts(
        #     depth_image,
        #     ENV_DICT["depth"]["artifact_prob"],
        #     ENV_DICT["depth"]["artifact_height_mean_std"],
        #     ENV_DICT["depth"]["artifact_width_mean_std"],
        # )
        depth_image = self.gaussian_blur_transform(depth_image[None, :]).squeeze()

        depth_image = self.resize_transform_output_shape(depth_image[None, :]).squeeze()

        depth_image = (depth_image - ENV_DICT["depth"]["near_clip"]) / (
            ENV_DICT["depth"]["far_clip"] - ENV_DICT["depth"]["near_clip"]
        ) - 0.5
        return depth_image
