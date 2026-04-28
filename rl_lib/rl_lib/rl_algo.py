from collections import deque
from copy import deepcopy
import itertools
import statistics
import time
import os

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import wandb

from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import update_cfg_from_args, class_to_dict

from rl_lib.replay_buffer import ReplayBuffer, DatasetSpec, StorageBuffer
from rl_lib.agent import (
    ACAgent,
    get_sequential_model,
    ScanEncoder,
    DepthOnlyFCBackbone58x87,
    RecurrentDepthBackbone,
    ActorType,
    Actor,
    StateHistoryEncoder,
)
from rl_lib.utils import to_numpy, batch_to_torch
from rl_lib.video_recorder import VideoRecorder

# torch.autograd.set_detect_anomaly(True)


class AlgoRunner:
    def __init__(
        self,
        log_dir,
        env,
        args,
        env_name,
        device,
        actor_type: ActorType,
        enable_vids: bool = True,
    ):
        self.log_dir = log_dir
        self.env = env
        self.args = args
        self.env_name = env_name
        self.device = device
        self.actor_type = actor_type

        self.num_envs = self.env.num_envs
        self.num_actions = self.env.num_actions

        env_cfg, train_cfg = task_registry.get_cfgs(self.env_name)
        env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)

        train_cfg_dict = class_to_dict(train_cfg)
        env_cfg_dict = class_to_dict(env_cfg)

        print("env cfg")
        print(env_cfg_dict)

        self.moe_n_experts = train_cfg_dict["algorithm"]["moe_n_experts"]
        self.moe_top_k = train_cfg_dict["algorithm"]["moe_top_k"]
        self.moe_loss_coeff = train_cfg_dict["algorithm"]["moe_loss_coeff"]
        self.moe_noise_mat_init = train_cfg_dict["algorithm"]["moe_noise_mat_init"]
        self.gate_noise_with_x = train_cfg_dict["algorithm"]["gate_noise_with_x"]
        self.moe_layer_idx = train_cfg_dict["algorithm"]["moe_layer_idx"]

        self.iterations_step_with_teacher_before_student = env_cfg_dict["depth"][
            "iterations_step_with_teacher_before_student"
        ]

        self.delta_yaw_thresh = train_cfg_dict["algorithm"]["delta_yaw_thresh"]

        # {'asset': {'angular_damping': 0.0, 'armature': 0.0, 'collapse_fixed_joints': True, 'default_dof_drive_mode': 3, 'density': 0.001, 'disable_gravity': False,
        #  'file': '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf', 'fix_base_link': False, 'flip_visual_attachments': True,
        #   'foot_name': 'foot', 'linear_damping': 0.0, 'max_angular_velocity': 1000.0, 'max_linear_velocity': 1000.0,
        #   'penalize_contacts_on': ['thigh', 'calf', 'base'], 'replace_cylinder_with_capsule': True, 'self_collisions': 1,
        #   'terminate_after_contacts_on': ['base'], 'thickness': 0.01}, 'commands': {'ang_vel_clip': 0.4,
        #   'crclm_incremnt': {'ang_vel_yaw': 0.1, 'heading': 0.5, 'lin_vel_x': 0.1, 'lin_vel_y': 0.1}, 'curriculum': False, 'heading_command': True,
        #    'lin_vel_clip': 0.2, 'max_curriculum': 1.0, 'max_ranges': {'ang_vel_yaw': [0, 0], 'heading': [-1.6, 1.6], 'lin_vel_x': [0.3, 0.8],
        #     'lin_vel_y': [-0.3, 0.3]}, 'num_commands': 4, 'ranges': {'ang_vel_yaw': [0, 0], 'heading': [0, 0], 'lin_vel_x': [0.0, 1.5],
        #      'lin_vel_y': [0.0, 0.0]}, 'resampling_time': 6.0, 'waypoint_delta': 0.7}, 'control': {'action_scale': 0.25, 'control_type': 'P',
        #      'damping': {'joint': 1}, 'decimation': 4, 'stiffness': {'joint': 40.0}}, 'depth': {'angle': [-5, 5], 'buffer_len': 2, 'camera_num_envs': 192,
        #       'camera_terrain_num_cols': 20, 'camera_terrain_num_rows': 10, 'dis_noise': 0.0, 'far_clip': 2, 'horizontal_fov': 87, 'invert': True,
        #        'near_clip': 0, 'original': (106, 60), 'position': [0.27, 0, 0.03], 'resized': (87, 58), 'scale': 1, 'update_interval': 5, 'use_camera': False},
        #         'domain_rand': {'action_buf_len': 8, 'action_curr_step': [0, 1], 'action_curr_step_scratch': [0, 1], 'action_delay': True, 'action_delay_view': 1,
        #          'added_com_range': [-0.2, 0.2], 'added_mass_range': [0.0, 3.0], 'delay_update_global_steps': 192000, 'friction_range': [0.6, 2.0],
        #          'max_push_vel_xy': 0.5, 'motor_strength_range': [0.8, 1.2], 'push_interval': 401.0, 'push_interval_s': 8, 'push_robots': True,
        #          'randomize_base_com': True, 'randomize_base_mass': True, 'randomize_friction': True, 'randomize_motor': True}, 'env': {'contact_buf_len': 100,
        #           'env_spacing': 3.0, 'episode_length_s': 20, 'history_encoding': True, 'history_len': 10, 'include_foot_contacts': True, 'n_priv': 9,
        #            'n_priv_latent': 29, 'n_proprio': 53, 'n_scan': 132, 'next_goal_threshold': 0.2, 'num_actions': 12, 'num_envs': 8, 'num_future_goal_obs': 2,
        #             'num_observations': 753, 'num_privileged_obs': None, 'obs_type': 'og', 'rand_pitch_range': 1.6, 'rand_y_range': 0.5,
        #              'rand_yaw_range': 1.2, 'randomize_start_pitch': False, 'randomize_start_pos': False, 'randomize_start_vel': False, 'randomize_start_y': False,
        #              'randomize_start_yaw': False, 'reach_goal_delay': 0.1, 'reorder_dofs': True, 'send_timeouts': True}, 'init_member_classes': {},
        #              'init_state': {'ang_vel': [0.0, 0.0, 0.0], 'default_joint_angles': {'FL_hip_joint': 0.1, 'RL_hip_joint': 0.1, 'FR_hip_joint': -0.1,
        #               'RR_hip_joint': -0.1, 'FL_thigh_joint': 0.8, 'RL_thigh_joint': 1.0, 'FR_thigh_joint': 0.8, 'RR_thigh_joint': 1.0, 'FL_calf_joint': -1.5,
        #                'RL_calf_joint': -1.5, 'FR_calf_joint': -1.5, 'RR_calf_joint': -1.5}, 'lin_vel': [0.0, 0.0, 0.0], 'pos': [0.0, 0.0, 0.42],
        #                'rot': [0.0, 0.0, 0.0, 1.0]}, 'noise': {'add_noise': False, 'noise_level': 1.0, 'noise_scales': {'ang_vel': 0.05, 'dof_pos': 0.01,
        #                'dof_vel': 0.05, 'gravity': 0.02, 'height_measurements': 0.02, 'lin_vel': 0.05, 'rotation': 0.0}, 'quantize_height': True},
        #                'normalization': {'clip_actions': 1.2, 'clip_observations': 100.0, 'obs_scales': {'ang_vel': 0.25, 'dof_pos': 1.0, 'dof_vel': 0.05,
        #                 'height_measurements': 5.0, 'lin_vel': 2.0}}, 'play': {'load_student_config': False, 'mask_priv_obs': False},
        #                 'rewards': {'base_height_target': 0.25, 'max_contact_force': 40.0, 'only_positive_rewards': True,
        #                  'scales': {'action_rate': -0.1, 'ang_vel_xy': -0.05, 'collision': -10.0, 'delta_torques': -1e-07,
        #                   'dof_acc': -2.5e-07, 'dof_error': -0.04, 'feet_edge': -1, 'feet_stumble': -1, 'hip_pos': -0.5,
        #                   'lin_vel_z': -1.0, 'orientation': -1.0, 'torques': -1e-05, 'tracking_goal_vel': 1.5,
        #                    'tracking_yaw': 0.5}, 'soft_dof_pos_limit': 0.9, 'soft_dof_vel_limit': 1, 'soft_torque_limit': 0.4,
        #                     'tracking_sigma': 0.2}, 'seed': 1, 'sim': {'dt': 0.005, 'gravity': [0.0, 0.0, -9.81],
        #                     'physx': {'bounce_threshold_velocity': 0.5, 'contact_collection': 2, 'contact_offset': 0.01,
        #                     'default_buffer_size_multiplier': 5, 'max_depenetration_velocity': 1.0, 'max_gpu_contact_pairs': 8388608,
        #                     'num_position_iterations': 4, 'num_threads': 10, 'num_velocity_iterations': 0, 'rest_offset': 0.0, 'solver_type': 1},
        #                     'substeps': 1, 'up_axis': 1}, 'terrain': {'all_vertical': False, 'border_size': 5, 'curriculum': True, 'downsampled_scale': 0.075,
        #                      'dynamic_friction': 1.0, 'edge_width_thresh': 0.05, 'gap_size': [0.02, 0.1], 'height': [0.02, 0.06],
        #                       'hf2mesh_method': 'grid', 'horizontal_scale': 0.05, 'horizontal_scale_camera': 0.1, 'max_error': 0.1, 'max_error_camera': 2,
        #                        'max_init_terrain_level': 5, 'measure_heights': True, 'measure_horizontal_noise': 0.0,
        #                         'measured_points_x': [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2],
        #                          'measured_points_y': [-0.75, -0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75],
        #                           'mesh_type': 'trimesh', 'no_flat': True, 'num_cols': 40, 'num_goals': 8, 'num_rows': 10,
        #                            'num_sub_terrains': 400, 'origin_zero_z': True, 'restitution': 0.0, 'selected': False,
        #                            'simplify_grid': False, 'slope_treshold': 1.5, 'static_friction': 1.0, 'stepping_stone_distance': [0.02, 0.08],
        #                             'terrain_dict': {'smooth slope': 0.0, 'rough slope up': 0.0, 'rough slope down': 0.0, 'rough stairs up': 0.0,
        #                              'rough stairs down': 0.0, 'discrete': 0.0, 'stepping stones': 0.0, 'gaps': 0.0, 'smooth flat': 0,
        #                               'pit': 0.0, 'wall': 0.0, 'platform': 0.0, 'large stairs up': 0.0, 'large stairs down': 0.0, 'parkour': 0.2,
        #                                'parkour_hurdle': 0.2, 'parkour_flat': 0.2, 'parkour_step': 0.2, 'parkour_gap': 0.2, 'demo': 0.0}, 'terrain_kwargs': None,
        #                                 'terrain_length': 18.0, 'terrain_proportions': array([0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        #                                 0. , 0.2, 0.2, 0.2, 0.2, 0.2, 0. ]), 'terrain_width': 4, 'vertical_scale': 0.005, 'y_range': [-0.4, 0.4]},
        #                                  'viewer': {'lookat': [11.0, 5, 3.0], 'pos': [10, 0, 6], 'ref_env': 0}}

        print("train cfg")
        print(train_cfg_dict)

        # {'algorithm': {'clip_param': 0.2, 'dagger_update_freq': 20, 'desired_kl': 0.01, 'entropy_coef': 0.01, 'gamma': 0.99, 'lam': 0.95,
        # 'learning_rate': 0.0002, 'max_grad_norm': 1.0, 'num_learning_epochs': 5, 'num_mini_batches': 4,
        # 'priv_reg_coef_schedual': [0, 0.1, 2000, 3000], 'priv_reg_coef_schedual_resume': [0, 0.1, 0, 1], 'schedule': 'adaptive',
        # 'use_clipped_value_loss': True, 'value_loss_coef': 1.0}, 'depth_encoder': {'buffer_len': 2, 'depth_shape': (87, 58),
        # 'hidden_dims': 512, 'if_depth': False, 'learning_rate': 0.001, 'num_steps_per_env': 120}, 'estimator': {'hidden_dims': [128, 64],
        # 'learning_rate': 0.0001, 'num_prop': 53, 'num_scan': 132, 'priv_states_dim': 9, 'train_with_estimated_states': True}, 'init_member_classes': {},
        #  'policy': {'activation': 'elu', 'actor_hidden_dims': [512, 256, 128], 'continue_from_last_std': True, 'critic_hidden_dims': [512, 256, 128],
        #  'init_noise_std': 1.0, 'priv_encoder_dims': [64, 20], 'rnn_hidden_size': 512, 'rnn_num_layers': 1, 'rnn_type': 'lstm',
        #  'scan_encoder_dims': [128, 64, 32], 'tanh_encoder_output': False}, 'runner': {'algorithm_class_name': 'PPO', 'checkpoint': -1,
        #  'experiment_name': 'rough_a1', 'load_run': -1, 'max_iterations': 15000, 'num_steps_per_env': 24, 'policy_class_name': 'ActorCritic',
        #  'resume': False, 'resume_path': None, 'run_name': '', 'save_interval': 100}, 'runner_class_name': 'OnPolicyRunner', 'seed': 1}

        phase1 = True

        activation = train_cfg_dict["policy"]["activation"]

        if activation == "elu":
            activation = nn.ELU
        elif activation == "relu":
            activation = nn.ReLU
        elif activation == "selu":
            activation = nn.SELU
        elif activation == "crelu":
            activation = nn.crelu
        elif activation == "lrelu":
            activation = nn.LeakyReLU
        elif activation == "tanh":
            activation = nn.Tanh
        elif activation == "sigmoid":
            actiation = nn.Sigmoid
        else:
            raise NotImplementedError

        self.max_grad_norm = train_cfg_dict["algorithm"]["max_grad_norm"]

        self.use_clipped_value_loss = train_cfg_dict["algorithm"][
            "use_clipped_value_loss"
        ]
        self.value_loss_coef = train_cfg_dict["algorithm"]["value_loss_coef"]
        self.clip_param = train_cfg_dict["algorithm"]["clip_param"]
        self.desired_kl = train_cfg_dict["algorithm"]["desired_kl"]
        self.schedule = train_cfg_dict["algorithm"]["schedule"]

        self.num_learning_epochs = train_cfg_dict["algorithm"]["num_learning_epochs"]
        self.num_mini_batches = train_cfg_dict["algorithm"]["num_mini_batches"]

        self.gamma = train_cfg_dict["algorithm"]["gamma"]
        self.lam = train_cfg_dict["algorithm"]["lam"]

        self.num_steps_per_env = train_cfg_dict["runner"]["num_steps_per_env"]

        self.use_camera = env_cfg_dict["depth"]["use_camera"]
        if self.use_camera:
            self.num_steps_per_env = train_cfg_dict["depth_encoder"][
                "num_steps_per_env"
            ]

        self.batch_size = self.num_envs * self.num_steps_per_env
        self.mini_batch_size = self.batch_size // self.num_mini_batches

        self.dagger_update_freq = train_cfg_dict["algorithm"]["dagger_update_freq"]

        self.save_interval = train_cfg_dict["runner"]["save_interval"]
        self.learning_rate = train_cfg_dict["algorithm"]["learning_rate"]
        self.estimator_lr = train_cfg_dict["estimator"]["learning_rate"]

        self.clip_param = train_cfg_dict["algorithm"]["clip_param"]

        self.priv_states_dim = train_cfg_dict["estimator"]["priv_states_dim"]
        self.n_priv_latent = env_cfg_dict["env"]["n_priv_latent"]

        self.do_encoding_tricks = train_cfg_dict["algorithm"]["do_encoding_tricks"]

        if self.do_encoding_tricks:
            priv_encoder_dims = train_cfg_dict["policy"]["priv_encoder_dims"]
            priv_encoder_output_dim = priv_encoder_dims[-1]
            priv_encoder_dims = priv_encoder_dims[:-1]
            self.priv_encoder = get_sequential_model(
                self.n_priv_latent,
                priv_encoder_output_dim,
                priv_encoder_dims,
                activation,
                True,
            )
            self.priv_encoder = self.priv_encoder.to(self.device)

        self.entropy_coef = train_cfg_dict["algorithm"]["entropy_coef"]

        # Adaptation
        self.priv_reg_coef_schedual = train_cfg_dict["algorithm"][
            "priv_reg_coef_schedual"
        ]
        self.counter = 0

        self.history_len = env_cfg_dict["env"]["history_len"]
        self.n_prop = env_cfg_dict["env"]["n_proprio"]

        if self.do_encoding_tricks:
            self.history_encoder = StateHistoryEncoder(
                activation, self.n_prop, self.history_len, priv_encoder_output_dim
            )
            self.history_encoder = self.history_encoder.to(self.device)

            self.hist_encoder_optimizer = torch.optim.Adam(
                self.history_encoder.parameters(), lr=self.learning_rate
            )

        self.n_priv_explicit = env_cfg_dict["env"]["n_priv"]
        # actor_input = n_prop + scan_output_dim + self.n_priv_explicit + priv_encoder_output_dim
        actor_hidden_dims = train_cfg_dict["policy"]["actor_hidden_dims"]
        actor_output = self.num_actions

        critic_hidden_dims = train_cfg_dict["policy"]["critic_hidden_dims"]

        self.n_scan_input = env_cfg_dict["env"]["n_scan"]
        self.scan_output_size = train_cfg_dict["policy"]["scan_encoder_dims"][-1]
        self.scan_hidden_dims = train_cfg_dict["policy"]["scan_encoder_dims"][:-1]
        self.encode_scan = self.scan_hidden_dims is not None and self.n_scan_input > 0
        self.scan_encoder = ScanEncoder(
            self.n_scan_input,
            self.scan_output_size,
            self.scan_hidden_dims,
            nn.Tanh,
            device=self.device,
        )

        num_priv_yaw = 2
        self.num_obs = (
            self.env.cfg.env.n_proprio
            + self.env.cfg.env.history_len * self.env.cfg.env.n_proprio
            + self.n_scan_input
        )

        if self.do_encoding_tricks:
            self.actor_input = (
                self.env.cfg.env.n_proprio
                + self.env.cfg.env.history_len * self.env.cfg.env.n_proprio
                + self.scan_output_size
                + num_priv_yaw
                + priv_encoder_output_dim
                + self.n_priv_explicit
            )
        else:
            self.actor_input = (
                self.env.cfg.env.n_proprio
                + self.env.cfg.env.history_len * self.env.cfg.env.n_proprio
                + self.scan_output_size
                + num_priv_yaw
            )

        if self.env.cfg.env.num_privileged_obs is not None:
            if self.do_encoding_tricks:
                self.num_obs_priv = (
                    self.env.cfg.env.n_proprio_priv
                    + self.env.cfg.env.history_len * self.env.cfg.env.n_proprio_priv
                    + self.n_scan_input
                )
            else:
                # take out com, friction, motor_strengths...
                self.n_com_critic = 4
                self.n_friction_critic = 1
                self.n_motor_strengths_critic = 24
                new_obs_priv_num = self.env.cfg.env.n_proprio_priv - 4 - 1 - 24
                self.num_obs_priv = (
                    new_obs_priv_num
                    + self.env.cfg.env.history_len * new_obs_priv_num
                    + self.n_scan_input
                )
            self.critic_obs = self.num_obs_priv
        else:
            raise NotImplementedError  # self.critic_obs = self.env.cfg.env.n_proprio + self.env.cfg.env.history_len * self.env.cfg.env.n_proprio

        env_classes = self.env.env_class.detach().cpu().numpy()

        self.unique_env_classes, self.unique_env_indices = np.unique(
            env_classes, return_index=True
        )

        print(f"unique env classes: {self.unique_env_classes}")

        self.logstd_init = train_cfg_dict["algorithm"]["logstd_init"]
        self.use_many_params_std = train_cfg_dict["algorithm"]["use_many_params_std"]

        self.dropout = train_cfg_dict["algorithm"]["dropout"]
        self.dropout_prob = train_cfg_dict["algorithm"]["dropout_prob"]

        self.ac_agent = ACAgent(
            self.actor_input,
            self.num_actions,
            self.critic_obs,
            self.device,
            actor_hidden_dims,
            critic_hidden_dims,
            activation,
            self.actor_type,
            self.moe_n_experts,
            self.moe_top_k,
            scan_idx=env_cfg_dict["env"]["n_proprio"],
            scan_len=self.scan_output_size,
            moe_noise_mat_init=self.moe_noise_mat_init,
            env_keys=self.unique_env_classes,
            logstd_init=self.logstd_init,
            use_many_params_std=self.use_many_params_std,
            gate_noise_with_x=self.gate_noise_with_x,
            moe_layer_idx=self.moe_layer_idx,
            dropout=self.dropout,
            dropout_prob=self.dropout_prob,
        )

        self.past_obs_for_depth_encoder = train_cfg_dict["algorithm"][
            "past_obs_for_depth_encoder"
        ]
        self.n_proprio = env_cfg_dict["env"]["n_proprio"]
        if self.use_camera:
            # 'hidden_dims': 512, 'if_depth': False, 'learning_rate': 0.001, 'num_steps_per_env': 120}
            self.depth_backbone = DepthOnlyFCBackbone58x87(
                env_cfg_dict["env"]["n_proprio"], self.scan_output_size
            )
            self.depth_encoder = RecurrentDepthBackbone(
                self.depth_backbone,
                env_cfg_dict["env"]["n_proprio"]
                + env_cfg_dict["env"]["n_proprio"] * self.past_obs_for_depth_encoder,
            ).to(self.device)

            if self.actor_type in [ActorType.SINGLE_POLICY, ActorType.MIX_OF_EXPERTS]:
                self.depth_actor = deepcopy(self.ac_agent.actor)
            elif self.actor_type == ActorType.POLICY_PER_SKILL:
                self.depth_actor = Actor(
                    self.actor_input,
                    self.num_actions,
                    actor_hidden_dims,
                    activation,
                    self.device,
                    logstd_init=self.logstd_init,
                )
            else:
                raise NotImplementedError

            self.depth_actor_optimizer = torch.optim.Adam(
                [*self.depth_actor.parameters(), *self.depth_encoder.parameters()],
                lr=train_cfg_dict["depth_encoder"]["learning_rate"],
            )

        if self.do_encoding_tricks:
            estimator_hidden_dim = train_cfg_dict["estimator"]["hidden_dims"]

            self.estimator = get_sequential_model(
                self.n_prop,
                self.n_priv_explicit,
                estimator_hidden_dim,
                activation,
                False,
            )
            self.estimator = self.estimator.to(self.device)
            self.estimator_optimizer = torch.optim.Adam(
                self.estimator.parameters(), lr=self.estimator_lr
            )

        self.num_privileged_obs = env_cfg_dict["env"]["num_privileged_obs"]

        self.critic_obs_key = "critic_observations"
        self.dones_key = "is_dones"
        self.values_key = "values"
        self.rewards_key = "rewards"
        self.bootstrapped_rewards_key = "adj_rewards"
        self.advantages_key = "advantages"

        keys_for_specs = [
            "observations",
            "obs_critic",
            self.values_key,
            self.advantages_key,
            "actions",
            self.rewards_key,
            self.dones_key,
            "old_action_log_probs",
            self.bootstrapped_rewards_key,
            "mus",
            "sigmas",
            "env_type",
        ]
        dtypes_for_specs = [
            torch.float32,
            torch.float32,
            torch.float32,
            torch.float32,
            torch.float32,
            torch.float32,
            torch.bool,
            torch.float32,
            torch.float32,
            torch.float32,
            torch.float32,
            torch.float32,
        ]
        shapes_for_specs = [
            (self.num_obs,),
            (self.critic_obs,),
            (1,),
            (1,),
            (self.num_actions,),
            (1,),
            (1,),
            (1,),
            (1,),
            (self.num_actions,),
            (self.num_actions,),
            (1,),
        ]

        # if self.use_camera:
        #     keys_for_specs += ["scandots_latent", "depth_latent"]
        #     dtypes_for_specs += [torch.float32, torch.float32]
        #     shapes_for_specs += [(self.scan_output_size,), (self.scan_output_size,)]

        self.data_specs = DatasetSpec(
            keys_for_specs, shapes_for_specs, dtypes_for_specs
        )
        self.discount_factor = 0.99

        # self.replay_buffer = ReplayBuffer(self.data_specs, max_size_replay_buffer,
        #                                   self.discount_factor, self.num_envs, self.num_steps_per_env)

        self.storage_buffer = StorageBuffer(
            self.data_specs, self.num_envs, self.num_steps_per_env, self.device
        )

        # still need depth encoder and such

        self.global_steps = 0
        self.tot_time = 0.0

        self.train_with_estimated_states = train_cfg_dict["estimator"][
            "train_with_estimated_states"
        ]
        self.num_learning_iterations = train_cfg_dict["runner"]["max_iterations"]

        self.starting_buffer_size = 5000

        # self.actor_opt = torch.optim.Adam(list(self.scan_encoder.parameters()) + list(self.priv_encoder.parameters()) + list(self.ac_agent.actor.parameters()), lr=self.learning_rate)
        if self.encode_scan:
            if self.do_encoding_tricks:
                self.actor_opt = torch.optim.Adam(
                    list(self.ac_agent.critic.parameters())
                    + list(self.ac_agent.actor.parameters())
                    + list(self.scan_encoder.parameters())
                    + list(self.priv_encoder.parameters()),
                    lr=self.learning_rate,
                )
            else:
                self.actor_opt = torch.optim.Adam(
                    list(self.ac_agent.critic.parameters())
                    + list(self.ac_agent.actor.parameters())
                    + list(self.scan_encoder.parameters()),
                    lr=self.learning_rate,
                )

        else:
            raise NotImplementedError

        # self.critic_opt = torch.optim.Adam(self.ac_agent.critic.parameters(), lr=self.learning_rate)
        # print("scan_encoder parameters")
        # [print(name, param.shape) for name, param in self.scan_encoder.named_parameters()]
        # print("priv_encoder parameters")
        # [print(name, param.shape) for name, param in self.priv_encoder.named_parameters()]
        # print("actor params")
        # [print(name, param.shape) for name, param in self.ac_agent.actor.named_parameters()]
        # print("critic params")
        # [print(name, param.shape) for name, param in self.ac_agent.critic.named_parameters()]

        # print("hist encoder params")
        # [print(name, param.shape) for name, param in self.history_encoder.named_parameters()]
        # print("estimator params")
        # [print(name, param.shape) for name, param in self.estimator.named_parameters()]
        self.enable_vids = train_cfg_dict["algorithm"]["enable_vids"]
        self.num_envs_to_record = train_cfg_dict["algorithm"]["num_envs_to_video"]
        self.record_vid_step_interval = train_cfg_dict["algorithm"]["vid_step_interval"]
        print("setting up vids")
        if self.enable_vids:

            num_envs = self.num_envs

            envs_to_record = []

            for j in range(len(self.unique_env_indices)):
                if len(envs_to_record) == self.num_envs_to_record:
                    break
                envs_to_record.append(self.unique_env_indices[j])

            if len(envs_to_record) < self.num_envs_to_record:
                for j in range(num_envs):
                    if len(envs_to_record) == self.num_envs_to_record:
                        break
                    if j in envs_to_record:
                        continue
                    envs_to_record.append(j)

            self.envs_to_record = envs_to_record  # [289, 1089, 2048]
            if self.log_dir is not None:
                self.vid_recorder = VideoRecorder(
                    self.envs_to_record, self.log_dir, "cpu"
                )
                self.vid_recorder.setup(self.env)
        print("setup vids")
        self.cur_learning_iteration = 0

    def get_scan_latent(self, obs):
        if self.encode_scan:
            scan_obs = obs[
                :,
                self.env.cfg.env.n_proprio : self.env.cfg.env.n_proprio
                + self.n_scan_input,
            ]
            scan_latent = self.scan_encoder(scan_obs)
        else:
            raise NotImplementedError
        return scan_obs, scan_latent

    def get_obs_actor_from_obs(
        self,
        obs,
        scan_obs,
        scan_latent,
        n_proprio: int,
        n_scan: int,
        history_len: int,
        yaw,
        priv_states,
        priv_latent,
    ):
        old_obs_for_buffer = torch.cat(
            (obs[:, :n_proprio], scan_obs, obs[:, -history_len * n_proprio :]), dim=1
        )
        old_obs_for_actor = torch.cat(
            (obs[:, :n_proprio], scan_latent, obs[:, -history_len * n_proprio :], yaw),
            dim=1,
        )
        if self.do_encoding_tricks:
            old_obs_for_actor = torch.cat(
                (old_obs_for_actor, priv_states, priv_latent), dim=1
            )

        return old_obs_for_buffer, old_obs_for_actor

    def get_priv_state_from_critic(self, critic_obs):
        priv_states = torch.concat(
            (
                critic_obs[
                    :,
                    self.env.priv_obs_lin_vel_start_idx : self.env.priv_obs_lin_vel_end_idx,
                ],
                # minus 3 for linear velocity
                torch.zeros(
                    (critic_obs.shape[0], self.n_priv_explicit - 3),
                    dtype=torch.float32,
                    device=self.device,
                ),
            ),
            dim=1,
        )
        return priv_states

    def get_priv_latent_and_explicit(
        self, obs, critic_obs, get_estimated_states, hist_encoding
    ):
        if get_estimated_states:
            obs_est = obs.clone()
            priv_states = self.estimator(
                obs_est[:, : self.env.cfg.env.n_proprio].detach()
            )
        else:
            priv_states = self.get_priv_state_from_critic(critic_obs)
        if hist_encoding:
            hist = obs[
                :, -self.env.cfg.env.history_len * self.env.cfg.env.n_proprio :
            ].detach()
            priv_latent = self.history_encoder(
                hist.view(-1, self.env.cfg.env.history_len, self.env.cfg.env.n_proprio)
            )
        else:
            priv_obs = self.get_priv_obs_from_critic(critic_obs)
            priv_latent = self.priv_encoder(priv_obs)
        return priv_states, priv_latent

    def get_obs_for_depth_encoder(self, obs):
        obs_for_depth_encoder = obs[:, : self.env.cfg.env.n_proprio].clone()
        if self.past_obs_for_depth_encoder > 0:
            obs_for_depth_encoder = torch.cat(
                (
                    obs_for_depth_encoder,
                    obs[:, -self.past_obs_for_depth_encoder * self.n_proprio :],
                ),
                dim=1,
            )
        return obs_for_depth_encoder

    def step_env_and_add_to_storage_buffer(
        self, it: int, record_vid: bool, prior_depth_latent=None, prior_yaw_est=None
    ):
        depth_latent, priv_yaws_est = (None, None)
        hist_encoding = it % self.dagger_update_freq == 0

        obs = self.env.get_observations()

        if self.env.cfg.env.num_privileged_obs is not None:
            critic_obs = self.env.get_privileged_observations()
        else:
            raise NotImplementedError

        if self.do_encoding_tricks:
            if not self.use_camera:
                priv_states, priv_latent = self.get_priv_latent_and_explicit(
                    obs, critic_obs, self.train_with_estimated_states, hist_encoding
                )
            else:
                priv_states, priv_latent = self.get_priv_latent_and_explicit(
                    obs, critic_obs, False, True
                )
        else:
            critic_obs = critic_obs[:, : self.num_obs_priv].clone()
            priv_states, priv_latent = (None, None)

        with torch.inference_mode():

            scan_obs, scan_latent = self.get_scan_latent(obs)

            cur_yaw = critic_obs[
                :,
                self.env.priv_obs_target_yaw_start_idx : self.env.priv_obs_target_yaw_end_idx,
            ]

            old_obs_for_buffer, old_obs_for_actor = self.get_obs_actor_from_obs(
                obs,
                scan_obs,
                scan_latent,
                self.env.cfg.env.n_proprio,
                self.env.cfg.env.n_scan,
                self.env.cfg.env.history_len,
                cur_yaw,
                priv_states,
                priv_latent,
            )

            actions, actions_log_prob, mus, sigmas, dists = self.ac_agent.act(
                old_obs_for_actor, self.env.env_class
            )
            actions_to_return = None

            actions_for_step = actions.detach()
            actions_for_step.requires_grad = False

        if self.use_camera:

            delta_yaw_ok = torch.abs(cur_yaw[:, 0]) < self.delta_yaw_thresh

            new_delta_yaw = torch.where(
                delta_yaw_ok.unsqueeze(1), prior_yaw_est.clone(), cur_yaw
            )

            if self.do_encoding_tricks:
                priv_states_student, priv_latent_student = (
                    self.get_priv_latent_and_explicit(obs, critic_obs, True, True)
                )
            else:
                critic_obs = critic_obs[:, : self.num_obs_priv].clone()
                priv_states_student, priv_latent_student = (None, None)

            old_obs_for_buffer, old_obs_for_actor = self.get_obs_actor_from_obs(
                obs,
                obs[
                    :,
                    self.env.cfg.env.n_proprio : self.env.cfg.env.n_proprio
                    + self.n_scan_input,
                ],
                prior_depth_latent,
                self.env.cfg.env.n_proprio,
                self.env.cfg.env.n_scan,
                self.env.cfg.env.history_len,
                new_delta_yaw,
                priv_states_student,
                priv_latent_student,
            )
            dists_cam, mus_cam, sigmas_cam = self.depth_actor(old_obs_for_actor)
            # actions_cam = dists_cam.rsample()
            actions_cam = mus_cam
            actions_log_prob_cam = dists_cam.log_prob(actions_cam)
            if self.actor_type in [
                ActorType.SINGLE_POLICY,
                ActorType.POLICY_PER_SKILL,
                ActorType.MIX_OF_EXPERTS,
            ]:
                actions_log_prob_cam = actions_log_prob_cam.sum(dim=-1)
                # prior_depth_latent,
            actions_to_return = (
                mus.clone(),
                actions_cam,
                critic_obs[
                    :,
                    self.env.priv_obs_target_yaw_start_idx : self.env.priv_obs_target_yaw_end_idx,
                ].clone(),
                prior_yaw_est,
                delta_yaw_ok,
            )

            if self.actor_type == ActorType.MIX_OF_EXPERTS:
                importances = torch.sum(self.depth_actor.weights, dim=0)
                coefficient_variation = importances.std() / importances.mean()
                moe_component = self.moe_loss_coeff * torch.pow(
                    coefficient_variation, 2
                )

                weights = self.depth_actor.weights.detach().clone()

                actions_to_return = (
                    mus.clone(),
                    actions_cam,
                    critic_obs[
                        :,
                        self.env.priv_obs_target_yaw_start_idx : self.env.priv_obs_target_yaw_end_idx,
                    ].clone(),
                    prior_yaw_est,
                    delta_yaw_ok,
                    moe_component,
                    weights,
                    self.env.env_class.clone(),
                )
            else:
                actions_to_return = (
                    mus.clone(),
                    actions_cam,
                    critic_obs[
                        :,
                        self.env.priv_obs_target_yaw_start_idx : self.env.priv_obs_target_yaw_end_idx,
                    ].clone(),
                    prior_yaw_est,
                    delta_yaw_ok,
                )

            if it >= self.iterations_step_with_teacher_before_student:
                actions_for_step = actions_cam.detach()
                actions_for_step.requires_grad = False

        with torch.inference_mode():
            new_obs, new_priv_obs, rewards, dones, infos = self.env.step(
                actions_for_step
            )  # obs has changed to next_obs !! if done obs has been reset

        if self.env.cfg.env.num_privileged_obs is not None:
            new_critic_obs = new_priv_obs
        else:
            new_critic_obs = new_obs.clone()

        if not self.do_encoding_tricks:
            new_critic_obs = new_critic_obs[:, : self.num_obs_priv].clone()

        with torch.inference_mode():
            values = self.ac_agent.get_value(critic_obs)
        rewards_total = rewards.clone()

        if "time_outs" in infos:
            rewards_total += (
                self.discount_factor * torch.squeeze(values, 1) * infos["time_outs"]
            )

        advantages = torch.zeros(
            rewards.shape, device=self.device
        )  # advantages we calculate at end of rollout
        adj_returns = torch.zeros(
            rewards.shape, device=self.device
        )  # bootstrapped returns we calc at end as well with advantage and value

        step_data = [
            old_obs_for_buffer,
            critic_obs,
            values,
            advantages.unsqueeze(1),
            actions,
            rewards_total.unsqueeze(1),
            dones.unsqueeze(1),
            actions_log_prob.unsqueeze(1),
            adj_returns.unsqueeze(1),
            mus,
            sigmas,
            self.env.env_class.unsqueeze(1),
        ]

        if self.use_camera:
            if infos["depth"] is not None:
                # clone here is important or mutate in place
                obs_for_depth_encoder = self.get_obs_for_depth_encoder(
                    old_obs_for_buffer
                )
                depth_latent, priv_yaws_est = self.depth_encoder(
                    infos["depth"].clone(), obs_for_depth_encoder
                )
            else:
                depth_latent = prior_depth_latent
                priv_yaws_est = prior_yaw_est
        step_data = [val.detach() for val in step_data]
        self.storage_buffer.add(step_data)

        vids_done = True
        # doing videos
        if record_vid:
            vids_done = self.vid_recorder.render(dones, it)

        # important when using camera that actions_to_return has gradient still....
        return (
            new_critic_obs,
            None,
            rewards,
            dones,
            infos,
            vids_done,
            actions_to_return,
            depth_latent,
            priv_yaws_est,
        )

    def log(self, metrics, it):

        width = 80
        pad = 35

        it_deets = (
            f" \033[1m Learning iteration {it}/{self.num_learning_iterations} \033[0m "
        )

        ep_string = f"""{'#' * width}\n""" f"""{it_deets.center(width, ' ')}\n\n"""

        keys_to_remove = []
        new_entries = {}

        for key, val in metrics.items():
            if type(val) == np.ndarray:
                val = val.item()
                new_entries[key] = val
                ep_string += f"""{f'{key}:':>{pad}} {val:.4f}\n"""
                keys_to_remove.append(key)
            elif type(val) != list:
                ep_string += f"""{f'{key}:':>{pad}} {val:.4f}\n"""
            else:
                ep_string += f"""{f'{key}:':>{pad}} {val}\n"""

                for j in range(len(val)):
                    new_entries[key + "_" + str(j)] = val[j]
                keys_to_remove.append(key)

        for bad_key in keys_to_remove:
            metrics.pop(bad_key)

        metrics = {**metrics, **new_entries}

        wandb.log(metrics, step=it)

        eta = self.tot_time / (it + 1) * (self.num_learning_iterations - it)
        mins = eta // 60
        secs = eta % 60
        ep_string += f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n"""
        print(ep_string)

    def update_ppo(self):
        mean_value_loss = 0
        mean_value = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_estimator_loss = 0
        mean_priv_reg_loss = 0
        mean_actor_loss = 0
        mean_advantage = 0
        mean_bootstrapped_returns = 0
        moe_cv_loss = 0

        grad_size_mb = 0.0

        moe_weights = []
        # moe_avg_weights_by_terrain_class = torch.zeros((self.env.cfg.terrain.num_cols, self.moe_n_experts), device=self.device, dtype=torch.float32, requires_grad=False)

        all_stddevs = torch.zeros(
            self.num_actions, device=self.device, requires_grad=False
        )
        n_std_dev_sums = 0

        moe_weights_by_env_class = {}

        moe_noise_mat_gating = torch.zeros(
            self.moe_n_experts, device=self.device, requires_grad=False
        )

        for i in range(len(self.unique_env_classes)):
            class_id = self.unique_env_classes[i]
            moe_weights_by_env_class[class_id] = []

        for batch_data in self.storage_buffer.sample(
            self.num_mini_batches, self.num_learning_epochs
        ):
            (
                obs,
                critic_obs,
                values,
                advantages,
                actions,
                rewards,
                dones,
                old_action_log_prob,
                bootstrapped_rewards,
                old_mus,
                old_sigmas,
                env_class,
            ) = batch_data

            scan_obs, scan_latent = self.get_scan_latent(obs)

            if self.do_encoding_tricks:
                priv_states, priv_latent = self.get_priv_latent_and_explicit(
                    obs, critic_obs, True, hist_encoding=False
                )
                real_explicit_states = self.get_priv_state_from_critic(critic_obs)
            else:
                priv_states, priv_latent, real_explicit_states = (None, None, None)

            old_obs_for_buffer, old_obs_for_actor = self.get_obs_actor_from_obs(
                obs,
                scan_obs,
                scan_latent,
                self.env.cfg.env.n_proprio,
                self.env.cfg.env.n_scan,
                self.env.cfg.env.history_len,
                critic_obs[
                    :,
                    self.env.priv_obs_target_yaw_start_idx : self.env.priv_obs_target_yaw_end_idx,
                ],
                real_explicit_states,
                priv_latent,
            )

            _, _, means, stddevs, dists = self.ac_agent.act(
                old_obs_for_actor, env_class.squeeze(1)
            )

            actions_log_probs = dists.log_prob(actions)
            if self.actor_type in [
                ActorType.SINGLE_POLICY,
                ActorType.POLICY_PER_SKILL,
                ActorType.MIX_OF_EXPERTS,
            ]:
                actions_log_probs = actions_log_probs.sum(dim=-1)
                all_stddevs += torch.mean(stddevs, dim=0)
                n_std_dev_sums += 1
                entropy_batch = dists.entropy().sum(dim=-1)

            if self.do_encoding_tricks:
                priv_obs = self.get_priv_obs_from_critic(critic_obs)
                priv_latent_batch = self.priv_encoder(priv_obs)

                # Adaptation module update
                with torch.inference_mode():
                    hist = obs[:, -self.history_len * self.env.cfg.env.n_proprio :]
                    hist_latent_batch = self.history_encoder(
                        hist.view(-1, self.history_len, self.env.cfg.env.n_proprio)
                    )
                priv_reg_loss = (
                    (priv_latent_batch - hist_latent_batch.detach())
                    .norm(p=2, dim=1)
                    .mean()
                )
                priv_reg_stage = min(
                    max((self.counter - self.priv_reg_coef_schedual[2]), 0)
                    / self.priv_reg_coef_schedual[3],
                    1,
                )
                priv_reg_coef = (
                    priv_reg_stage
                    * (self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0])
                    + self.priv_reg_coef_schedual[0]
                )

                # Estimator
                priv_states_predicted = self.estimator(
                    obs[:, : self.env.cfg.env.n_proprio].detach()
                )  # obs in batch is with true priv_states
                estimator_loss = (
                    (priv_states_predicted - real_explicit_states).pow(2).mean()
                )
                self.estimator_optimizer.zero_grad()
                estimator_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.estimator.parameters(), self.max_grad_norm
                )
                self.estimator_optimizer.step()
            else:
                priv_reg_loss = torch.tensor(
                    0.0, device=self.device, dtype=torch.float32
                )
                priv_reg_coef = torch.tensor(
                    0.0, device=self.device, dtype=torch.float32
                )
                estimator_loss = torch.tensor(
                    0.0, device=self.device, dtype=torch.float32
                )

            # KL
            if self.desired_kl != None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(stddevs / old_sigmas + 1.0e-5)
                        + (torch.square(old_sigmas) + torch.square(old_mus - means))
                        / (2.0 * torch.square(stddevs))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.actor_opt.param_groups:
                        param_group["lr"] = self.learning_rate
                    # for param_group in self.critic_opt.param_groups:
                    #     param_group['lr'] = self.learning_rate

            # Surrogate loss

            # print("transformed_obs_shape")
            # print(transformed_obs.shape)

            # print("actions_log_probs shape")
            # print(actions_log_probs.shape)
            # print("old_actions_log_probs shape")
            # print(old_action_log_prob.shape)

            # print("advantages shape")
            # print(advantages.shape)

            # print("advantages")
            # print(advantages)

            ratio = torch.exp(
                actions_log_probs - torch.squeeze(old_action_log_prob).detach()
            )
            surrogate = -torch.squeeze(advantages) * ratio
            surrogate_clipped = -torch.squeeze(advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # print("ratio is")
            # print(ratio)
            # print("surrogate")
            # print(surrogate)
            # print("surrogate_clipped")
            # print(surrogate_clipped)
            # print("surrogate_loss")
            # print(surrogate_loss)

            # Value function loss
            new_values = self.ac_agent.get_value(critic_obs.detach())
            if self.use_clipped_value_loss:
                value_clipped = values + (new_values - values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (new_values - bootstrapped_rewards).pow(2)
                value_losses_clipped = (value_clipped - bootstrapped_rewards).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (bootstrapped_rewards - new_values).pow(2).mean()
            # self.critic_opt.zero_grad()
            # value_loss.backward()
            # nn.utils.clip_grad_norm_(self.ac_agent.critic.parameters(), self.max_grad_norm)
            # self.critic_opt.step()

            entropy_loss = self.entropy_coef * entropy_batch.mean()
            priv_loss = priv_reg_coef * priv_reg_loss

            value_loss = self.value_loss_coef * value_loss

            actor_loss = surrogate_loss - entropy_loss + priv_loss + value_loss

            if self.actor_type == ActorType.MIX_OF_EXPERTS:
                importances = torch.sum(self.ac_agent.actor.weights, dim=0)
                coefficient_variation = importances.std() / importances.mean()
                moe_component = self.moe_loss_coeff * torch.pow(
                    coefficient_variation, 2
                )
                actor_loss += moe_component

            # loss = self.teacher_alpha * imitation_loss + (1 - self.teacher_alpha) * loss
            self.actor_opt.zero_grad()
            actor_loss.backward()

            grad_nbytes = 0.0

            for param in self.ac_agent.actor.parameters():
                grad_nbytes += param.grad.element_size() * param.grad.nelement()

            grad_size_mb = grad_nbytes / (1024**2)

            # nn.utils.clip_grad_norm_(list(self.scan_encoder.parameters()) + list(self.priv_encoder.parameters()) + list(self.ac_agent.actor.parameters()), self.max_grad_norm)
            nn.utils.clip_grad_norm_(
                list(self.ac_agent.critic.parameters())
                + list(self.ac_agent.actor.parameters())
                + list(self.scan_encoder.parameters()),
                self.max_grad_norm,
            )
            self.actor_opt.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_loss += entropy_loss.item()
            mean_estimator_loss += estimator_loss.item()
            mean_priv_reg_loss += priv_reg_loss.item()
            mean_advantage += advantages.mean().item()
            mean_value += new_values.mean().item()
            mean_bootstrapped_returns += bootstrapped_rewards.mean().item()
            mean_actor_loss += actor_loss.item()

            if self.actor_type == ActorType.MIX_OF_EXPERTS:
                weights = self.ac_agent.actor.weights
                moe_cv_loss += moe_component.item()

                moe_weights.append(weights.detach())

                for i in range(len(self.unique_env_classes)):
                    class_id = self.unique_env_classes[i]
                    weights_for_class = weights.detach()[
                        env_class.squeeze(1) == class_id, :
                    ]
                    moe_weights_by_env_class[class_id].append(weights_for_class)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value /= num_updates
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_estimator_loss /= num_updates
        mean_priv_reg_loss /= num_updates
        mean_actor_loss /= num_updates
        grad_size_mb /= num_updates
        mean_bootstrapped_returns /= num_updates
        self.storage_buffer.clear()
        self.counter += 1

        metrics = {
            "Loss/value_function": mean_value_loss,
            "Policy/mean_value": mean_value,
            "Loss/surrogate": mean_surrogate_loss,
            "Loss/entropy": mean_entropy_loss,
            "Loss/actor_total": mean_actor_loss,
            "Loss/mean_advantage": mean_advantage,
            "Policy/mean_bootstrapped_returns": mean_bootstrapped_returns,
            "Loss/learning_rate": self.learning_rate,
            "Train/actor_grad_size_mb": grad_size_mb
        }

        if self.do_encoding_tricks:
            metrics["Loss/estimator"] = mean_estimator_loss
            metrics["Loss/priv_reg"] = mean_priv_reg_loss
            metrics["Loss/priv_ref_lambda"] = priv_reg_coef

        if not self.actor_type == ActorType.MIX_OF_EXPERTS:
            metrics["Policy/mean_noise_std"] = torch.mean(
                all_stddevs / n_std_dev_sums
            ).item()
        else:
            moe_weights = torch.cat(moe_weights, dim=0)
            # moe_avg_weights_by_terrain_class /= num_updates
            moe_cv_loss /= num_updates

            metrics["Policy/moe_mean_std"] = torch.mean(
                all_stddevs / n_std_dev_sums
            ).item()

            metrics["Policy/moe_noise_scalar"] = (
                self.ac_agent.actor.noise_mat.mean(dim=0)
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

            metrics["Policy/moe_overall_avg_weights"] = (
                torch.mean(moe_weights, dim=0).detach().cpu().tolist()
            )
            metrics["Policy/moe_min_weight"] = min(
                metrics["Policy/moe_overall_avg_weights"]
            )
            metrics["Policy/moe_max_weight"] = max(
                metrics["Policy/moe_overall_avg_weights"]
            )

            metrics["Loss/moe_cv_loss"] = moe_cv_loss

            for i in range(len(self.unique_env_classes)):
                class_id = self.unique_env_classes[i]
                metric_name = "Policy/moe_env_class_" + str(class_id) + "_weights"
                all_weights_class = torch.cat(moe_weights_by_env_class[class_id], dim=0)
                avg_weight_class = (
                    torch.mean(all_weights_class, dim=0).detach().cpu().tolist()
                )
                metrics[metric_name] = avg_weight_class

        return metrics

    def get_priv_obs_from_critic(self, critic_obs):

        priv_obs = torch.concat(
            (
                critic_obs[
                    :,
                    self.env.priv_obs_mass_params_start_idx : self.env.priv_obs_mass_params_end_idx,
                ],
                critic_obs[
                    :,
                    self.env.priv_obs_friction_coeffs_start_idx : self.env.priv_obs_friction_coeffs_end_idx,
                ],
                critic_obs[
                    :,
                    self.env.priv_obs_motor_strength_start_idx : self.env.priv_obs_motor_strength_end_idx,
                ],
            ),
            dim=1,
        )
        priv_obs = torch.concat(
            (
                priv_obs,
                torch.zeros(
                    self.n_priv_latent - priv_obs.shape[1],
                    dtype=torch.float32,
                    device=self.device,
                ),
            ),
            dim=1,
        )
        return priv_obs

    def update_dagger(self):
        mean_hist_latent_loss = 0
        for batch_data in self.storage_buffer.sample(
            self.num_mini_batches, self.num_learning_epochs
        ):
            (
                obs,
                critic_obs,
                values,
                advantages,
                actions,
                rewards,
                dones,
                old_action_log_prob,
                bootstrapped_rewards,
                old_mus,
                old_sigmas,
                env_class,
            ) = batch_data

            with torch.inference_mode():
                priv_obs = self.get_priv_obs_from_critic(critic_obs)
                priv_latent_batch = self.priv_encoder(priv_obs)

            hist = obs[:, -self.history_len * self.env.cfg.env.n_proprio :]
            hist_latent_batch = self.history_encoder(
                hist.view(-1, self.history_len, self.env.cfg.env.n_proprio)
            )

            hist_latent_loss = (
                (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
            )
            self.hist_encoder_optimizer.zero_grad()
            hist_latent_loss.backward()
            nn.utils.clip_grad_norm_(
                self.history_encoder.parameters(), self.max_grad_norm
            )
            self.hist_encoder_optimizer.step()

            mean_hist_latent_loss += hist_latent_loss.item()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_hist_latent_loss /= num_updates
        self.counter += 1
        return mean_hist_latent_loss

    def update_depth(
        self,
        teacher_action_buffer,
        student_action_buffer,
        teacher_yaw_buffer,
        student_yaw_buffer,
        delta_yaw_ok,
        moe_cv_losses=None,
        moe_weights=None,
        env_classes=None,
    ):
        mean_value = 0
        mean_advantage = 0
        mean_bootstrapped_returns = 0
        moe_cv_loss = 0


        

        moe_avg_weights = torch.tensor(
            [0.0 for i in range(self.moe_n_experts)],
            device=self.device,
            requires_grad=False,
        )

        # moe_avg_weights_by_terrain_class = torch.zeros((self.env.cfg.terrain.num_cols, self.moe_n_experts), device=self.device, dtype=torch.float32, requires_grad=False)

        # depth_latent_buffer = []
        # scandots_latent_buffer = []

        if self.actor_type == ActorType.MIX_OF_EXPERTS:
            all_moe_weights = None

        for batch_data in self.storage_buffer.sample(
            self.num_mini_batches, self.num_learning_epochs
        ):
            (
                obs,
                critic_obs,
                values,
                advantages,
                actions,
                rewards,
                dones,
                old_action_log_prob,
                bootstrapped_rewards,
                old_mus,
                old_sigmas,
                env_class,
            ) = batch_data

            mean_advantage += advantages.mean().item()
            mean_value += values.mean().item()
            mean_bootstrapped_returns += bootstrapped_rewards.mean().item()

        # scandots_latent_buffer = torch.cat(scandots_latent_buffer, dim=0)
        # depth_latent_buffer = torch.cat(depth_latent_buffer, dim=0)
        depth_encoder_loss = 0
        actions_teacher_buffer = torch.cat(teacher_action_buffer, dim=0)
        actions_student_buffer = torch.cat(student_action_buffer, dim=0)
        yaw_teacher_buffer = torch.cat(teacher_yaw_buffer, dim=0)
        yaw_student_buffer = torch.cat(student_yaw_buffer, dim=0)

        depth_actor_loss = (
            (actions_teacher_buffer.detach() - actions_student_buffer)
            .norm(p=2, dim=1)
            .mean()
        )
        yaw_loss = (
            (yaw_teacher_buffer.detach() - yaw_student_buffer).norm(p=2, dim=1).mean()
        )

        loss = depth_actor_loss + yaw_loss

        if moe_cv_losses is not None and len(moe_cv_losses) > 1:
            moe_cv_losses = [t.reshape(1) for t in moe_cv_losses]
            moe_cv_losses = torch.cat(moe_cv_losses, dim=0)
            moe_cv_loss = torch.mean(moe_cv_losses)

            loss += moe_cv_loss

        self.depth_actor_optimizer.zero_grad()
        loss.backward()

        grad_nbytes = 0.0

        for param in self.depth_actor.named_parameters():
            if param[1].grad is None:
                continue
            grad_nbytes += param[1].grad.element_size() * param[1].grad.nelement()

        grad_size_mb = grad_nbytes / (1024**2)

        nn.utils.clip_grad_norm_(
            list(self.depth_actor.parameters()), self.max_grad_norm
        )  #  + list(self.depth_encoder.parameters())
        self.depth_actor_optimizer.step()

        self.depth_encoder.detach_hidden_states()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value /= num_updates
        mean_advantage /= num_updates
        mean_bootstrapped_returns /= num_updates
        self.storage_buffer.clear()
        self.counter += 1

        all_delta_yaw_ok = torch.cat(delta_yaw_ok, dim=0)
        delta_yaw_pct = (
            torch.nonzero(all_delta_yaw_ok).size(0) / all_delta_yaw_ok.numel()
        )

        metrics = {
            "Policy/mean_value": mean_value,
            "Loss/depth_action_loss": depth_actor_loss.item(),
            "Loss/depth_yaw_loss": yaw_loss.item(),
            "Loss/delta_yaw_ok_percent": delta_yaw_pct,
            "Loss/depth_total_loss": loss,
            "Loss/mean_advantage": mean_advantage,
            "Policy/mean_bootstrapped_returns": mean_bootstrapped_returns,
            "Loss/learning_rate": self.learning_rate,
            "Train/actor_grad_size_mb": grad_size_mb
        }

        if not self.actor_type == ActorType.MIX_OF_EXPERTS:
            metrics["Policy/mean_noise_std"] = (
                torch.exp(self.depth_actor.logstd.data).mean().item()
            )
        else:
            # moe_avg_weights_by_terrain_class /= num_updates
            metrics["Policy/moe_mean_std"] = [
                torch.exp(self.depth_actor.logstd.data).mean().item()
            ]

            moe_weights = torch.cat(moe_weights, dim=0)
            # moe_avg_weights_by_terrain_class /= num_updates
            metrics["Policy/moe_noise_scalar"] = (
                self.ac_agent.actor.noise_mat.mean(dim=0)
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )
            metrics["Policy/moe_overall_avg_weights"] = (
                torch.mean(moe_weights, dim=0).detach().cpu().tolist()
            )
            metrics["Policy/moe_avg_weight"] = sum(
                metrics["Policy/moe_overall_avg_weights"]
            ) / len(metrics["Policy/moe_overall_avg_weights"])
            metrics["Policy/moe_min_weight"] = min(
                metrics["Policy/moe_overall_avg_weights"]
            )
            metrics["Policy/moe_max_weight"] = max(
                metrics["Policy/moe_overall_avg_weights"]
            )

            metrics["Loss/moe_cv_loss"] = moe_cv_loss

            all_envs = torch.cat(env_classes, dim=0)
            for i in range(len(self.unique_env_classes)):
                class_id = self.unique_env_classes[i]
                metric_name = "Policy/moe_env_class_" + str(class_id) + "_weights"
                all_weights_class = moe_weights[all_envs == class_id, :]
                avg_weight_class = (
                    torch.mean(all_weights_class, dim=0).detach().cpu().tolist()
                )
                metrics[metric_name] = avg_weight_class

        return metrics

    def train(self):

        # self.scan_encoder.train()
        # self.priv_encoder.train()
        self.ac_agent.set_train(True)
        cur_episode_length = np.zeros((self.num_envs,), dtype=np.float32)
        cur_reward_sum = np.zeros((self.num_envs,), dtype=np.float32)

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        record_vid = self.enable_vids
        vids_done = False
        self.save_snapshot(0)

        self.prior_depth_latent, self.prior_yaw_est = (None, None)

        if self.use_camera:
            obs = self.env.get_observations()
            obs_depth_encoder = self.get_obs_for_depth_encoder(obs)
            self.prior_depth_latent, self.prior_yaw_est = self.depth_encoder(
                self.env.depth_buffer.clone().to(self.device)[:, -1], obs_depth_encoder
            )

            self.student_action_buffer = []
            self.teacher_action_buffer = []
            self.teacher_yaw_buffer = []
            self.student_yaw_buffer = []
            self.delta_yaw_ok_buffer = []
            self.moe_cv_losses = []
            self.moe_weights = []
            self.env_classes = []

        for it in range(self.cur_learning_iteration, self.num_learning_iterations):
            metrics = {}
            ep_infos = []
            start = time.time()
            # print("collecting data", flush=True)
            new_priv_obs = None

            hist_encoding = it % self.dagger_update_freq == 0

            for j in range(self.num_steps_per_env):

                (
                    new_obs,
                    _,
                    rewards,
                    dones,
                    infos,
                    vids_done,
                    actions_student_teacher,
                    new_depth_latent,
                    new_yaw_est,
                ) = self.step_env_and_add_to_storage_buffer(
                    it, record_vid, self.prior_depth_latent, self.prior_yaw_est
                )

                if self.use_camera:
                    self.prior_depth_latent = new_depth_latent
                    self.prior_yaw_est = new_yaw_est
                    self.teacher_action_buffer.append(actions_student_teacher[0])
                    self.student_action_buffer.append(actions_student_teacher[1])
                    self.teacher_yaw_buffer.append(actions_student_teacher[2])
                    self.student_yaw_buffer.append(actions_student_teacher[3])
                    self.delta_yaw_ok_buffer.append(actions_student_teacher[4])

                    if self.actor_type == ActorType.MIX_OF_EXPERTS:
                        moe_component = actions_student_teacher[5]
                        self.moe_cv_losses.append(moe_component)
                        self.moe_weights.append(actions_student_teacher[6])
                        self.env_classes.append(actions_student_teacher[7])

                self.global_steps += self.num_envs

                if self.enable_vids:
                    if record_vid and vids_done:
                        record_vid = False
                        self.vid_recorder.reset()

                    if it % self.record_vid_step_interval == 0:
                        record_vid = True

                if "episode" in infos:
                    ep_infos.append(infos["episode"])

                cur_reward_sum += to_numpy(rewards).reshape((self.num_envs,))
                cur_episode_length += 1
                new_ids = to_numpy((dones > 0).nonzero(as_tuple=False)).reshape((-1,))
                rewbuffer.extend(cur_reward_sum[new_ids].tolist())
                lenbuffer.extend(cur_episode_length[new_ids].tolist())
                cur_reward_sum[new_ids] = 0
                cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start

            with torch.inference_mode():
                last_values = self.ac_agent.get_value(new_obs).detach()

            dones = self.storage_buffer.get_buffer_data(self.dones_key)
            values = self.storage_buffer.get_buffer_data(self.values_key)
            rewards = self.storage_buffer.get_buffer_data(self.rewards_key)

            adj_returns, advantages = self.ac_agent.calc_advantages(
                last_values, dones, values, rewards, self.gamma, self.lam
            )

            self.storage_buffer.set_buffer_val(
                self.advantages_key, advantages.to(self.device)
            )
            self.storage_buffer.set_buffer_val(
                self.bootstrapped_rewards_key, adj_returns.to(self.device)
            )
            start = time.time()

            if self.use_camera:
                new_metrics = self.update_depth(
                    self.teacher_action_buffer,
                    self.student_action_buffer,
                    self.teacher_yaw_buffer,
                    self.student_yaw_buffer,
                    self.delta_yaw_ok_buffer,
                    self.moe_cv_losses,
                    self.moe_weights,
                    self.env_classes,
                )
                self.student_action_buffer.clear()
                self.teacher_action_buffer.clear()
                self.teacher_yaw_buffer.clear()
                self.student_yaw_buffer.clear()
                self.delta_yaw_ok_buffer.clear()
                self.moe_weights.clear()
                self.env_classes.clear()
                self.moe_cv_losses.clear()
                obs = self.env.get_observations()
                obs_depth_encoder = self.get_obs_for_depth_encoder(obs)
                self.prior_depth_latent, self.prior_yaw_est = self.depth_encoder(
                    self.env.depth_buffer.clone().to(self.device)[:, -1],
                    obs_depth_encoder,
                )
            else:
                new_metrics = self.update_ppo()
                if hist_encoding and self.do_encoding_tricks:
                    print("Updating dagger...")
                    mean_hist_latent_loss = self.update_dagger()
                    new_metrics["Loss/mean_hist_latent_loss"] = mean_hist_latent_loss

            metrics = {**metrics, **new_metrics}

            stop = time.time()
            learn_time = stop - start

            metrics["Perf/collection time"] = collection_time
            metrics["Perf/learning_time"] = learn_time
            metrics["Perf/global_steps"] = self.global_steps
            # metrics['Train/buffer_size'] = self.replay_buffer.cur_size
            if len(rewbuffer) > 0:
                metrics["Train/mean_reward"] = statistics.mean(rewbuffer)
                metrics["Train/mean_episode_length"] = statistics.mean(lenbuffer)
            for key in ep_infos[0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in ep_infos:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                metrics["Episode_rew/" + key] = to_numpy(value)
            self.tot_time += learn_time + collection_time

            if self.log_dir is not None:
                self.log(metrics, it)
            saved_snapshot = False
            save_snapshot = False

            if it % self.save_interval == 0:
                save_snapshot = True
            if save_snapshot:
                print("saving snapshot")
                self.save_snapshot(it)
                saved_snapshot = True

            if it != 0 and it % 1000 == 0 and self.record_vid_step_interval < 1000:
                self.record_vid_step_interval *= 2
            if it != 0 and it % 1000 == 0 and self.save_interval < 1000:
                self.save_interval *= 2

            if self.use_camera and saved_snapshot:
                # saving snapshot will modify params and we can't trace grad back, so reset it
                self.depth_encoder.detach_hidden_states()
                obs = self.env.get_observations()
                obs_depth_encoder = self.get_obs_for_depth_encoder(obs)
                self.prior_depth_latent, self.prior_yaw_est = self.depth_encoder(
                    self.env.depth_buffer.clone().to(self.device)[:, -1],
                    obs_depth_encoder,
                )

            self.cur_learning_iteration += 1

    def _load_modules(self, state_dict):
        self.ac_agent.actor.load_state_dict(state_dict["actor_state_dict"])
        self.ac_agent.critic.load_state_dict(state_dict["critic_state_dict"])
        self.scan_encoder.load_state_dict(state_dict["scan_encoder_state_dict"])
        self.actor_opt.load_state_dict(state_dict["actor_optimizer_state_dict"])
        # self.critic_opt.load_state_dict(state_dict["critic_optimizer_state_dict"])
        if self.do_encoding_tricks:
            self.priv_encoder.load_state_dict(state_dict["priv_encoder_state_dict"])
            self.history_encoder.load_state_dict(
                state_dict["history_encoder_state_dict"]
            )
            self.hist_encoder_optimizer.load_state_dict(
                state_dict["history_encoder_opto_state_dict"]
            )
            self.estimator.load_state_dict(state_dict["estimator_state_dict"])
            self.estimator_optimizer.load_state_dict(
                state_dict["estimator_opto_state_dict"]
            )
            self.priv_encoder.train()
            self.history_encoder.train()
            self.estimator.train()
        self.cur_learning_iteration = state_dict["learn_iter"]
        self.ac_agent.set_train(True)

    def _load_depth_modules(self, state_dict):
        print("loading depth actor")
        self.ac_agent.actor.train(False)
        self.ac_agent.critic.train(False)
        self.scan_encoder.train(False)
        self.history_encoder.eval()
        self.estimator.eval()
        self.priv_encoder.eval()
        self.depth_actor.load_state_dict(state_dict["depth_actor_state_dict"])
        self.depth_backbone.load_state_dict(state_dict["depth_backbone"])
        self.depth_encoder.load_state_dict(state_dict["depth_encoder"])
        self.depth_actor.train()
        self.depth_backbone.train()
        self.depth_encoder.train()
        self.depth_actor_optimizer.load_state_dict(
            state_dict["depth_actor_opto_state_dict"]
        )

    def save_snapshot(self, it):
        snapshot = os.path.join(self.log_dir, f"snapshot_{it}.pt")

        print("saving snapshot")

        state_dict = {
            "actor_type": self.actor_type,
            "env_classes": self.unique_env_classes.tolist(),
            "critic_state_dict": self.ac_agent.critic.state_dict(),
            "actor_state_dict": self.ac_agent.actor.state_dict(),
            "scan_encoder_state_dict": self.scan_encoder.state_dict(),
            "actor_optimizer_state_dict": self.actor_opt.state_dict(),
            # 'critic_optimizer_state_dict': self.critic_opt.state_dict(),
            "learn_iter": self.cur_learning_iteration,
            "global_steps": self.global_steps,
        }

        if self.do_encoding_tricks:
            more_states = {
                "priv_encoder_state_dict": self.priv_encoder.state_dict(),
                "history_encoder_state_dict": self.history_encoder.state_dict(),
                "history_encoder_opto_state_dict": self.hist_encoder_optimizer.state_dict(),
                "estimator_state_dict": self.estimator.state_dict(),
                "estimator_opto_state_dict": self.estimator_optimizer.state_dict(),
            }
            state_dict = {**state_dict, **more_states}

        if self.use_camera:
            state_dict["depth_actor_state_dict"] = self.depth_actor.state_dict()
            state_dict["depth_backbone"] = self.depth_backbone.state_dict()
            state_dict["depth_encoder"] = self.depth_encoder.state_dict()
            state_dict["depth_actor_opto_state_dict"] = (
                self.depth_actor_optimizer.state_dict()
            )

        with open(snapshot, "wb") as f:
            torch.save(state_dict, f)
        with open(snapshot, "rb") as f:
            state_dict = torch.load(f, map_location=self.device, weights_only=False)

        self._load_modules(state_dict)

        if self.use_camera:
            self._load_depth_modules(state_dict)

    def load_snapshot(self, path: str):
        with open(path, "rb") as f:
            state_dict = torch.load(f, map_location=self.device, weights_only=False)

        assert self.actor_type == state_dict["actor_type"]
        assert self.unique_env_classes.tolist() == state_dict["env_classes"]
        self._load_modules(state_dict)
        self.cur_learning_iteration = self.cur_learning_iteration + 1
        if self.use_camera and "depth_actor_state_dict" in state_dict.keys():
            self._load_depth_modules(state_dict)
        # if first load for training new depth policy off of scandots policy
        elif self.use_camera and "depth_actor_state_dict" not in state_dict.keys():
            if self.actor_type in [ActorType.SINGLE_POLICY, ActorType.MIX_OF_EXPERTS]:
                self.depth_actor = deepcopy(self.ac_agent.actor)
            elif self.actor_type == ActorType.POLICY_PER_SKILL:
                pass
            else:
                raise NotImplementedError
