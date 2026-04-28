"""Basic model configs for Unitree Go2"""

import numpy as np
import os.path as osp

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

go2_action_scale = 0.25
go2_const_dof_range = dict(
    Hip_max=1.0472,
    Hip_min=-1.0472,
    Front_Thigh_max=3.4907,
    Front_Thigh_min=-1.5708,
    Rear_Thingh_max=4.5379,
    Rear_Thingh_min=-0.5236,
    Calf_max=-0.83776,
    Calf_min=-2.7227,
)


class Go2RoughCfg(LeggedRobotCfg):

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.5]  # [m]
        default_joint_angles = {  # 12 joints in the order of simulation
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

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        stiffness = {"joint": 40.0}  # [N*m/rad]
        damping = {"joint": 1.0}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = go2_action_scale
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        computer_clip_torque = False
        motor_clip_torque = True

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf"
        name = "go2"
        foot_name = "foot"
        front_hip_names = ["FL_hip_joint", "FR_hip_joint"]
        rear_hip_names = ["RL_hip_joint", "RR_hip_joint"]
        penalize_contacts_on = ["base", "Head", "thigh", "calf"]
        terminate_after_contacts_on = []  # , "thigh", "calf"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        sdk_dof_range = go2_const_dof_range
        dof_velocity_override = 35.0

        joint_limits_low = {  # 12 joints in the order of simulation
            "FL_hip_joint": -1.0472,
            "FL_thigh_joint": -1.5708,
            "FL_calf_joint": -2.7227,
            "FR_hip_joint": -1.0472,
            "FR_thigh_joint": -1.5708,
            "FR_calf_joint": -2.7227,
            "RL_hip_joint": -1.0472,
            "RL_thigh_joint": -0.5236,
            "RL_calf_joint": -2.7227,
            "RR_hip_joint": -1.0472,
            "RR_thigh_joint": -0.5236,
            "RR_calf_joint": -2.7227,
        }

        joint_limits_high = {  # 12 joints in the order of simulation
            "FL_hip_joint": 1.0472,
            "FL_thigh_joint": 3.4907,
            "FL_calf_joint": -0.83776,
            "FR_hip_joint": 1.0472,
            "FR_thigh_joint": 3.4907,
            "FR_calf_joint": -0.83776,
            "RL_hip_joint": 1.0472,
            "RL_thigh_joint": 4.5379,
            "RL_calf_joint": -0.83776,
            "RR_hip_joint": 1.0472,
            "RR_thigh_joint": 4.5379,
            "RR_calf_joint": -0.83776,
        }

        torque_limits = {
            "FL_hip_joint": 25,
            "FL_thigh_joint": 40,
            "FL_calf_joint": 40,
            "FR_hip_joint": 25,
            "FR_thigh_joint": 40,
            "FR_calf_joint": 40,
            "RL_hip_joint": 25,
            "RL_thigh_joint": 40,
            "RL_calf_joint": 40,
            "RR_hip_joint": 25,
            "RR_thigh_joint": 40,
            "RR_calf_joint": 40,
        }

    class rewards(LeggedRobotCfg.rewards):
        # class scales(LeggedRobotCfg.rewards.scales):
        #     exceed_dof_pos_limits = -0.4
        #     exceed_torque_limits_l1norm = -0.4
        #     dof_vel_limits = -0.4
        soft_dof_vel_limit = 0.9
        soft_dof_pos_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 100
        base_height_target = 1.0
        # class scales( LeggedRobotCfg.rewards.scales ):
        # torques = -0.0002
        # dof_pos_limits = -10.0

    class depth:
        use_camera = False
        camera_num_envs = 192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        # 14cm to middle of body? If height of cam is too low, object is too low

        # position_min = [0.23088, 0, 0.125]  # front camera 150.88mm (from cad drawing to mount holes) + 4.5cm to middle of body + 4.5cm from mount holes to cam center = 334.34mm
        # position_max = [0.25088, 0, 0.135]  # front camera
        # position_min = [0.28, 0, 0.14]  # front camera 70+150.88+(131.0÷2)   70mm from mount holes forward plus 150.88 to rail mount holes plus half of rail mount holes dist
        # position_max = [0.30, 0, 0.15]  # front camera, 286.38 should be it
        # position_min = [0.345, 0, 0.13]  # front camera 150.88mm (from cad drawing to mount holes) + 4.5cm to middle of body + 4.5cm from mount holes to cam center = 334.34mm
        # position_max = [0.345, 0, 0.13]  # front camera
        # degrees, euler

        # position = dict(mean=[0.28638, -0.0175, 0.14], std=[0.01, 0.0025, 0.02])
        position = dict(mean=[0.32, -0.0175, 0.15], std=[0.01, 0.0025, 0.02])

        angle_min = [-1, 21.2, -1.0]
        angle_max = [1, 24.6, 1.0]
        # angle_min = [0.0, 23.0, 0.0]
        # angle_max = [0.0, 23.0, 0.0]

        # we do 15 fps in reality, if we run policy at 50hz, for each 3.33 steps in sim per update ((1/15)/(1/50))
        update_interval = 4  # 5 works without retraining, 8 worse

        original = (int(640 / 4), int(480 / 4))
        resized = (87, 58)
        buffer_len = 2

        # realsense d435f
        # https://store.realsenseai.com/buy-intel-realsense-depth-camera-d435f.html, FOV 87
        horizontal_fov = [85, 89]
        # horizontal_fov = [87, 87]
        # this is from original image size
        left_clip = 20
        right_clip = 5
        # bottom_clip = 64 // 4
        bottom_clip = 16
        near_clip = 0.15
        far_clip = 3.0
        dis_noise = 0.0
        # dis_noise = 0.00
        gaussian_blur_kernel = 3
        gaussian_blur_sigma = (0.1, 2.0)

        scale = 1
        invert = True

        artifact_prob = 0.001
        artifact_height_mean_std = (3, 3)
        artifact_width_mean_std = (3, 3)

        contour_detection_kernel_size = 3
        contour_threshold = 1.0
        contour_nuke_prob = 0.1

        iterations_step_with_teacher_before_student = 0
        do_depth_noise = True

    class noise:
        add_noise = True

        class noise_scales:
            rotation = 0.025
            dof_pos = 0.01
            dof_vel = 1.5
            ang_vel = 0.2

        contact_filt_flip_prob = 0.05
        global_steps_delay = (
            0  # 24*280*50 # Go2RoughCfgPPO.runner.num_steps_per_env * 2000
        )


class Go2RoughCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_go2"

    class depth_encoder:
        if_depth = Go2RoughCfg.depth.use_camera
        depth_shape = Go2RoughCfg.depth.resized
        buffer_len = Go2RoughCfg.depth.buffer_len
        hidden_dims = 512
        learning_rate = 1.0e-3
        num_steps_per_env = Go2RoughCfg.depth.update_interval * 24
