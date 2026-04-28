from crc_module import get_crc

import copy
import rclpy
from rclpy.node import Node
import numpy as np
import torch
import torchvision
import time
import os
import cv2

import signal
import sys

from unitree_go.msg import SportModeState, LowState, LowCmd
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from get_algo_wo_isaac_gym import (
    RobotRLController,
    ENV_DICT,
    DOF_MAP,
    TORQUE_LIMITS,
    DOF_SIGNS,
    SIT_JOINT_ANGLES_ARR,
    TURN_ON_MOTOR_MODE,
    ACTION_SCALE,
)

ROLL_LIMIT = np.pi / 3
PITCH_LIMIT = np.pi / 3


class Go2Controller(Node):

    def __init__(self, device: str, do_policy: bool = False):
        super().__init__("go2_controller")
        self.device = device
        self.do_policy = True
        self.robo_rl_controller = RobotRLController(device, vx=0.6)

        # to do low level, gotta kill the sports mode state controller
        # self.sport_sub = self.create_subscription(
        #     SportModeState,
        #     '/sportmodestate',
        #     self.sportmodestate_cb,
        #     10)

        self.depth_cam_sub = self.create_subscription(
            Image, "/camera/depth/image_rect_raw", self._forward_depth_callback, 1
        )
        self.depth_cam_sub  # prevent unused variable warning

        self.br = CvBridge()

        self.low_sub = self.create_subscription(
            LowState, "/lowstate", self.lowstate_cb, 10
        )
        self.low_sub

        # ROS publishers
        self.low_cmd_pub = self.create_publisher(LowCmd, "/lowcmd", 1)

        # self.low_cmd_pub = self.create_publisher(
        #     LowCmd,
        #     "/notthis_fake",
        #     1
        # )
        self.low_cmd_pub

        self.last_sports_mode_msg = None
        self.last_lowstate_msg = None

        self.do_policy = do_policy
        if self.do_policy:
            self.obs_timer = self.create_timer(
                self.robo_rl_controller.dt, self.action_timer
            )
        else:
            self.wiggle_timer = self.create_timer(
                self.robo_rl_controller.dt, self.do_wiggle
            )

        self.device = device

        self.low_cmd_buffer = LowCmd()

        self.dryrun = False

        self.traj = None
        self.target_joint_idx_sim = None
        self.cur_traj_idx = 0

        self.out_dir_name = os.path.dirname(__file__)

        self.ready = False

        self.depth_buffer = torch.zeros(
            1,
            ENV_DICT["depth"]["buffer_len"],
            ENV_DICT["depth"]["resized"][1],
            ENV_DICT["depth"]["resized"][0],
            device=self.device,
            requires_grad=False,
        )
        self.init_depth_buf = True

        self.ready_start_time = 0.0

    def _forward_depth_callback(self, msg):
        depth_img_cv = self.br.imgmsg_to_cv2(msg, "16UC1")

        final_img = self.robo_rl_controller._process_depth_img(depth_img_cv)

        # print(f"processed_img max: {processed_img.max()}, processed_img min: {processed_img.min()}")
        # depth_img_color = cv2.applyColorMap(cv2.convertScaleAbs(
        #     depth_img_cv.astype(np.float32) * 0.001, alpha=255/3.0), cv2.COLORMAP_JET)

        # processed_depth_img_color = cv2.applyColorMap(cv2.convertScaleAbs(
        #     processed_img.detach().cpu().numpy(), alpha=255/3.0), cv2.COLORMAP_JET)

        # processed_final_img_color = cv2.applyColorMap(cv2.convertScaleAbs(
        #     final_img.detach().cpu().numpy(), alpha=255/3.0), cv2.COLORMAP_JET)

        # depth_img_meters = depth_img_cv.astype(np.float32) * 0.001
        # depth_img_meters[-30:,-30:] = 3.0
        # cv2.imshow("input b/w", depth_img_meters)
        # cv2.waitKey(1)
        # cv2.imshow("Input", depth_img_color)
        # cv2.waitKey(1)

        # cv2.imshow("Input Processed", processed_depth_img_color)
        # cv2.waitKey(1)
        # cv2.imshow("final_img Processed", processed_final_img_color)
        # cv2.waitKey(1)

        if self.init_depth_buf:
            self.depth_buffer[0] = torch.stack(
                [final_img] * ENV_DICT["depth"]["buffer_len"], dim=0
            ).to(self.device)
            self.init_depth_buf = False
        else:
            self.depth_buffer[0] = torch.cat(
                [self.depth_buffer[0, 1:], final_img.to(self.device).unsqueeze(0)],
                dim=0,
            ).to(self.device)

    def set_ready(self):
        self.ready = True
        self.ready_start_time = time.time()

    def get_act_to_default(self, goals, num_steps, cur_step):

        times = np.array([0, num_steps])

        target_pos = np.zeros(self.robo_rl_controller.num_dofs, dtype=np.float32)
        for i in range(self.robo_rl_controller.num_dofs):
            res = np.interp(cur_step, times, goals[i, :])
            target_pos[i] = res

        return target_pos

    def get_joint_pos_from_msg_sim(self, msg):

        sim_joint_pos = np.zeros(self.robo_rl_controller.num_dofs, dtype=np.float32)
        for sim_idx in range(self.robo_rl_controller.num_dofs):
            real_idx = DOF_MAP[sim_idx]
            motor_state = msg.motor_state[real_idx]
            sim_joint_pos[sim_idx] = motor_state.q
        return sim_joint_pos

    def lowstate_not_ready(self):
        return self.last_lowstate_msg is None or all(
            np.isclose(
                self.get_joint_pos_from_msg_sim(self.last_lowstate_msg), 0.0, atol=0.1
            )
        )

    def move_to_pos(self, goal_pos, time_to_pos: float = 1.5):

        while self.lowstate_not_ready():

            rclpy.spin_once(self)
            time.sleep(0.1)

        num_steps = int(time_to_pos / self.robo_rl_controller.dt)
        cur_step = 0

        joint_pos = self.get_joint_pos_from_msg_sim(self.last_lowstate_msg)
        goals = np.zeros((self.robo_rl_controller.num_dofs, 2), dtype=np.float32)

        goals[:, 1] = goal_pos

        goals[:, 0] = joint_pos

        for i in range(num_steps):

            # print(f"step: {i}")

            # print(f"cur pos: {self.get_joint_pos_from_msg_sim(self.last_lowstate_msg)}")

            target_joint_pos = self.get_act_to_default(goals, num_steps, cur_step)
            # print(f"target: {target_joint_pos}")
            target_joint_pos = torch.tensor(
                target_joint_pos, device=self.device, dtype=torch.float32
            )
            self._publish_legs_cmd(target_joint_pos)

            start_time = time.time()

            while time.time() - start_time < self.robo_rl_controller.dt:
                rclpy.spin_once(self)

            cur_step += 1

    def get_traj(self):
        run_time = 5.0
        start_move = 2.0
        move_radians = -np.pi / 8
        time_to_move = 0.5
        time_to_stay = 1.0

        time_to_init = 1.0

        target_joint_idx_sim = -1
        target_joint_name = "RR_calf_joint"
        for i in range(self.robo_rl_controller.num_dofs):
            name = self.robo_rl_controller.dof_names[i]
            if name == target_joint_name:
                target_joint_idx_sim = i

        num_iterations = int(run_time / self.robo_rl_controller.dt)

        robot_id = 0

        # we track time, pos, and vel
        tracked_data = np.zeros((num_iterations, 6), dtype=np.float32)
        # tracked_data[:, 4] = env.default_dof_pos_all[robot_id, target_joint_idx].cpu().numpy()
        tracked_data[:, 4] = 0.0

        increments_to_move = int(time_to_move / self.robo_rl_controller.dt)
        angle_increment = move_radians / increments_to_move

        start_up_move_idx = int(start_move / self.robo_rl_controller.dt)

        start_down_move_idx = int(
            (start_move + time_to_move + time_to_stay) / self.robo_rl_controller.dt
        )

        # action scale: target angle = actionScale * action + defaultAngle
        # moves = env.default_dof_pos_all[robot_id, target_joint_idx].cpu().numpy() + np.arange(increments_to_move) * angle_increment
        default_pos_targ = (
            self.robo_rl_controller.default_dof_pos[target_joint_idx_sim].cpu().numpy()
        )
        moves = (1 / ACTION_SCALE) * np.arange(increments_to_move) * angle_increment

        tracked_data[start_up_move_idx : start_up_move_idx + increments_to_move, 4] = (
            moves
        )

        tracked_data[
            start_up_move_idx + increments_to_move : start_down_move_idx, 4
        ] = moves[
            -1
        ]  # env.default_dof_pos_all[robot_id, target_joint_idx].cpu().numpy() + move_radians

        tracked_data[
            start_down_move_idx : start_down_move_idx + increments_to_move, 4
        ] = np.flip(moves)

        return tracked_data, target_joint_idx_sim, start_up_move_idx

    def send_action(self, actions):
        """Send the action to the robot motors, which does the preprocessing
        just like env.step in simulation.
        Thus, the actions has the batch dimension, whose size is 1.
        """
        # print("act")
        clipped_scaled_action = self.robo_rl_controller.do_action_limits_and_scale(
            actions
        )
        robot_coordinates_action = (
            clipped_scaled_action + self.robo_rl_controller.default_dof_pos.unsqueeze(0)
        )

        self._publish_legs_cmd(robot_coordinates_action[0])

    def sportmodestate_cb(self, msg):
        self.last_sports_mode_msg = msg

    def lowstate_cb(self, msg):
        self.last_lowstate_msg = msg

        # skip safety check if we're not commanding the robot with policy
        if not self.ready or self.dryrun:
            return

        for sim_idx in range(self.robo_rl_controller.num_dofs):
            real_idx = DOF_MAP[sim_idx]
            joint_pos = msg.motor_state[real_idx].q
            if (
                joint_pos > self.robo_rl_controller.joint_pos_protect_high[sim_idx]
                or joint_pos < self.robo_rl_controller.joint_pos_protect_low[sim_idx]
            ):
                self.ready = False
                self.get_logger().error(
                    f"Joint {sim_idx}(sim), {real_idx}(real) position out of range at {joint_pos}"
                )
                self.get_logger().error("The motors and this process shuts down.")
                self._turn_off_motors()
            tau_est = msg.motor_state[real_idx].tau_est
            if abs(tau_est) > TORQUE_LIMITS[sim_idx]:
                self.ready = False
                self.get_logger().error(
                    f"Joint {sim_idx}(sim), {real_idx}(real) tau out of range at {tau_est}"
                )
                self.get_logger().error("The motors and this process shuts down.")
                self._turn_off_motors()

        if (
            np.abs(msg.imu_state.rpy[0]) > ROLL_LIMIT
            or np.abs(msg.imu_state.rpy[1]) > PITCH_LIMIT
        ):
            self.ready = False
            self.get_logger().error(
                f"Pitch: {msg.imu_state.rpy[1]} (tol: {PITCH_LIMIT}), Roll: {msg.imu_state.rpy[0]}, out of tol (tol: {ROLL_LIMIT})"
            )
            self.get_logger().error("The motors and this process shuts down.")
            self._turn_off_motors()

    def action_timer(self):

        if not self.ready:
            return None
        if self.lowstate_not_ready() or self.init_depth_buf:
            return None

        last_lowstate_msg = self.last_lowstate_msg
        with torch.inference_mode():
            obs, depth_latent = self.msg_to_obs(last_lowstate_msg)
            actions = self.robo_rl_controller.obs_latent_to_act(obs, depth_latent)
            self.send_action(actions)
        return True

    def do_wiggle(self):

        if not self.ready or self.lowstate_not_ready():
            return None

        if self.traj is None or self.cur_traj_idx >= self.traj.shape[0]:

            if not self.traj is None and self.cur_traj_idx >= self.traj.shape[0]:
                # save current info
                with open(
                    os.path.join(self.out_dir_name, "out_traj.bin"), "wb"
                ) as file_handle:
                    np.save(file_handle, self.traj)
                print("saved traj")

            print("making traj")
            tracked_data, target_joint_idx_sim, self.start_up_move_idx = self.get_traj()
            print(f"sim idx: {target_joint_idx_sim}")
            self.traj = tracked_data
            self.target_joint_idx_sim = target_joint_idx_sim
            self.cur_traj_idx = 0
            self.start_time = time.time()

        last_low_msg = self.last_lowstate_msg
        base_ang_vel, rpy, pos, vel, tau, foot_force = self.msg_to_needed_arrs(
            last_low_msg
        )

        # for limits on torque/pos, we have to update this in the base class
        self.robo_rl_controller.dof_pos_[:] = torch.tensor(
            pos, device=self.device, dtype=torch.float32
        )
        self.robo_rl_controller.dof_vel_[:] = torch.tensor(
            vel, device=self.device, dtype=torch.float32
        )
        self.robo_rl_controller.tau_est_[:] = torch.tensor(
            tau, device=self.device, dtype=torch.float32
        )

        self.traj[self.cur_traj_idx, 0] = time.time() - self.start_time
        self.traj[self.cur_traj_idx, 1] = pos[self.target_joint_idx_sim]
        self.traj[self.cur_traj_idx, 2] = vel[self.target_joint_idx_sim]
        self.traj[self.cur_traj_idx, 3] = tau[self.target_joint_idx_sim]

        action = torch.zeros(
            (1, self.robo_rl_controller.num_dofs),
            device=self.device,
            dtype=torch.float32,
        )
        action[0, self.target_joint_idx_sim] = torch.tensor(
            self.traj[self.cur_traj_idx, 4], device=self.device
        )
        self.robo_rl_controller.action_buf[
            self.robo_rl_controller.action_buf_len - 1, :
        ] = action.flatten()

        # print(f"cur_pos: {pos[self.target_joint_idx_sim]}, goal: {self.traj[self.cur_traj_idx, 4]}")

        self.send_action(action)

        if self.cur_traj_idx == self.start_up_move_idx:
            print("moving")
        self.cur_traj_idx += 1

    def msg_to_needed_arrs(self, msg):

        # this seems to be in the body frame, when i rotate the robot around z x and y ang vel follow body frame
        base_ang_vel = np.array(msg.imu_state.gyroscope)

        rpy = np.array(msg.imu_state.rpy)

        pos = np.zeros(12, dtype=np.float32)
        vel = np.zeros(12, dtype=np.float32)
        tau = np.zeros(12, dtype=np.float32)

        for sim_idx in range(self.robo_rl_controller.num_dofs):
            real_idx = DOF_MAP[sim_idx]
            motor_state = msg.motor_state[real_idx]
            pos[sim_idx] = motor_state.q
            vel[sim_idx] = motor_state.dq
            tau[sim_idx] = motor_state.tau_est

        foot_force = np.array(msg.foot_force)

        foot_force = self.robo_rl_controller.reindex_feet(
            foot_force.reshape((1, -1))
        ).flatten()

        return base_ang_vel, rpy, pos, vel, tau, foot_force

    def msg_to_obs(self, msg):

        base_ang_vel, rpy, pos, vel, tau, foot_force = self.msg_to_needed_arrs(msg)
        depth_buf = self.depth_buffer[0, -1, :, :]
        obs, depth_latent = self.robo_rl_controller.arrs_to_obs(
            base_ang_vel, rpy, pos, vel, tau, foot_force, depth_buf
        )

        return obs, depth_latent

    def _publish_legs_cmd(self, robot_coordinates_action: torch.Tensor):
        """Publish the joint commands to the robot legs in robot coordinates system.
        robot_coordinates_action: shape (NUM_DOF,), in simulation order.
        """
        for sim_idx in range(self.robo_rl_controller.num_dofs):
            real_idx = DOF_MAP[sim_idx]
            if not self.dryrun:
                self.low_cmd_buffer.motor_cmd[real_idx].mode = TURN_ON_MOTOR_MODE[
                    sim_idx
                ]
            if sim_idx == self.target_joint_idx_sim:
                self.traj[self.cur_traj_idx, 5] = (
                    robot_coordinates_action[self.target_joint_idx_sim].item()
                    * DOF_SIGNS[sim_idx]
                )
            self.low_cmd_buffer.motor_cmd[real_idx].q = (
                robot_coordinates_action[sim_idx].item() * DOF_SIGNS[sim_idx]
            )
            self.low_cmd_buffer.motor_cmd[real_idx].dq = 0.0
            self.low_cmd_buffer.motor_cmd[real_idx].tau = 0.0
            self.low_cmd_buffer.motor_cmd[real_idx].kp = (
                self.robo_rl_controller.p_gains[sim_idx].item()
            )
            self.low_cmd_buffer.motor_cmd[real_idx].kd = (
                self.robo_rl_controller.d_gains[sim_idx].item()
            )

        self.low_cmd_buffer.crc = get_crc(self.low_cmd_buffer)
        self.low_cmd_pub.publish(self.low_cmd_buffer)

    def _turn_off_motors(self):
        """Turn off the motors"""
        self.ready = False
        new_msg = LowCmd()
        for sim_idx in range(self.robo_rl_controller.num_dofs):
            real_idx = DOF_MAP[sim_idx]
            new_msg.motor_cmd[real_idx].mode = 0x00
            new_msg.motor_cmd[real_idx].q = 0.0
            new_msg.motor_cmd[real_idx].dq = 0.0
            new_msg.motor_cmd[real_idx].tau = 0.0
            new_msg.motor_cmd[real_idx].kp = 0.0
            new_msg.motor_cmd[real_idx].kd = 0.0
        new_msg.crc = get_crc(new_msg)
        self.low_cmd_pub.publish(new_msg)
        for i in range(5):
            rclpy.spin_once(self)


def main(args=None):

    rclpy.init(args=args)

    device = "cuda:0"

    # global exit_flag
    # exit_flag = False
    # def signal_handler(sig, frame):
    #     global exit_flag
    #     exit_flag = True
    #     go2_controller.ready = False
    #     go2_controller._turn_off_motors()
    #     print('You pressed Ctrl+C!')
    # signal.signal(signal.SIGINT, signal_handler)

    go2_controller = Go2Controller(device=device, do_policy=True)

    print("standing")
    go2_controller.move_to_pos(
        go2_controller.robo_rl_controller.default_dof_pos.cpu().detach().numpy(), 1.5
    )
    go2_controller.set_ready()

    run_time = 3.5
    start_time = time.time()

    print("moving")
    while (
        rclpy.utilities.ok()
        and go2_controller.ready
        and time.time() - start_time < run_time
    ):
        rclpy.spin_once(go2_controller)

    print("sitting")
    if rclpy.utilities.ok() and go2_controller.ready:
        go2_controller.ready = False
        go2_controller.move_to_pos(SIT_JOINT_ANGLES_ARR, 2.5)
    else:
        print("shutting off motors")
        go2_controller._turn_off_motors()
    go2_controller.destroy_node()
    rclpy.shutdown()

    # except KeyboardInterrupt:
    #     pass
    # except ExternalShutdownException:
    #     sys.exit(1)


if __name__ == "__main__":
    main()
