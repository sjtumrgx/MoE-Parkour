import copy
import rclpy
from rclpy.node import Node
import numpy as np
import torch
import torchvision
import time
import os
import cv2

import matplotlib.pyplot as plt

import signal
import sys

# import imageio

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


class CamViewer(Node):

    def __init__(self, device: str = "cuda:0"):
        super().__init__("cam_viewer")
        self.device = device
        self.robo_rl_controller = RobotRLController(device)

        self.depth_cam_sub = self.create_subscription(
            Image, "/camera/depth/image_rect_raw", self._forward_depth_callback, 1
        )
        self.depth_cam_sub  # prevent unused variable warning
        self.color_cam_sub = self.create_subscription(
            Image, "/camera/color/image_raw", self._forward_rgb_cb, 1
        )
        self.color_cam_sub  # prevent unused variable warning
        self.br = CvBridge()

        self.imgs_to_save = []
        self.depth_ctr = 0
        self.color_ctr = 0

        self.out_dir = "/docker_mount/vids"
        os.makedirs(self.out_dir, exist_ok=True)

    #     os.makedirs(out_dir, exist_ok=True)

    def _forward_depth_callback(self, msg):
        depth_img_cv = self.br.imgmsg_to_cv2(msg, "16UC1")

        depth_img_cv_flt = depth_img_cv.astype(np.float32) * 0.001

        final_img = self.robo_rl_controller._process_depth_img(depth_img_cv)

        cv2.imshow("input depth (b/w)", depth_img_cv_flt + 0.5)
        cv2.waitKey(1)

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_img_cv_flt, alpha=50.0), cv2.COLORMAP_JET
        )
        cv2.imshow("input depth (colourized)", depth_colormap)
        cv2.waitKey(1)

        cv2.imshow("processed depth (b/w)", final_img.detach().cpu().numpy() + 0.5)
        cv2.waitKey(1)

        # self.imgs_to_save.append([depth_colormap])

        # step_filename_depth = os.path.join(
        #     self.out_dir, str(self.depth_ctr).zfill(6) + "_depth" + ".jpg"
        # )
        # with open(step_filename_depth, "wb") as file_handle:
        #     imageio.imwrite(file_handle, depth_colormap, format="JPEG")

        self.depth_ctr += 1

        # fig, ax = plt.subplots()
        # img_to_plot = cv2.normalize(depth_img_cv_flt + 0.5, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # ax.imshow(img_to_plot, cmap="Greys")
        # plt.show()
        # cv2.imshow("processed_input", final_img.detach().cpu().numpy() + 0.5)
        # cv2.waitKey(1)

    def _forward_rgb_cb(self, msg):
        img_cv = self.br.imgmsg_to_cv2(msg, "rgb8")
        cv2.imshow("input rgb", img_cv[:, :, [2, 1, 0]])
        cv2.waitKey(1)
        # self.imgs_to_save[-1].append(img_cv)
        # step_filename_color = os.path.join(self.out_dir, str(self.color_ctr).zfill(6) + "_color" + ".jpg")
        # with open(step_filename_color, "wb") as file_handle:
        #     imageio.imwrite(file_handle, img_cv, format="JPEG")
        self.color_ctr += 1

    # def process_imgs(self):

    #     cur_ctr = 0

    #     out_dir = "/docker_mount/vids"

    #     os.makedirs(out_dir, exist_ok=True)
    #     filename_n_zeros = 6

    #     print("processing imgs")

    #     for i in range(len(self.imgs_to_save)):
    #         print(i)
    #         imgs = self.imgs_to_save[i]
    #         depth_img = imgs[0]
    #         color_img = imgs[1]

    #         step_filename_color = os.path.join(out_dir, str(cur_ctr).zfill(filename_n_zeros) + "_color" + ".jpg")
    #         step_filename_depth = os.path.join(out_dir, str(cur_ctr).zfill(filename_n_zeros) + "_depth" + ".jpg")
    #         with open(step_filename_color, "wb") as file_handle:
    #             imageio.imwrite(file_handle, step_filename_color, format="JPEG")
    #         with open(step_filename_depth, "wb") as file_handle:
    #             imageio.imwrite(file_handle, step_filename_depth, format="JPEG")

    #     fps = 15
    #     out_vid_color = os.path.join(out_dir, f"vid_color.mp4")
    #     in_dir_pattern = os.path.join(out_dir, "*color.jpg")
    #     ffmpeg_cmd = ["ffmpeg", "-y", "-framerate", f"{fps}", "-pattern_type", "glob", "-i", f"{in_dir_pattern}", "-c:v", "libx264", "-pix_fmt", "yuv420p", out_vid_color]

    #     res = subprocess.run(ffmpeg_cmd, shell=False, capture_output=True, timeout=None)
    #     assert res.returncode == 0, f"{res}"
    #     out_vid_depth = os.path.join(out_dir, f"vid_depth.mp4")
    #     in_dir_pattern = os.path.join(out_dir, "*depth.jpg")
    #     ffmpeg_cmd = ["ffmpeg", "-y", "-framerate", f"{fps}", "-pattern_type", "glob", "-i", f"{in_dir_pattern}", "-c:v", "libx264", "-pix_fmt", "yuv420p", out_vid_depth]

    #     res = subprocess.run(ffmpeg_cmd, shell=False, capture_output=True, timeout=None)
    #     assert res.returncode == 0, f"{res}"


def main(args=None):

    rclpy.init(args=args)

    device = "cuda:0"

    cam_viewer = CamViewer(device=device)

    print("spinning")
    while rclpy.utilities.ok():
        rclpy.spin_once(cam_viewer)

    # ffmpeg -y -framerate 15 -pattern_type glob -i '*color.jpg' -c:v libx264 -pix_fmt yuv420p color.mp4
    print("done spinning")
    cam_viewer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
