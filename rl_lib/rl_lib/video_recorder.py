from typing import List, Optional
import cv2
import logging
import math
import threading
import subprocess

import numpy as np
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

try:
    import imageio
    import isaacgym
    import isaacgym.torch_utils as torch_utils
    from isaacgym import gymapi
except ImportError:
    imageio = None
    isaacgym = None
    torch_utils = None
    gymapi = None


class VideoRecorder:
    def __init__(self, envs_to_record: List[int], output_dir: str, device: str) -> None:

        if not os.path.exists(output_dir):
            raise ValueError

        self.output_base_dir = output_dir
        self.output_working_dir = os.path.join(output_dir, "img_working_dir")
        os.makedirs(output_dir, exist_ok=True)
        self.envs_to_record = envs_to_record
        self.output_env_dirs = {i: None for i in envs_to_record}

        self.device = device

        for i in envs_to_record:
            output_dir = os.path.join(self.output_working_dir, f"env_{str(i).zfill(5)}")
            os.makedirs(output_dir, exist_ok=True)
            self.output_env_dirs[i] = output_dir

        self._cameras = []

        self.filename_n_zeros = 4
        self.camera_width = 1280
        self.camera_height = 720
        self._camera_type = gymapi.IMAGE_COLOR
        self.max_ep_len = 1000

        # self.image_cache = torch.zeros((len(self.envs_to_record), self.max_ep_len, self.camera_height, self.camera_width, 3), dtype=torch.uint8, requires_grad=False, device=self.device)
        self.image_cache_idx = torch.zeros((len(self.envs_to_record),), requires_grad=False, dtype=torch.int32, device=self.device)
        self.image_cache_done = torch.zeros((len(self.envs_to_record),), requires_grad=False, dtype=torch.bool, device=self.device )
        self.img_cache_in_progress = torch.zeros((len(self.envs_to_record),), requires_grad=False, dtype=torch.bool, device=self.device )

        # for text info
        # font
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.text_goal_template = "goal: vx: {:.2f}, vy: {:.2f}, vth: {:.2f}"
        self.text_cur_template = "cur : vx: {:.2f}, vy: {:.2f}, vth: {:.2f}"

        # org
        # width, then height
        self.org_goal = (5, self.camera_height - 30)
        self.org_cur = (5, self.camera_height - 15)

        # fontScale
        self.fontScale = 0.5
 
        # Blue color in BGR
        self.color = (0, 255, 0)

        # Line thickness of 2 px
        self.thickness = 1
 


    def attach_view_camera(self, i, env_handle, actor_handle, root_pos):
        if True:
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.camera_width
            camera_props.height = self.camera_height
            # camera_props.enable_tensors = True
            # camera_props.horizontal_fov = camera_horizontal_fov

            camera_handle = self._gym.create_camera_sensor(env_handle, camera_props)
            self._cameras.append(camera_handle)
            
            print("created camera handle")
            print(camera_handle)

            cam_pos = root_pos + np.array([0, 1, 0.5])
            self._gym.set_camera_location(camera_handle, env_handle, gymapi.Vec3(*cam_pos), gymapi.Vec3(*root_pos))

    def setup(self, env) -> None:
        """Setup the web viewer

        :param gym: The gym
        :type gym: isaacgym.gymapi.Gym
        :param sim: Simulation handle
        :type sim: isaacgym.gymapi.Sim
        :param envs: Environment handles
        :type envs: list of ints
        :param cameras: Camera handles
        :type cameras: list of ints
        """
        self._gym = env.gym
        self._sim = env.sim
        self._envs = env.envs
        self._cameras = []
        self._env = env
        self.cam_pos_rel = np.array([0, 2, 1])

        self.dt = env.dt
        self.fps = int(1/self.dt)
        for i in self.envs_to_record:
            root_pos = self._env.root_states[i, :3].cpu().numpy()
            self.attach_view_camera(i, self._envs[i], self._env.actor_handles[i], root_pos)
    
    def reset(self):
        self.image_cache_idx[:] = 0
        self.img_cache_in_progress[:] = False
        self.image_cache_done[:] = False

        # remove prior images
        for env_idx, out_dir in self.output_env_dirs.items():
            old_imgs = os.listdir(out_dir)
            for old_img in old_imgs:
                if not old_img.endswith('.jpg'):
                    continue
                os.remove(os.path.join(out_dir, old_img))

    
    def finish_vid(self, img_idx: int, it: int):
        # write out images to dir, process to mp4 with ffmpeg
        env_idx = self.envs_to_record[img_idx]
        out_dir = self.output_env_dirs[env_idx]
       
        out_vid = os.path.join(out_dir, f"vid_{str(it).zfill(6)}.mp4")
        in_dir_pattern = os.path.join(out_dir, "*.jpg")
        ffmpeg_cmd = ["ffmpeg", "-y", "-framerate", f"{self.fps}", "-pattern_type", "glob", "-i", f"{in_dir_pattern}", "-c:v", "libx264", "-pix_fmt", "yuv420p", out_vid]

        res = subprocess.run(ffmpeg_cmd, shell=False, capture_output=True, timeout=None)
        assert res.returncode == 0, f"{res}"
    
    def render(self, is_dones, it: int,
               fetch_results: bool = True,
               step_graphics: bool = True,
               render_all_camera_sensors: bool = True) -> None:
        """Render and get the image from the current camera

        This function must be called after the simulation is stepped (post_physics_step).
        The following Isaac Gym functions are called before get the image.
        Their calling can be skipped by setting the corresponding argument to False

        - fetch_results
        - step_graphics
        - render_all_camera_sensors

        :param fetch_results: Call Gym.fetch_results method (default: True)
        :type fetch_results: bool
        :param step_graphics: Call Gym.step_graphics method (default: True)
        :type step_graphics: bool
        :param render_all_camera_sensors: Call Gym.render_all_camera_sensors method (default: True)
        :type render_all_camera_sensors: bool
        """

        # isaac gym API
        if fetch_results:
            self._gym.fetch_results(self._sim, True)
        if step_graphics:
            self._gym.step_graphics(self._sim)
        if render_all_camera_sensors:
            self._gym.render_all_camera_sensors(self._sim)

        # get image, but first we wait for a reset so we have a fresh episode
        for i in range(len(self.envs_to_record)):

            env_idx = self.envs_to_record[i]

            prev_done = self.image_cache_done[i]

            if prev_done:
                continue

            is_inprogress = self.img_cache_in_progress[i]

            if not is_inprogress and is_dones[env_idx]:
                self.img_cache_in_progress[i] = True
            elif not is_inprogress:
                continue
            else:
                if is_dones[env_idx]:
                    self.image_cache_done[i] = True
                    self.finish_vid(i, it)
                    continue

            if self.image_cache_idx[i] >= self.max_ep_len:
                continue
            image = self._gym.get_camera_image(self._sim,
                                            self._envs[env_idx],
                                            self._cameras[i],
                                            self._camera_type)
            # copy needed for cv2...
            image = image.reshape(image.shape[0], -1, 4)[..., :3].copy()
            vx, vy, vtheta = self._env.commands[env_idx, 0:3].detach().cpu().numpy()
            fmt_str_goal = self.text_goal_template.format(vx, vy, vtheta)
            vx, vy, vtheta = self._env.base_lin_vel[env_idx, 0:3].detach().cpu().numpy()
            fmt_str_cur = self.text_cur_template.format(vx, vy, vtheta)

            image = cv2.putText(image, fmt_str_goal, self.org_goal, self.font, 
                   self.fontScale, self.color, self.thickness, cv2.LINE_AA)
            image = cv2.putText(image, fmt_str_cur, self.org_cur, self.font, 
                   self.fontScale, self.color, self.thickness, cv2.LINE_AA)

            if self._env.cfg.depth.use_camera:
                self._image_depth = self._env.depth_buffer[env_idx, -1].cpu().numpy() + 0.5
                self._image_depth = np.uint8(255 * self._image_depth)
        
            root_pos = self._env.root_states[env_idx, :3].cpu().numpy()
            cam_pos = root_pos + self.cam_pos_rel
            self._gym.set_camera_location(self._cameras[i], self._envs[env_idx], gymapi.Vec3(*cam_pos), gymapi.Vec3(*root_pos))

            cur_img_idx = self.image_cache_idx[i]

            out_dir = self.output_env_dirs[env_idx]

            step_filename = os.path.join(out_dir, str(cur_img_idx.detach().cpu().item()).zfill(self.filename_n_zeros) + ".jpg")
            with open(step_filename, "wb") as file_handle:
                imageio.imwrite(file_handle, image, format="JPEG")
            self.image_cache_idx[i] = cur_img_idx + 1


        return torch.all(self.image_cache_done)


