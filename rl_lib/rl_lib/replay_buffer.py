from collections import OrderedDict
from dataclasses import dataclass
from bisect import bisect_left
import random
from typing import Dict, Tuple
import torch

import numpy as np

OBSERVATION_KEY = "observations"
ACTIONS_KEY = "actions"
REWARDS_KEY = "rewards"
DONE_KEY = "is_dones"
ADVANTAGES_KEY = "advantages"
NEXT_OBSERVATION_KEY = "next_observations"

@dataclass
class DatasetSpec:
    keys: Tuple[str]
    shapes: Tuple[int]
    dtypes: Tuple[type]

    def __post_init__(self):
        assert len(self.keys) == len(self.shapes) == len(self.dtypes)
        num_items = len(self.keys)
        for i in range(num_items):
            assert type(self.keys[i]) == str
            assert type(self.shapes[i]) == tuple
            for j in range(len(self.shapes[i])):
                assert type(self.shapes[i][j]) == int

# this one stores just one episode per env, and uses pytorch to store on device
class StorageBuffer:
    def __init__(self, specs: DatasetSpec, num_envs: int, steps_per_episode: int, device: str):
        self.specs = specs
        self.device = device
        self.num_envs = num_envs
        self.steps_per_episode = steps_per_episode

        assert OBSERVATION_KEY in self.specs.keys
        assert DONE_KEY in self.specs.keys
        assert ADVANTAGES_KEY in self.specs.keys

        self.step_idx = 0
        self.buffer_data = []
        self.buffer_data_key_index_map = OrderedDict()
        self._reset_buffer()

    def _reset_buffer(self):
    
        self.buffer_data = []
        self.buffer_data_key_index_map = OrderedDict()
        self.step_idx = 0

        for i in range(len(self.specs.keys)):
            dtype = self.specs.dtypes[i]
            shape = self.specs.shapes[i]
            k = self.specs.keys[i]

            desired_shape = (self.steps_per_episode, self.num_envs, *shape)
            new_arr = torch.zeros(desired_shape, dtype=dtype, device=self.device, requires_grad=False)
            self.buffer_data.append(new_arr)
            self.buffer_data_key_index_map[k] = i

    def add(self, step_data):
        if self.step_idx >= self.steps_per_episode:
            raise AssertionError("Rollout buffer overflow")

        assert len(step_data) == len(self.specs.keys)
        
        for i in range(len(self.specs.keys)):
            key_name = self.specs.keys[i]
            data_to_add = step_data[i]
            expected_shape = self.specs.shapes[i]
            expected_dtype = self.specs.dtypes[i]

            desired_shape = (self.num_envs, *expected_shape)

            assert data_to_add.shape == desired_shape, f"key {key_name} had wrong shape. Expected {desired_shape}, got {data_to_add.shape}"
            assert data_to_add.dtype == expected_dtype, f"key {key_name} has wrong dtype. Expected {expected_dtype} and got {data_to_add.dtype}"

            self.buffer_data[i][self.step_idx].copy_(data_to_add)
        self.step_idx += 1

    def clear(self):
        self.step_idx = 0

    def sample(self, num_mini_batches: int, num_epochs: int=8):

        len_out_data = len(self.specs.keys)

        batch_size = self.num_envs * self.steps_per_episode
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                out_data = [None] * len_out_data
                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                for i in range(len(self.specs.keys)):
                    out_data[i] = self.buffer_data[i].flatten(0, 1)[batch_idx]

                
                yield out_data

    def set_buffer_val(self, key_str: str, vals):
        assert key_str in self.buffer_data_key_index_map.keys()
        key_idx = self.buffer_data_key_index_map[key_str]
        expected_shape = self.specs.shapes[key_idx]
        expected_dtype = self.specs.dtypes[key_idx]

        desired_shape = (self.steps_per_episode, self.num_envs, *expected_shape)

        assert vals.shape == desired_shape, f"{key_str} had wrong shape. Expected {desired_shape}, got {vals.shape}"
        assert vals.dtype == expected_dtype, f"{key_str} has wrong dtype. Expected {expected_dtype} and got {vals.dtype}"

        self.buffer_data[key_idx] = vals

    def get_buffer_data(self, key_str: str):
        assert key_str in self.buffer_data_key_index_map.keys()
        key_idx = self.buffer_data_key_index_map[key_str]
        return self.buffer_data[key_idx]


# this one will store old episodes as well, and uses numpy
class ReplayBuffer:
    def __init__(self, specs: DatasetSpec, max_size: int, discount_factor: float,
                 num_envs: int, max_num_steps_per_episode: int, add_next_observation_on_sample: bool = True,
                 device: str = None):
        self.max_size = max_size
        self.discount_factor = discount_factor
        self.specs = specs
        self.add_next_observation_on_sample = add_next_observation_on_sample
        self.max_num_steps_per_episode = max_num_steps_per_episode

        assert OBSERVATION_KEY in self.specs.keys
        assert DONE_KEY in self.specs.keys
        self.num_envs = num_envs

        self.cur_size = 0

        self.buffer_data = []
        self.buffer_data_key_index_map = OrderedDict()
        self.episode_indices_in_buffer = np.zeros((0,), dtype=np.uint32)

        for i in range(len(self.specs.keys)):
            dtype = self.specs.dtypes[i]
            shape = self.specs.shapes[i]
            k = self.specs.keys[i]
            new_arr = np.zeros((0, *shape), dtype=dtype)
            self.buffer_data.append(new_arr)
            self.buffer_data_key_index_map[k] = i

        # working buffer will be a list of size num envs, with each entry a list of size number specs, with each entry the buffer
        self.working_buffer = [[None] * len(self.specs.keys) for i in range(self.num_envs)]
        self.all_env_indices = np.arange(self.num_envs)
        self._reset_working_buffer_by_index(self.all_env_indices)

        # make sure reasonable max_size, if we want to store 5 rollouts for however environments
        assert self.max_size > max_num_steps_per_episode * self.num_envs * 5

    def _reset_working_buffer_by_index(self, env_indices):

        for j in env_indices:
            self.working_buffer[j] = [None] * len(self.specs.keys)
        for i in range(len(self.specs.keys)):
            for j in env_indices:
                dtype = self.specs.dtypes[i]
                shape = self.specs.shapes[i]
                new_arr = np.zeros((0, *shape), dtype=dtype)
                self.working_buffer[j][i] = new_arr


    def add(self, step_data):
        """
        step data is multidimensional where it has outer dimensions of num keys we trackign in buffer, then
        eache entry there has shape we expect, then each has dtype we expect. We use numpy to store.
    
        """

        assert len(step_data) == len(self.specs.keys)
        
        for i in range(len(self.specs.keys)):
            key_name = self.specs.keys[i]
            data_to_add = step_data[i]
            expected_shape = self.specs.shapes[i]
            expected_dtype = self.specs.dtypes[i]

            assert data_to_add.shape == (self.num_envs, *expected_shape), f"key {key_name} had wrong shape. Expected {(self.num_envs, *expected_shape)}, got {data_to_add.shape}"
            assert data_to_add.dtype == expected_dtype, f"key {key_name} has wrong dtype. Expected {expected_dtype} and got {data_to_add.dtype}"

            for j in range(self.num_envs):
                ep_data = data_to_add[j]
                self.working_buffer[j][i] = np.concatenate([self.working_buffer[j][i], ep_data.reshape((1, *(ep_data.shape)))])

    def move_from_working_to_buffer(self, env_indices):

        ep_lens = [self.working_buffer[j][0].shape[0] for j in env_indices]
        amount_to_add = sum(ep_lens)

        if self.cur_size + amount_to_add > self.max_size and len(self.episode_indices_in_buffer) > 1:
            num_to_remove = self.cur_size + amount_to_add - self.max_size
            number_removed = 0

            while number_removed < num_to_remove:
                
                ep_starting_idx = self.episode_indices_in_buffer[0]
                next_ep_idx = self.episode_indices_in_buffer[1]

                old_ep_len = next_ep_idx - ep_starting_idx

                indices_to_remove = np.arange(ep_starting_idx, next_ep_idx, dtype=np.uint32)
                for i in range(len(self.specs.keys)):
                    
                    self.buffer_data[i] = np.delete(self.buffer_data[i], indices_to_remove, 0)

                self.episode_indices_in_buffer = np.delete(self.episode_indices_in_buffer, 0, 0)
                self.episode_indices_in_buffer -= old_ep_len

                self.cur_size -= old_ep_len

                number_removed += old_ep_len

        for j in env_indices:
            ep_len = self.working_buffer[j][0].shape[0]
            new_ep_idx = self.buffer_data[0].shape[0]
            for i in range(len(self.specs.keys)):
                new_data = self.working_buffer[j][i]
                self.buffer_data[i] = np.concatenate([self.buffer_data[i], new_data])
            self.episode_indices_in_buffer = np.concatenate([self.episode_indices_in_buffer, [new_ep_idx]])
            self.cur_size += ep_len

        self._reset_working_buffer_by_index(env_indices)


    def sample(self, num_samples: int):

        len_out_data = len(self.specs.keys)
        if self.add_next_observation_on_sample:
            len_out_data += 1

        out_data = [None] * len_out_data

        sample_indices = np.random.randint(0, high=self.cur_size, size=num_samples, dtype=np.uint32)

        for i in range(len(self.specs.keys)):
            out_data[i] = self.buffer_data[i][sample_indices, :]

        if self.add_next_observation_on_sample:
            obs_key_idx = self.buffer_data_key_index_map[OBSERVATION_KEY]

            # use the is done flag to check for where we are inadvertently at end of traj
            is_done_key_idx = self.buffer_data_key_index_map[DONE_KEY]

            is_dones = self.buffer_data[is_done_key_idx][sample_indices, :].reshape((-1,))
            next_obs_indices = np.where(is_dones, sample_indices, sample_indices + 1)
            out_data[-1] = self.buffer_data[obs_key_idx][next_obs_indices, :]

        return out_data