import unittest

import numpy as np

from rl_lib.replay_buffer import ReplayBuffer, DatasetSpec



class TestReplayBuffer(unittest.TestCase):
    def test_add_resize(self):
        test_data_keys = ["observations", "actions", "rewards", "is_dones"]
        test_data_shapes = [(99,), (12,), (1,), (1,)]
        test_data_types = [np.float32, np.float32, np.float32, bool]
        test_data_specs = DatasetSpec(test_data_keys, test_data_shapes, test_data_types)

        ep_len = 100

        num_envs = 64

        disc = 0.99

        max_size = 35000
        max_num_steps_per_episode = 100

        buffer = ReplayBuffer(test_data_specs, max_size, disc, num_envs, max_num_steps_per_episode)

        eps_needed = (max_size // (num_envs * ep_len)) + 1

        for i in range(eps_needed):
            for j in range(ep_len):

                step_data = []
                for k in range(len(test_data_specs.keys)):

                    expected_shape = test_data_specs.shapes[k]
                    expected_dtype = test_data_specs.dtypes[k]

                    d = np.zeros((num_envs, *expected_shape), dtype=expected_dtype)
                    step_data.append(d)

                buffer.add(step_data)
            
            buffer.move_from_working_to_buffer(np.arange(num_envs))


        self.assertLessEqual(buffer.cur_size, max_size)

        sample = buffer.sample(256)

        self.assertEqual(sample[0].shape, (256, 99))
        self.assertEqual(sample[-1].shape, (256, 99))


if __name__ == "__main__":
    unittest.main()