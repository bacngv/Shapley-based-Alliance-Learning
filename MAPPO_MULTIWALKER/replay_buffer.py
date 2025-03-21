import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim  # thêm action_dim
        self.episode_limit = args.max_cycles
        self.batch_size = args.batch_size
        self.episode_num = 0
        self.buffer = None
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {
            'obs_n': np.zeros([self.batch_size, self.episode_limit, self.N, self.obs_dim]),
            's': np.zeros([self.batch_size, self.episode_limit, self.state_dim]),
            'v_n': np.zeros([self.batch_size, self.episode_limit + 1, self.N]),
            'raw_a_n': np.zeros([self.batch_size, self.episode_limit, self.N, self.action_dim]),
            'a_logprob_n': np.zeros([self.batch_size, self.episode_limit, self.N]),
            'r_n': np.zeros([self.batch_size, self.episode_limit, self.N]),
            'done_n': np.zeros([self.batch_size, self.episode_limit, self.N])
        }
        self.episode_num = 0


    def store_transition(self, episode_step, obs_n, s, v_n, raw_a_n, a_logprob_n, r_n, done_n):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.buffer['raw_a_n'][self.episode_num][episode_step] = raw_a_n  # Sửa từ 'a_n' thành 'raw_a_n'
        self.buffer['a_logprob_n'][self.episode_num][episode_step] = a_logprob_n
        self.buffer['r_n'][self.episode_num][episode_step] = r_n
        self.buffer['done_n'][self.episode_num][episode_step] = done_n
    def store_last_value(self, episode_step, v_n):
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.episode_num += 1

    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():
            batch[key] = torch.tensor(self.buffer[key], dtype=torch.float32)
        return batch
