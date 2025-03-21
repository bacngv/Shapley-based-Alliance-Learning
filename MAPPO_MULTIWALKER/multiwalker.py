# multiwalker.py
import gymnasium as gym
import numpy as np
from pettingzoo.sisl import multiwalker_v9

class GymMultiWalkerWrapper(gym.Env):
    """
    Wrapper để chuyển môi trường multiwalker của PettingZoo sang giao diện Gym hỗ trợ nhiều agent.
    Sử dụng 3 agent: 'walker_0', 'walker_1', 'walker_2'.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode="rgb_array", **kwargs):
        # Thêm tham số render_mode vào khi khởi tạo environment
        self.env = multiwalker_v9.parallel_env(render_mode=render_mode, **kwargs)
        # Lấy danh sách các agent, sử dụng 3 agent đầu tiên
        self.agents = self.env.possible_agents[:3]
        # Reset ban đầu để thiết lập không gian quan sát/hành động
        obs, _ = self.env.reset()
        self.observation_space = self.env.observation_space(self.agents[0])
        self.action_space = self.env.action_space(self.agents[0])

    def reset(self, seed=None, options=None):
        result = self.env.reset(seed=seed, options=options)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        obs = {agent: obs[agent] for agent in self.agents}
        return obs, info

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        obs = {agent: obs[agent] for agent in self.agents}
        rewards = {agent: rewards[agent] for agent in self.agents}
        # Ghép terminated và truncated thành done flag
        done_flags = {agent: terminated[agent] or truncated[agent] for agent in self.agents}
        infos = {agent: infos[agent] for agent in self.agents}
        return obs, rewards, done_flags, infos

    def render(self, mode="human"):
        return self.env.render()

    def close(self):
        self.env.close()
