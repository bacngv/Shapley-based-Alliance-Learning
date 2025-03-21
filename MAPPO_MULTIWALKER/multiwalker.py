# multiwalker.py
import gymnasium as gym
import numpy as np
from pettingzoo.sisl import multiwalker_v9

class GymMultiWalkerWrapper(gym.Env):
    """
    Wrapper để chuyển môi trường multiwalker của PettingZoo sang giao diện Gym hỗ trợ nhiều agent.
    Ở đây, ta sử dụng 3 agent: ví dụ 'walker_0', 'walker_1', 'walker_2'.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, **kwargs):
        # Sử dụng giao diện parallel giúp nhận/tải dữ liệu cho nhiều agent cùng lúc.
        self.env = multiwalker_v9.parallel_env(**kwargs)
        # Lấy danh sách tất cả agent khả dụng và chỉ dùng 3 agent đầu tiên.
        self.agents = self.env.possible_agents[:3]
        # Reset ban đầu để khởi tạo không gian của các agent.
        obs, _ = self.env.reset()
        # Giả sử tất cả agent có cùng không gian quan sát và hành động.
        self.observation_space = self.env.observation_space(self.agents[0])
        self.action_space = self.env.action_space(self.agents[0])

    def reset(self, seed=None, options=None):
        result = self.env.reset(seed=seed, options=options)
        # Kiểm tra nếu hàm reset trả về tuple (obs, info)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        # Lọc chỉ lấy các agent đã chọn
        obs = {agent: obs[agent] for agent in self.agents}
        return obs, info

    def step(self, actions):
        """
        Thực hiện bước hành động với đầu vào là dictionary:
            actions = { 'walker_0': action0, 'walker_1': action1, 'walker_2': action2 }
        Trả về: obs, rewards, done_flags, infos
        """
        # Với gymnasium, step trả về 5 giá trị: obs, rewards, terminated, truncated, infos
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        # Lọc lại kết quả chỉ của 3 agent đã chọn.
        obs = {agent: obs[agent] for agent in self.agents}
        rewards = {agent: rewards[agent] for agent in self.agents}
        # Gộp terminated và truncated thành 1 flag done
        done_flags = {agent: terminated[agent] or truncated[agent] for agent in self.agents}
        infos = {agent: infos[agent] for agent in self.agents}
        return obs, rewards, done_flags, infos

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        self.env.close()
