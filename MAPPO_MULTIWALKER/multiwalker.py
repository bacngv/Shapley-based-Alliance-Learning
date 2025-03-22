import gymnasium as gym
import numpy as np
from pettingzoo.sisl import multiwalker_v9

class GymMultiWalkerWrapper(gym.Env):
    """
    Wrapper to adapt PettingZoo's multiwalker environment to the Gym interface for multiple agents.
    Uses 3 agents: 'walker_0', 'walker_1', 'walker_2'.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode="rgb_array", **kwargs):
        # Add the render_mode parameter when initializing the environment
        self.env = multiwalker_v9.parallel_env(render_mode=render_mode,forward_reward=50.0, **kwargs)
        # Retrieve the list of agents, using the first three agents
        self.agents = self.env.possible_agents[:3]
        # Initial reset to establish observation and action spaces
        obs, _ = self.env.reset()
        self.observation_space = self.env.observation_space(self.agents[0])
        self.action_space = self.env.action_space(self.agents[0])

    def reset(self, seed=None, options=None):
        # Reset the environment and retrieve initial observations and info
        result = self.env.reset(seed=seed, options=options)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        # Filter observations for the selected agents
        obs = {agent: obs[agent] for agent in self.agents}
        return obs, info

    def step(self, actions):
        # Take a step in the environment using the given actions
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        # Filter data for the selected agents
        obs = {agent: obs[agent] for agent in self.agents}
        rewards = {agent: rewards[agent] for agent in self.agents}
        # Combine terminated and truncated flags into a single done flag
        done_flags = {agent: terminated[agent] or truncated[agent] for agent in self.agents}
        infos = {agent: infos[agent] for agent in self.agents}
        return obs, rewards, done_flags, infos

    def render(self, mode="human"):
        # Render the environment
        return self.env.render()

    def close(self):
        # Close the environment
        self.env.close()
