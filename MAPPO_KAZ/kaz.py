import gym
from gym import spaces
import numpy as np
from pettingzoo.butterfly import knights_archers_zombies_v10

class KAZGymWrapper(gym.Env):
    """
    A wrapper for the PettingZoo Knights Archers Zombies environment that converts it to a Gym interface.
    
    This version supports multi-agent by returning dictionaries of observations, rewards, done flags, and info for all agents.
    """
    metadata = {"render.modes": ["human", "rgb_array"]}
    
    def __init__(self, render_mode="human", **kwargs):
        super(KAZGymWrapper, self).__init__()
        
        # Create the PettingZoo parallel environment
        self.env = knights_archers_zombies_v10.parallel_env(render_mode=render_mode, **kwargs)
        
        # Save the list of agents
        self.agents = self.env.possible_agents
        
        # Set the action and observation spaces as dictionaries for each agent
        self.action_space = self.env.action_spaces
        self.observation_space = self.env.observation_spaces
    
    def reset(self, **kwargs):
        """
        Resets the environment and returns a dictionary of observations for all agents.
        """
        observations = self.env.reset(**kwargs)
        # If reset returns a tuple (obs, info), unpack to get only the observations
        if isinstance(observations, tuple):
            observations, _ = observations
        return observations

    def step(self, actions):
        """
        Executes actions for all agents.
        
        Parameters:
            - actions: a dictionary containing the action for each agent
        
        Returns:
            - observations: a dictionary of observations for all agents
            - rewards: a dictionary of rewards for each agent
            - done: True if any agent reaches a terminated or truncated state
            - infos: a dictionary of additional information for each agent
        """
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        # End the game if any agent reaches a terminated or truncated state
        done = any(terminations.values()) or any(truncations.values())
        return observations, rewards, done, infos

    def render(self, mode="human"):
        return self.env.render()
    
    def close(self):
        self.env.close()
