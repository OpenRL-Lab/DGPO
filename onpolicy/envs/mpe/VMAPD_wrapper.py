import numpy as np
import copy
import gym
from gym import spaces

class VMAPDWrapper(gym.Wrapper):

    def __init__(self, env, max_z, fix_z):
        gym.Wrapper.__init__(self, env)
        self.max_z = max_z
        self.fix_z = fix_z
        self.cur_z = -1
        self.num_agents = self.env.n
        self.z_space = [spaces.Discrete(self.max_z) for _ in range(self.num_agents)]
        self.z_obs_space = [copy.copy(self.share_observation_space[i]) for i in range(self.num_agents)]
        self.z_local_obs_space = [copy.copy(self.observation_space[i]) for i in range(self.num_agents)]
        for observation_space in self.observation_space:
            observation_space.shape = (observation_space.shape[0] + self.max_z,)
        for observation_space in self.share_observation_space:
            observation_space.shape = (observation_space.shape[0] + self.max_z,)
    
    def reset(self, fix_z=None):
        if fix_z is not None:
            self.cur_z = fix_z
        elif self.fix_z is not None:
            self.cur_z = self.fix_z
        else:
            self.cur_z = np.random.randint(self.max_z) 
        obs_n = self.env.reset()
        z_vec = np.eye(self.max_z)[self.cur_z]
        for a_id in range(self.num_agents):
            obs_n[a_id] = np.concatenate([z_vec, obs_n[a_id]])
        return obs_n
    
    def step(self, actions):
        obs_n, reward_n, done_n, info_n = self.env.step(actions)
        z_vec = np.eye(self.max_z)[self.cur_z]
        for a_id in range(self.num_agents):
            obs_n[a_id] = np.concatenate([z_vec, obs_n[a_id]])
        return obs_n, reward_n, done_n, info_n
