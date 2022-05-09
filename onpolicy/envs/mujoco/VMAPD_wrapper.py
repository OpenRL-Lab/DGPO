import numpy as np
import copy
import gym
from gym import spaces

class VMAPDWrapper(object):

    def __init__(self, env, max_z, fix_z):
        super().__init__()
        self.env = env
        self.max_z = max_z
        self.fix_z = fix_z
        self.cur_z = -1
        self.num_agents = 1
        self.z_space = [spaces.Discrete(self.max_z) for _ in range(self.num_agents)]
        self.z_obs_space = [copy.copy(self.env.observation_space[0])]
        self.z_local_obs_space = [copy.copy(self.env.observation_space[0])]
        self.observation_space = [copy.copy(self.env.observation_space[0])]
        self.share_observation_space = [copy.copy(self.env.observation_space[0])]
        self.action_space = self.env.action_space
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
        z_vec = np.expand_dims(z_vec, 0)
        z_vec = z_vec.repeat(self.num_agents, 0)
        obs_n = np.concatenate([z_vec, np.array(obs_n)], -1)
        return obs_n
    
    def step(self, actions):
        obs_n, reward_n, done_n, info_n = self.env.step(actions)
        z_vec = np.eye(self.max_z)[self.cur_z]
        z_vec = np.expand_dims(z_vec, 0)
        z_vec = z_vec.repeat(self.num_agents, 0)
        obs_n = np.concatenate([z_vec, np.array(obs_n)], -1)
        return obs_n, reward_n, done_n, info_n

    def seed(self, seed):
        self.env.seed(seed)

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        img = self.env.render(mode)
        return img