import numpy as np
import copy
import gym
from gym import spaces

class VMAPDWrapper(object):

    def __init__(self, env, max_z):
        super().__init__()
        self.env = env
        self.max_z = max_z
        self.cur_z = -1
        self.num_agents = self.env.n_agents
        self.z_space = [spaces.Discrete(self.max_z) for _ in range(self.num_agents)]
        self.z_obs_space = [copy.copy(self.env.share_observation_space[i]) for i in range(self.num_agents)]
        self.z_local_obs_space = [copy.copy(self.env.observation_space[i]) for i in range(self.num_agents)]
        self.observation_space = self.env.observation_space
        self.share_observation_space = self.env.share_observation_space
        self.action_space = self.env.action_space
        for observation_space in self.observation_space:
            observation_space[0] += self.max_z
        for observation_space in self.share_observation_space:
            observation_space[0] += self.max_z
    
    def reset(self, fix_z=None, **kwargs):
        if fix_z is not None:
            self.cur_z = fix_z
        else:
            self.cur_z = np.random.randint(self.max_z) 
        local_o, global_s, avail_actions = self.env.reset(**kwargs)
        z_vec = np.eye(self.max_z)[self.cur_z]
        for a_id in range(self.num_agents):
            local_o[a_id] = np.concatenate([z_vec, local_o[a_id]])
            global_s[a_id] = np.concatenate([z_vec, global_s[a_id]])
        return local_o, global_s, avail_actions
    
    def step(self, actions):
        local_o, global_s, reward, done, info, available_actions = self.env.step(actions)
        z_vec = np.eye(self.max_z)[self.cur_z]
        for a_id in range(self.num_agents):
            local_o[a_id] = np.concatenate([z_vec, local_o[a_id]])
            global_s[a_id] = np.concatenate([z_vec, global_s[a_id]])
        return local_o, global_s, reward, done, info, available_actions

    def seed(self, seed):
        self.env.seed(seed)
