import torch
import copy
import numpy as np

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_ex_Critic, R_in_Critic, R_Discriminator, AlphaModel
from onpolicy.utils.util import update_linear_schedule


class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu"), \
        z_space=None, z_obs_space=None, z_local_obs_space=None):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.max_z = args.max_z
        self.num_agents = args.num_agents

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        self.z_space = z_space
        self.z_obs_space = z_obs_space
        self.z_local_obs_space = z_local_obs_space

        self.alpha_model = AlphaModel(args) 
        self.alpha_optimizer = torch.optim.Adam(self.alpha_model.parameters(),
                                                    lr=1e-5, eps=self.opti_eps,
                                                    weight_decay=self.weight_decay)
        

        self.discriminator = R_Discriminator(args, self.z_obs_space, self.z_space, self.device) 
        self.discri_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                                    lr=self.lr, eps=self.opti_eps,
                                                    weight_decay=self.weight_decay)
        
        self.local_discri = R_Discriminator(args, self.z_local_obs_space, self.z_space, self.device) 
        self.local_discri_optimizer = torch.optim.Adam(self.local_discri.parameters(),
                                                    lr=self.lr, eps=self.opti_eps,
                                                    weight_decay=self.weight_decay)

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)

        self.ex_critic = R_ex_Critic(args, self.share_obs_space, self.device)
        self.ex_critic_optimizer = torch.optim.Adam(self.ex_critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        
        
        self.in_critic = R_in_Critic(args, self.share_obs_space, self.device)
        self.in_critic_optimizer = torch.optim.Adam(self.in_critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.discri_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.local_discri_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.ex_critic_optimizer, episode, episodes, self.critic_lr)
        update_linear_schedule(self.in_critic_optimizer, episode, episodes, self.critic_lr)
        update_linear_schedule(self.alpha_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_ex_critic, rnn_states_in_critic, \
                        masks, available_actions=None, deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        actions, action_log_probs, rnn_states_actor = \
            self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)

        ex_values, rnn_states_ex_critic = self.ex_critic(cent_obs, rnn_states_ex_critic, masks)
        in_values, rnn_states_in_critic = self.in_critic(cent_obs, rnn_states_in_critic, masks)

        return ex_values, in_values, actions, action_log_probs, rnn_states_actor, rnn_states_ex_critic, rnn_states_in_critic

    def get_values(self, cent_obs, rnn_states_ex_critic, rnn_states_in_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        ex_values, _ = self.ex_critic(cent_obs, rnn_states_ex_critic, masks)
        in_values, _ = self.in_critic(cent_obs, rnn_states_in_critic, masks)

        return ex_values, in_values

    def get_ex_values(self, cent_obs, rnn_states_ex_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        ex_values, _ = self.ex_critic(cent_obs, rnn_states_ex_critic, masks)

        return ex_values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_ex_critic, rnn_states_in_critic, \
                            action, masks, available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """

        action_log_probs, dist_entropy = \
            self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, available_actions, active_masks)

        ex_values, _ = self.ex_critic(cent_obs, rnn_states_ex_critic, masks)
        in_values, _ = self.in_critic(cent_obs, rnn_states_in_critic, masks)

        return ex_values, in_values, action_log_probs, dist_entropy

    def evaluate_z(self, cent_obs, rnn_states_z, masks, active_masks=None, isTrain=True):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        z_vec = cent_obs[:,:self.max_z]
        z_idx = np.argmax(z_vec, axis=1)
        z_idxs = np.expand_dims(z_idx, -1)
        cent_obs = cent_obs[:,self.max_z:]

        if isTrain:
            z_masks = None
        else:
            z_masks = np.ones([self.max_z, self.max_z])
            z_masks = np.tril(z_masks)
            z_masks = z_masks[z_idx]

        action_log_probs, rnn_states_z = \
            self.discriminator.evaluate_actions(cent_obs, rnn_states_z, z_idxs, masks, available_actions=z_masks, active_masks=active_masks)

        return action_log_probs, rnn_states_z

    def evaluate_local_z(self, obs, rnn_states_z, masks, active_masks=None, isTrain=True):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        z_idx = np.argmax(obs[:,:self.max_z], axis=1)
        z_idxs = np.expand_dims(z_idx, -1)
        obs = obs[:,self.max_z:]

        if isTrain:
            z_masks = None
        else:
            z_masks = np.ones([self.max_z, self.max_z])
            z_masks = np.tril(z_masks)
            z_masks = z_masks[z_idx]

        action_log_probs, rnn_states_z = \
            self.local_discri.evaluate_actions(obs, rnn_states_z, z_idxs, masks, available_actions=z_masks, active_masks=active_masks)

        return action_log_probs, rnn_states_z

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor

