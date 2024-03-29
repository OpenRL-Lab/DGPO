import torch
import numpy as np
from onpolicy.utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


class SharedReplayBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_space: (gym.Space) observation space of agents.
    :param cent_obs_space: (gym.Space) centralized observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """

    def __init__(self, args, num_agents, obs_space, cent_obs_space, act_space):

        # parameters
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits
        self.num_agents = args.num_agents
        self.max_z = args.max_z

        # shapes
        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)
        act_shape = get_shape_from_act_space(act_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        # data : obs
        self.share_obs = np.zeros((self.episode_length+1, self.n_rollout_threads, num_agents, *share_obs_shape), dtype=np.float32)
        self.obs = np.zeros((self.episode_length+1, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)

        # data : rnn_state
        r_shape = (self.episode_length+1, self.n_rollout_threads, num_agents, self.recurrent_N, self.hidden_size)
        self.rnn_states = np.zeros(r_shape,dtype=np.float32)
        self.rnn_states_ex_critic = np.zeros(r_shape,dtype=np.float32)
        self.rnn_states_in_critic = np.zeros(r_shape,dtype=np.float32)
        self.rnn_states_z = np.zeros(r_shape,dtype=np.float32)
        self.loc_rnn_states_z = np.zeros(r_shape,dtype=np.float32)

        # data : values
        v_shape = (self.episode_length+1, self.n_rollout_threads, num_agents, 1)
        self.ex_value_preds = np.zeros(v_shape, dtype=np.float32)
        self.in_value_preds = np.zeros(v_shape, dtype=np.float32)
        self.ex_returns = np.zeros(v_shape, dtype=np.float32)
        self.in_returns = np.zeros(v_shape, dtype=np.float32)

        # data : actions
        a_shape = (self.episode_length, self.n_rollout_threads, num_agents, act_shape)
        self.actions = np.zeros(a_shape, dtype=np.float32)
        self.action_log_probs = np.zeros(a_shape, dtype=np.float32)

        # data : discriminator
        d_shape = (self.episode_length, self.n_rollout_threads, num_agents, 1)
        self.z_log_probs = np.zeros(d_shape, dtype=np.float32)
        self.loc_z_log_probs = np.zeros(d_shape, dtype=np.float32)

        # data : rewards
        r_shape = (self.episode_length, self.n_rollout_threads, num_agents, 1)
        self.rewards = np.zeros(r_shape, dtype=np.float32)

        # data : masks
        m_shape = (self.episode_length+1, self.n_rollout_threads, num_agents, 1)
        self.masks = np.ones(m_shape, dtype=np.float32)
        self.bad_masks = np.ones(m_shape, dtype=np.float32)
        self.active_masks = np.ones(m_shape, dtype=np.float32)
        if act_space.__class__.__name__ == 'Discrete':
            a_shape = (self.episode_length+1, self.n_rollout_threads, num_agents, act_space.n)
            self.available_actions = np.ones(a_shape, dtype=np.float32)
        else:
            self.available_actions = None

        self.step = 0

    def insert(self, data, step):
        
        self.step = step

        key2var = {
            'share_obs':[self.share_obs,1],
            'obs':[self.obs,1],
            'rnn_states_actor':[self.rnn_states,1],
            'rnn_states_ex_critic':[self.rnn_states_ex_critic,1],
            'rnn_states_in_critic':[self.rnn_states_in_critic,1],
            'rnn_states_z':[self.rnn_states_z,1],
            'actions':[self.actions,0],
            'action_log_probs':[self.action_log_probs,0],
            'z_log_probs':[self.z_log_probs,0],
            'ex_value_preds':[self.ex_value_preds,0],
            'in_value_preds':[self.in_value_preds,0],
            'rewards':[self.rewards,0],
            'masks':[self.masks,1],
            'bad_masks':[self.bad_masks,1],
            'active_masks':[self.active_masks,1],
            'available_actions':[self.available_actions,1],
        }

        for key in data:
            if key not in ['dones', 'infos']:
                var = key2var[key][0]
                idx = key2var[key][1] + self.step
                var[idx] = data[key].copy()


    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states_z[0] = self.rnn_states_z[-1].copy()
        self.loc_rnn_states_z[0] = self.loc_rnn_states_z[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_ex_critic[0] = self.rnn_states_ex_critic[-1].copy()
        self.rnn_states_in_critic[0] = self.rnn_states_in_critic[-1].copy()
        self.z_log_probs[0] = self.z_log_probs[-1].copy()
        self.loc_z_log_probs[0] = self.loc_z_log_probs[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def compute_returns(self, next_ex_value, next_in_value, ex_value_normalizer=None, in_value_normalizer=None):
        
        # extrinsic return 
        gae = 0
        self.ex_value_preds[-1] = next_ex_value
        for step in reversed(range(self.rewards.shape[0])):
            value_tp1 = ex_value_normalizer.denormalize(self.ex_value_preds[step+1])
            value_t = ex_value_normalizer.denormalize(self.ex_value_preds[step])
            delta = self.rewards[step] + self.gamma[0] * value_tp1 * self.masks[step+1] - value_t
            gae = delta + self.gamma[0] * self.gae_lambda * self.masks[step+1] * gae
            self.ex_returns[step] = gae + value_t

        # intrinsic return 
        gae = 0
        self.in_value_preds[-1] = next_in_value
        for step in reversed(range(self.z_log_probs.shape[0])):
            # loc_r = np.mean(self.loc_z_log_probs[step], axis=1, keepdims=True).repeat(self.num_agents,1)
            rewards = self.z_log_probs[step] #* 2. - loc_r
            value_tp1 = in_value_normalizer.denormalize(self.in_value_preds[step+1])
            value_t = in_value_normalizer.denormalize(self.in_value_preds[step])
            delta = rewards + self.gamma[1] * value_tp1 * self.masks[step+1] - value_t
            gae = delta + self.gamma[1] * self.gae_lambda * self.masks[step+1] * gae
            self.in_returns[step] = gae + value_t
  
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        # step + 1
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * gae * self.masks[step + 1]
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                                self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(
                            self.value_preds[step])
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                                self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]
        """ 

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) * number of agents ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, num_agents,
                          n_rollout_threads * episode_length * num_agents,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[3:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N M Dim]-->[T N M Dim]-->[T*N*M,Dim]-->[index,Dim]
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,\
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,\
                  adv_targ, available_actions_batch

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        """
        Yield training data for non-chunked RNN training.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * num_agents
        assert n_rollout_threads * num_agents >= num_mini_batch, (
            "PPO requires the number of processes ({})* number of agents ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, num_agents, num_mini_batch))
        num_envs_per_batch = batch_size // num_mini_batch
        perm = torch.randperm(batch_size).numpy()

        share_obs = self.share_obs.reshape(-1, batch_size, *self.share_obs.shape[3:])
        obs = self.obs.reshape(-1, batch_size, *self.obs.shape[3:])
        rnn_states = self.rnn_states.reshape(-1, batch_size, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic.reshape(-1, batch_size, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(-1, batch_size, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions.reshape(-1, batch_size, self.available_actions.shape[-1])
        value_preds = self.value_preds.reshape(-1, batch_size, 1)
        returns = self.returns.reshape(-1, batch_size, 1)
        masks = self.masks.reshape(-1, batch_size, 1)
        active_masks = self.active_masks.reshape(-1, batch_size, 1)
        action_log_probs = self.action_log_probs.reshape(-1, batch_size, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, batch_size, 1)

        for start_ind in range(0, batch_size, num_envs_per_batch):
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                share_obs_batch.append(share_obs[:-1, ind])
                obs_batch.append(obs[:-1, ind])
                rnn_states_batch.append(rnn_states[0:1, ind])
                rnn_states_critic_batch.append(rnn_states_critic[0:1, ind])
                actions_batch.append(actions[:, ind])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[:-1, ind])
                value_preds_batch.append(value_preds[:-1, ind])
                return_batch.append(returns[:-1, ind])
                masks_batch.append(masks[:-1, ind])
                active_masks_batch.append(active_masks[:-1, ind])
                old_action_log_probs_batch.append(action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            # [N[T, dim]]
            T, N = self.episode_length, num_envs_per_batch
            # These are all from_numpys of size (T, N, -1)
            share_obs_batch = np.stack(share_obs_batch, 1)
            obs_batch = np.stack(obs_batch, 1)
            actions_batch = np.stack(actions_batch, 1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, 1)
            value_preds_batch = np.stack(value_preds_batch, 1)
            return_batch = np.stack(return_batch, 1)
            masks_batch = np.stack(masks_batch, 1)
            active_masks_batch = np.stack(active_masks_batch, 1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, 1)
            adv_targ = np.stack(adv_targ, 1)

            # States is just a (N, dim) from_numpy [N[1,dim]]
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            # Flatten the (T, N, ...) from_numpys to (T * N, ...)
            share_obs_batch = _flatten(T, N, share_obs_batch)
            obs_batch = _flatten(T, N, obs_batch)
            actions_batch = _flatten(T, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(T, N, old_action_log_probs_batch)
            adv_targ = _flatten(T, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,\
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,\
                  adv_targ, available_actions_batch

    def recurrent_generator(self, ex_advantages, in_advantages, num_mini_batch, data_chunk_length):
        """
        Yield training data for chunked RNN training.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param data_chunk_length: (int) length of sequence chunks with which to train RNN.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        if len(self.share_obs.shape) > 4:
            share_obs = self.share_obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.share_obs.shape[3:])
            obs = self.obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.obs.shape[3:])
        else:
            share_obs = _cast(self.share_obs[:-1])
            obs = _cast(self.obs[:-1])

        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        ex_advantages = _cast(ex_advantages)
        in_advantages = _cast(in_advantages)
        ex_value_preds = _cast(self.ex_value_preds[:-1])
        in_value_preds = _cast(self.in_value_preds[:-1])
        ex_returns = _cast(self.ex_returns[:-1])
        in_returns = _cast(self.in_returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])
        rnn_states = \
            self.rnn_states[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_z = \
            self.rnn_states_z[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states_z.shape[3:])
        loc_rnn_states_z = \
            self.loc_rnn_states_z[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.loc_rnn_states_z.shape[3:])
        rnn_states_ex_critic = \
            self.rnn_states_ex_critic[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states_ex_critic.shape[3:])
        rnn_states_in_critic = \
            self.rnn_states_in_critic[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states_in_critic.shape[3:])

        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_z_batch = []
            loc_rnn_states_z_batch = []
            rnn_states_ex_critic_batch = []
            rnn_states_in_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            ex_value_preds_batch = []
            in_value_preds_batch = []
            ex_return_batch = []
            in_return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            ex_adv_targ = []
            in_adv_targ = []

            for index in indices:

                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                share_obs_batch.append(share_obs[ind:ind + data_chunk_length])
                obs_batch.append(obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind + data_chunk_length])
                ex_value_preds_batch.append(ex_value_preds[ind:ind + data_chunk_length])
                in_value_preds_batch.append(in_value_preds[ind:ind + data_chunk_length])
                ex_return_batch.append(ex_returns[ind:ind + data_chunk_length])
                in_return_batch.append(in_returns[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind + data_chunk_length])
                ex_adv_targ.append(ex_advantages[ind:ind + data_chunk_length])
                in_adv_targ.append(in_advantages[ind:ind + data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                loc_rnn_states_z_batch.append(loc_rnn_states_z[ind])
                rnn_states_z_batch.append(rnn_states_z[ind])
                rnn_states_ex_critic_batch.append(rnn_states_ex_critic[ind])
                rnn_states_in_critic_batch.append(rnn_states_in_critic[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim)           
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            obs_batch = np.stack(obs_batch, axis=1)

            actions_batch = np.stack(actions_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
            ex_value_preds_batch = np.stack(ex_value_preds_batch, axis=1)
            in_value_preds_batch = np.stack(in_value_preds_batch, axis=1)
            ex_return_batch = np.stack(ex_return_batch, axis=1)
            in_return_batch = np.stack(in_return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            ex_adv_targ = np.stack(ex_adv_targ, axis=1)
            in_adv_targ = np.stack(in_adv_targ, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_z_batch = np.stack(rnn_states_z_batch).reshape(N, *self.rnn_states_z.shape[3:])
            loc_rnn_states_z_batch = np.stack(loc_rnn_states_z_batch).reshape(N, *self.loc_rnn_states_z.shape[3:])
            rnn_states_ex_critic_batch = np.stack(rnn_states_ex_critic_batch).reshape(N, *self.rnn_states_ex_critic.shape[3:])
            rnn_states_in_critic_batch = np.stack(rnn_states_in_critic_batch).reshape(N, *self.rnn_states_in_critic.shape[3:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            share_obs_batch = _flatten(L, N, share_obs_batch)
            obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            ex_value_preds_batch = _flatten(L, N, ex_value_preds_batch)
            in_value_preds_batch = _flatten(L, N, in_value_preds_batch)
            ex_return_batch = _flatten(L, N, ex_return_batch)
            in_return_batch = _flatten(L, N, in_return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            ex_adv_targ = _flatten(L, N, ex_adv_targ)
            in_adv_targ = _flatten(L, N, in_adv_targ)

            yield share_obs_batch, obs_batch, \
                    rnn_states_batch, rnn_states_z_batch, loc_rnn_states_z_batch, \
                    rnn_states_ex_critic_batch, rnn_states_in_critic_batch, \
                    actions_batch, ex_value_preds_batch, in_value_preds_batch, \
                    ex_return_batch, in_return_batch, masks_batch, active_masks_batch, \
                    old_action_log_probs_batch, ex_adv_targ, in_adv_targ, available_actions_batch
