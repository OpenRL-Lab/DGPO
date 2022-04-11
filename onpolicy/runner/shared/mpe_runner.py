import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
import wandb
import imageio
import cv2
import matplotlib
import matplotlib.pyplot as plt

def _t2n(x):
    return x.detach().cpu().numpy()

class MPERunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(MPERunner, self).__init__(config)

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):

            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):

                # Sample actions
                ex_values, in_values, actions, action_log_probs, rnn_states, \
                    rnn_states_ex_critic, rnn_states_in_critic, actions_env = self.collect(step)
                    
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                # # soft learning
                # rewards -= action_log_probs

                # insert data into buffer
                data = dict()
                data['obs'] = obs
                data['share_obs'] = self.obs2shareobs(obs.copy())
                data['rnn_states_actor'] = rnn_states
                data['rnn_states_ex_critic'] = rnn_states_ex_critic
                data['rnn_states_in_critic'] = rnn_states_in_critic
                data['actions'] = actions
                data['action_log_probs'] = action_log_probs
                data['ex_value_preds'] = ex_values
                data['in_value_preds'] = in_values
                data['rewards'] = rewards
                data['dones'] = dones
                self.insert(data, step)

                # VMAPD
                z_log_probs, loc_z_log_probs, rnn_states_z, loc_rnn_states_z = self.VMAPD_collect(step)
                data = dict()
                data['rnn_states_z'] = rnn_states_z
                data['loc_rnn_states_z'] = loc_rnn_states_z
                data['z_log_probs'] = z_log_probs
                data['loc_z_log_probs'] = loc_z_log_probs
                data['dones'] = dones
                self.insert(data, step)
                

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "MPE":
                    env_infos = {}
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            if 'individual_reward' in info[agent_id].keys():
                                idv_rews.append(info[agent_id]['individual_reward'])
                        agent_k = 'agent%i/individual_rewards' % agent_id
                        env_infos[agent_k] = idv_rews

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        
        obs = self.envs.reset()
        share_obs = self.obs2shareobs(obs)
        
        self.buffer.obs[0] = obs.copy()
        self.buffer.share_obs[0] = share_obs.copy()

    @torch.no_grad()
    def VMAPD_collect(self, step):
        self.trainer.prep_rollout()
        z_log_prob, rnn_state_z = self.trainer.policy.evaluate_z(
            np.concatenate(self.buffer.share_obs[step+1]),
            np.concatenate(self.buffer.rnn_states_z[step]),
            np.concatenate(self.buffer.masks[step+1]),
            isTrain=False,
        )
        loc_z_log_prob, loc_rnn_state_z = self.trainer.policy.evaluate_local_z(
            np.concatenate(self.buffer.obs[step+1]),
            np.concatenate(self.buffer.loc_rnn_states_z[step]),
            np.concatenate(self.buffer.masks[step+1]),
            isTrain=False,
        )
        # [self.envs, agents, dim]
        z_log_probs = np.array(np.split(_t2n(z_log_prob), self.n_rollout_threads))
        rnn_states_z = np.array(np.split(_t2n(rnn_state_z), self.n_rollout_threads))
        loc_z_log_probs = np.array(np.split(_t2n(loc_z_log_prob), self.n_rollout_threads))
        loc_rnn_states_z = np.array(np.split(_t2n(loc_rnn_state_z), self.n_rollout_threads))

        return z_log_probs, loc_z_log_probs, rnn_states_z, loc_rnn_states_z

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        ex_value, in_value, action, action_log_prob, \
            rnn_states, rnn_states_ex_critic, rnn_states_in_critic \
                = self.trainer.policy.get_actions(
                    np.concatenate(self.buffer.share_obs[step]),
                    np.concatenate(self.buffer.obs[step]),
                    np.concatenate(self.buffer.rnn_states[step]),
                    np.concatenate(self.buffer.rnn_states_ex_critic[step]),
                    np.concatenate(self.buffer.rnn_states_in_critic[step]),
                    np.concatenate(self.buffer.masks[step])
                )
        # [self.envs, agents, dim]
        ex_values = np.array(np.split(_t2n(ex_value), self.n_rollout_threads))
        in_values = np.array(np.split(_t2n(in_value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_ex_critic = np.array(np.split(_t2n(rnn_states_ex_critic), self.n_rollout_threads))
        rnn_states_in_critic = np.array(np.split(_t2n(rnn_states_in_critic), self.n_rollout_threads))
        # rearrange action
        actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)

        return ex_values, in_values, actions, action_log_probs, rnn_states,\
                    rnn_states_ex_critic, rnn_states_in_critic, actions_env

    def insert(self, data, step):    
        
        dones = (data['dones']==True)
        if 'rnn_states_actor' in data:
            data['rnn_states_actor'][dones] = \
                np.zeros(((dones).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        if 'rnn_states_ex_critic' in data:
            data['rnn_states_ex_critic'][dones] = \
                np.zeros(((dones).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        if 'rnn_states_in_critic' in data:
            data['rnn_states_in_critic'][dones] = \
                np.zeros(((dones).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        if 'rnn_states_z' in data:
            data['rnn_states_z'][dones] = \
                np.zeros(((dones).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        if 'loc_rnn_states_z' in data:
            data['loc_rnn_states_z'][dones] = \
                np.zeros(((dones).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)

        data['masks'] = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        data['masks'][dones] = np.zeros(((dones).sum(), 1), dtype=np.float32)
        
        self.buffer.insert(data, step)
    
    def obs2shareobs(self, obs):
        # input : [n_rollout, n_agents, z_num+obs_size]
        # output : [n_rollout, n_agents, z_num+obs_size*n_agents]
        z_vec = obs[:,:,:self.max_z]
        o_vec = obs[:,:,self.max_z:]
        share_obs = o_vec.reshape([self.n_rollout_threads, -1])
        share_obs = np.expand_dims(share_obs, 1)
        share_obs = share_obs.repeat(self.num_agents, axis=1)
        share_obs = np.concatenate([z_vec, share_obs], -1)
        return share_obs

    @torch.no_grad()
    def eval(self, total_num_steps):

        eval_episode_rewards = []
        eval_all_obs = []

        seed_num = np.arange(self.n_eval_rollout_threads//self.max_z)
        seed_num = np.expand_dims(seed_num, 1)
        seed_num = seed_num.repeat(self.max_z, 1)
        seed_num = seed_num.flatten()
        eval_obs = self.eval_envs.seed(seed_num)
        eval_obs = self.eval_envs.reset()
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):

            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True
            )

            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            if self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)
            eval_all_obs.append(eval_obs[:,:,self.max_z+2:self.max_z+4])

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        eval_all_obs = np.stack(eval_all_obs, 1)
        eval_all_obs = eval_all_obs.reshape([-1, self.max_z, self.episode_length*self.num_agents*2])
        eval_all_obs = eval_all_obs.transpose(1,0,2)
        eval_all_obs = eval_all_obs.reshape([self.max_z,-1])
        eval_dis_mat = np.expand_dims(eval_all_obs, 0) - np.expand_dims(eval_all_obs, 1)
        eval_dis_mat = np.mean(eval_dis_mat**2, -1)
        print("eval distance matrix: " + str(eval_dis_mat.mean()))
        eval_env_infos['eval_pos_distance'] = eval_dis_mat.mean()
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""

        envs = self.envs
        all_frames = []
        all_traj = []

        for z in range(self.max_z):

            self.envs.seed(seed=self.seed)
            obs = envs.reset(z)

            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                cv2.putText(image, str(z), (5, 25), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,0,0), 3)
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            episode_rewards = []
            z_traj = []
            
            for _ in range(self.episode_length):

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True
                )
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)
                z_traj.append(obs[0,:,self.max_z+2:self.max_z+4])

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    cv2.putText(image, str(z), (5, 25), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,0,0), 3)
                    all_frames.append(image)
                else:
                    envs.render('human')

            avg_rewards = np.mean(np.sum(np.array(episode_rewards), axis=0))
            print("average episode rewards is: " + str(avg_rewards))
            all_traj.append(z_traj)

        if self.all_args.save_gifs:
            # save gif
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
            # save heat map
            all_traj = np.array(all_traj).reshape([self.max_z, -1])
            dis_matrix = np.expand_dims(all_traj,0) - np.expand_dims(all_traj,1)
            dis_matrix = np.mean(dis_matrix**2, -1)

            fig, ax = plt.subplots()
            im = ax.imshow(dis_matrix)

            # Loop over data dimensions and create text annotations.
            for i in range(self.max_z):
                for j in range(self.max_z):
                    ax.text(j, i, "{:.4f}".format(dis_matrix[i, j]), 
                                ha="center", va="center", color="w")

            ax.set_title("distance_matrix")
            fig.tight_layout()
            plt.savefig(str(self.gif_dir) + "/distance_matrix.png")
            plt.close()

