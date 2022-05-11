import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
import wandb
import imageio
import cv2
import matplotlib
import matplotlib.pyplot as plt
from collections import deque

def _t2n(x):
    return x.detach().cpu().numpy()

class MujocoRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(MujocoRunner, self).__init__(config)
        self.episode_rewards = []

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        self.episode_rewards = deque(maxlen=self.episode_length)

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
                data['share_obs'] = obs.copy()
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

                
                if infos is not None:
                    for info in infos:
                        if 'episode' in info[0].keys():
                            self.episode_rewards.append(info[0]['episode']['r'])
                

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
                                
                if train_infos:
                    train_infos["FPS"] = int(total_num_steps / (end - start))
                    if len(self.episode_rewards) > 0:
                        train_infos["episode_rewards_mean"] = np.mean(self.episode_rewards)
                        train_infos["episode_rewards_median"] = np.median(self.episode_rewards)
                        train_infos["episode_rewards_min"] = np.min(self.episode_rewards)
                        train_infos["episode_rewards_max"] = np.max(self.episode_rewards)
                        print(
                            "mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                                .format(
                                    np.mean(self.episode_rewards),
                                    np.median(self.episode_rewards), 
                                    np.min(self.episode_rewards),
                                    np.max(self.episode_rewards)
                                )
                        )
                self.log_train(train_infos, total_num_steps)


            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        
        obs = self.envs.reset()
        share_obs = obs.copy()
        
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
            # isTrain=False,
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
        actions_env = actions

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

    @torch.no_grad()
    def eval(self, total_num_steps):

        eval_episode_rewards = []

        seed_num = np.arange(self.n_eval_rollout_threads) // self.max_z 
        z_num = np.arange(self.n_eval_rollout_threads) % self.max_z

        eval_obs = self.eval_envs.seed(seed_num.astype('int'))
        eval_obs = self.eval_envs.reset(z_num)
        
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        finish_time_step = np.zeros(self.n_eval_rollout_threads)

        for eval_step in range(self.episode_length*10):

            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True
            )

            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            eval_actions_env = eval_actions

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards.flatten()*(finish_time_step==0))

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
            finish_time_step += eval_dones.all(-1) * (finish_time_step==0) * eval_step

            if (finish_time_step>0).all():
                break

        eval_episode_rewards = np.array(eval_episode_rewards).sum(0)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.mean(eval_episode_rewards)
        eval_average_episode_rewards = eval_env_infos['eval_average_episode_rewards']
        eval_env_infos['eval_average_episode_length'] = np.mean(finish_time_step)
        eval_average_episode_length = eval_env_infos['eval_average_episode_length']
        print("eval average episode rewards {:.4f} eval_average_episode_length: {:.1f}"
            .format(
                eval_average_episode_rewards,
                eval_average_episode_length
            )
        )
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        
        import mujoco_py
        from pyvirtualdisplay import Display

        disp = Display()
        disp.start()

        envs = self.envs
        all_frames = []
        for z in range(self.max_z):

            self.envs.seed(self.seed)
            obs = envs.reset(z)

            if self.all_args.save_gifs:
                image = envs.render(mode='rgb_array')
                # cv2.putText(image, str(z), (5, 25), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,0,0), 3)
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            episode_rewards = []
            
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

                actions_env = actions   

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render(mode='rgb_array')
                    all_frames.append(image)
                else:
                    envs.render('human')
                
                if dones.all():
                    break
            
            avg_rewards = np.sum(np.array(episode_rewards))
            print("average episode rewards is: " + str(avg_rewards))

        if self.all_args.save_gifs:
            video_dir = str(self.gif_dir) + '/render.avi'
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            all_frames = np.concatenate(all_frames, 0)
            w, h, c = all_frames[0].shape
            gout = cv2.VideoWriter(video_dir, fourcc, 50.0, (h,w), True)
            for frame in all_frames:
                gout.write(np.uint8(frame[:,:,::-1]))
            gout.release()
        
        disp.stop()

