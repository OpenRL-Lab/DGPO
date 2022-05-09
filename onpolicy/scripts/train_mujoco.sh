#!/bin/sh
env="mujoco"
scenario="Hopper-v2" 
num_agents=1
algo="MAPPO"
exp="exp"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python3 train/train_mujoco.py --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} \
    --seed ${seed} --n_training_threads 1 --n_rollout_threads 128 --n_eval_rollout_threads 128 \
    --episode_length 2048 --num_env_steps 100000000000000 --ppo_epoch 10 --use_ReLU --gamma 0.999 0.99 \
    --wandb_name "cwz19" --user_name "cwz19" --log_interval 1 \
    --max_z 2 --div_thresh 1.1 --rex_thresh -2.5 \
    --discri_lr 1e-4 --alpha 0. --num_mini_batch 32 --use_eval
done