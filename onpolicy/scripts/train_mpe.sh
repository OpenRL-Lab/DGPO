#!/bin/sh
env="MPE"
scenario="simple_spread_fix" 
num_landmarks=3
num_agents=3
algo="rmappo"
exp="0504_check"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    my_seed=1
    echo "seed is ${my_seed}:"
    CUDA_VISIBLE_DEVICES=0 python3 train/train_mpe.py --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} \
    --num_landmarks ${num_landmarks} --seed ${my_seed} --n_training_threads 1 \
    --n_rollout_threads 128 --n_eval_rollout_threads 64 --episode_length 25 \
    --num_env_steps 200000000 --ppo_epoch 10 --use_ReLU --gamma 0.99 0.75 \
    --wandb_name "cwz19" --user_name "cwz19" --eval_interval 15 \
    --use_eval --max_z 2 --div_thresh 1.25 --discri_lr 1e-4
done