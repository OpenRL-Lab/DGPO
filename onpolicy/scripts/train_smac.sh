#!/bin/sh
env="StarCraft2"
map="3m"
algo="rmappo"
exp="vmapd_local"
num_agents=3
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python3 train/train_smac.py \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --num_agents ${num_agents} --n_training_threads 1 \
    --num_mini_batch 1 --num_env_steps 200000000 --ppo_epoch 10  \
    --use_ReLU --use_value_active_masks --wandb_name "cwz19" --user_name "cwz19" \
    --n_rollout_threads 32 --n_eval_rollout_threads 32 --episode_length 400 \
    --use_eval 
done
