#!/bin/sh
env="StarCraft2"
map="3m"
algo="rmappo"
exp="vmapd_share"
num_agents=3
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python3 train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --num_agents ${num_agents} --n_training_threads 1 --n_rollout_threads 32 \
    --num_mini_batch 1 --episode_length 400 --num_env_steps 10000000 --ppo_epoch 10 --use_ReLU --gain 0.01 \
    --lr 7e-4 --critic_lr 7e-4 --wandb_name "cwz19" --user_name "cwz19" \
    --use_eval #--use_wandb # 
done
