#!/bin/sh
env="StarCraft2"
map="2c_vs_64zg" #"3m"
algo="ours"
exp="exp"
num_agents=2
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    my_seed=1
    CUDA_VISIBLE_DEVICES=3 python3 train/train_smac.py \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${my_seed} --num_agents ${num_agents} --n_training_threads 1 \
    --num_mini_batch 1 --num_env_steps 200000000 --ppo_epoch 10 --gamma 0.99 0.75 \
    --use_ReLU --use_value_active_masks --wandb_name "cwz19" --user_name "cwz19" \
    --n_rollout_threads 64 --n_eval_rollout_threads 32 --episode_length 400 \
    --max_z 2 --div_thresh 1.25 --rex_thresh 1e6 \
    --discri_lr 1e-4 --alpha 1. \
    --use_eval
done
