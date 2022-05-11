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
    CUDA_VISIBLE_DEVICES=0 python3 train/train_smac.py \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --num_agents ${num_agents} --use_linear_lr_decay \
    --num_mini_batch 1 --num_env_steps 100000000 --ppo_epoch 5 --gamma 0.99 0.99 \
    --wandb_name "cwz19" --user_name "cwz19" --use_value_active_masks \
    --n_rollout_threads 30 --n_eval_rollout_threads 30 --episode_length 400 \
    --lr 5e-4 --critic_lr 5e-4 --num_mini_batch 4 \
    --max_z 3 --div_thresh 1.5 --rex_thresh 7.5 --discri_lr 1e-4 --alpha 2.5 \
    --use_eval
done

# 12.5, 7.5