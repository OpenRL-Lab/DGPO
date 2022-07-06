#!/bin/sh
env="StarCraft2"
map="3m" #"3m" "2s_vs_1sc"
algo="ours"
exp="exp"
num_agents=3
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    seed=6
    CUDA_VISIBLE_DEVICES=0 python3 train/train_smac.py \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --num_agents ${num_agents} --use_linear_lr_decay \
    --num_env_steps 5000000 --ppo_epoch 10 --gamma 0.99 0.75 \
    --wandb_name "cwz19" --user_name "cwz19" --use_value_active_masks \
    --n_rollout_threads 27 --n_eval_rollout_threads 18 --episode_length 400 \
    --lr 5e-4 --critic_lr 5e-4 --num_mini_batch 1 --sdpo_entropy_coeff 1e-2 \
    --max_z 3 --div_thresh 1.25 --rex_thresh 13.5 --discri_lr 1e-4 --alpha 2.5 \
    --use_eval --eval_interval 15
done

# 11.5, 13.5

