#!/bin/sh
env="MPE"
scenario="simple_spread_one_fix" 
num_landmarks=4
num_agents=1
algo="MAPPO" #"DIAYN_1"  #"ours"
exp="exp"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python3 train/train_mpe.py --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} \
    --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 1 \
    --n_rollout_threads 4 --n_eval_rollout_threads 2 --episode_length 15 \
    --num_env_steps 1000000 --ppo_epoch 10 --use_ReLU --gamma 0.99 0.99 \
    --wandb_name "cwz19" --user_name "cwz19" --eval_interval 15 \
    --max_z 2 --div_thresh 1.1 --rex_thresh -2.5 \
    --discri_lr 1e-4 --alpha 2.5 \
    --use_wandb
done