#!/bin/sh
env="MPE"
scenario="simple_spread"  #simple_speaker_listener  # simple_reference
num_landmarks=3
num_agents=3
algo="rmappo"
exp="0408_check"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=3 python3 train/train_mpe.py --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} \
    --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 1 \
    --n_rollout_threads 256 --n_eval_rollout_threads 64 --episode_length 25 \
    --num_env_steps 20000000 --ppo_epoch 10 --use_ReLU \
    --wandb_name "cwz19" --user_name "cwz19" \
    --use_eval
done