#!/bin/sh
env="MPE"
scenario="simple_speaker_listener" #"simple_spread"
num_landmarks=3
num_agents=2
algo="rmappo"
exp="check"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=1 python3 render/render_mpe.py --save_gifs --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} \
    --num_landmarks ${num_landmarks} --seed ${seed} --use_ReLU --gain 0.01 \
    --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 25 \
    --model_dir "results/MPE/simple_speaker_listener/rmappo/check/wandb/run-20220311_123034-2ryaryut/files" --use_wandb --share_policy
done