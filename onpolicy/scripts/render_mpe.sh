#!/bin/sh
env="MPE"
scenario="simple_spread_fix" #"simple_speaker_listener"
num_landmarks=3
num_agents=3
algo="rmappo"
exp="check"
seed=1


CUDA_VISIBLE_DEVICES=3 python3 render/render_mpe.py --save_gifs --env_name ${env} --algorithm_name ${algo} \
--experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} \
--num_landmarks ${num_landmarks} --seed ${seed} --use_ReLU --gain 0.01 \
--n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 25 \
--model_dir "results/MPE/simple_spread_fix/rmappo/0418_check/wandb/run-20220419_225218-22x4xlyz/files" \
--use_wandb --max_z 4