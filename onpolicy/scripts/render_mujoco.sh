#!/bin/sh
env="mujoco"
scenario="Hopper-v2"
num_agents=1
algo="MAPPO"
exp="check"
seed=1


CUDA_VISIBLE_DEVICES=0 python3 render/render_mujoco.py --save_gifs --env_name ${env} \
--algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
--seed ${seed} --num_agents ${num_agents} --use_ReLU --gain 0.01 --use_wandb \
--n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 10000 \
--model_dir "results/${env}/${scenario}/${algo}/exp/wandb/latest-run/files" \
--max_z 2