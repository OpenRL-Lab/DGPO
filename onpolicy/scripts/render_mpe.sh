#!/bin/sh
env="MPE"
scenario="spread(hard)"
num_landmarks=3
num_agents=3
algo="MAPPO"
exp="check"
seed=1


CUDA_VISIBLE_DEVICES=0 python3 render/render_mpe.py --save_gifs --env_name ${env} --algorithm_name ${algo} \
--experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} \
--num_landmarks ${num_landmarks} --seed ${seed} --use_ReLU --gain 0.01 \
--n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 15 \
--model_dir "results/MPE/spread(hard)/MAPPO/exp/run12/models" \
--use_wandb --max_z 2

# "/mfs/chenwenze19/on-policy/onpolicy/scripts/results/MPE/simple_spread_fix/ours/exp/wandb/latest-run/files"