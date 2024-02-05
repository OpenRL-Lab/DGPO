#!/bin/sh
env="MPE"
algo="MAPPO"
exp="exp"
seed_max=1

scenario="spread(hard)" 
num_landmarks=3
num_agents=3
rex_thresh=-37
div_thresh=1.1
max_z=2

# scenario="spread(easy)" 
# num_landmarks=4
# num_agents=1
# rex_thresh=-2.5
# div_thresh=1.1
# max_z=4


echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python3 train/train_mpe.py \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks}  \
    --n_rollout_threads 128 --n_eval_rollout_threads 128 --episode_length 15 \
    --num_env_steps 1000000 --ppo_epoch 10 --use_ReLU --gamma 0.99 0.99 --discri_lr 1e-4 \
    --max_z ${max_z}  --alpha 5. --rex_thresh ${rex_thresh} --div_thresh ${div_thresh} \
    --wandb_name "cwz19" --user_name "cwz19" --eval_interval 25 
done
