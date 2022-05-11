env="StarCraft2"
map="3m"
algo="ours"
exp="exp"
seed=1

CUDA_VISIBLE_DEVICES=0 python3 render/render_smac.py --env_name ${env} \
--algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} \
--n_training_threads 1 --num_mini_batch 1 --episode_length 2000 \
--use_ReLU --use_render --n_rollout_threads 1 --max_z 3 \
--model_dir "results/${env}/${map}/${algo}/${exp}/wandb/latest-run/files"