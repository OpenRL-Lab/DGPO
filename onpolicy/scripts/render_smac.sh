env="StarCraft2"
map="3m"
algo="rmappo"
exp="render"
seed=2

CUDA_VISIBLE_DEVICES=1 python3 render/render_smac.py --env_name ${env} \
--algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} \
--n_training_threads 1 --num_mini_batch 1 --episode_length 2000 \
--use_ReLU --use_render --n_rollout_threads 1 --max_z 3 \
--model_dir "results/StarCraft2/3m/rmappo/vmapd_local/wandb/run-20220507_010226-go8izlso/files"