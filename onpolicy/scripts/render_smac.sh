env="StarCraft2"
map="3m"
algo="rmappo"
exp="render"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python3 render/render_smac.py --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} \
    --n_training_threads 1 --render_episodes 2 --n_rollout_threads 1 --num_mini_batch 1 --episode_length 400 \
    --use_ReLU --use_render \
    --model_dir "results/StarCraft2/3m/rmappo/vmapd_share/wandb/run-20220312_001409-l8b45l89/files"
done

#--use_value_active_masks 