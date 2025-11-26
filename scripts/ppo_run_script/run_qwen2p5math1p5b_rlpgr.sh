#!/bin/bash
# export NCCL_SHM_DISABLE=1
# export NCCL_DEBUG=INFO   # 顺便把 NCCL log 打开，调试好用

# export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export WANDB_API_KEY="f4964340b710e6450355ca2bd2b2f29de3d86312"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Inside your container / shell
export RAY_DISABLE_DASHBOARD=1      # don't start the dashboard
export RAY_USAGE_STATS_ENABLED=0    # optional: also disable telemetry

# ray stop
# pkill -9 ray || true

python -m verl.trainer.main_ppo \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-Math-1.5B \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4500 \
    critic.ppo_max_token_len_per_gpu=4500 \
    reward_model.forward_max_token_len_per_gpu=4500 \
    actor_rollout_ref.actor.use_progress_aggregation=True \
    actor_rollout_ref.actor.progress_agg_method='min' \
    actor_rollout_ref.actor.min_step_size=2 \
    actor_rollout_ref.actor.max_progress=5 \
    actor_rollout_ref.actor.debug_print=False \
    trainer.experiment_name='RLPgR_${actor_rollout_ref.model.path}_prompts-${data.train_batch_size}_n-${actor_rollout_ref.rollout.n}' \
    2>&1 | tee training_log/rlpgr_qwen2p5math1p5b.log


# python launch_shell.py --gpus '0,1,2,3' --command 'sh scripts/ppo_run_script/run_qwen2p5math1p5b_rlpgr.sh' --interval 1 --util-threshold 90 --mem-threshold 60