# export NCCL_SHM_DISABLE=1
# export NCCL_DEBUG=INFO   # 顺便把 NCCL log 打开，调试好用

# export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export WANDB_API_KEY="f4964340b710e6450355ca2bd2b2f29de3d86312"
export CUDA_VISIBLE_DEVICES=4,5,6,7


# Inside your container / shell
export RAY_DISABLE_DASHBOARD=1      # don't start the dashboard
export RAY_USAGE_STATS_ENABLED=0    # optional: also disable telemetry

# ray stop
# pkill -9 ray || true

python -m verl.trainer.main_ppo  # plus your usual Hydra overrides