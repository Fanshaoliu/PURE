#!/bin/bash

# ================= 配置区域 (修改这里即可) =================
# 实验基础设置
MODEL_PATH="Qwen/Qwen2.5-Math-1.5B"
EXP_NAME_PREFIX="RLPgR-random_progress"
GPUS="4,5,6,7"

# WandB
WANDB_KEY="f4964340b710e6450355ca2bd2b2f29de3d86312"

# 关键超参数
MAX_PROMPT_LEN=1024
MAX_RESPONSE_LEN=3072
ROLLOUT_MAX_MODEL_LEN=8192
BATCH_SIZE=8192 # max_num_batched_tokens
GPU_MEM_UTIL=0.4

# ================= 环境与系统设置 =================
export CUDA_VISIBLE_DEVICES=$GPUS
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_API_KEY=$WANDB_KEY

# 计算 GPU 数量
GPU_COUNT=$(echo $GPUS | tr ',' '\n' | wc -l)

# vLLM & NCCL 优化配置
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export RAY_DISABLE_DASHBOARD=1
export RAY_USAGE_STATS_ENABLED=0
export CUDA_LAUNCH_BLOCKING=1

# 动态生成日志文件名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="training_log"
mkdir -p $LOG_DIR
LOG_FILE="${LOG_DIR}/${EXP_NAME_PREFIX}_qwen2p5math1p5b.log"

echo "[INFO] Starting PPO training on ${GPU_COUNT} GPUs..."
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] Log file: $LOG_FILE"

# ================= 自动重试逻辑 =================

MAX_RETRIES=20
COUNT=1

# 开启 pipefail，确保 python 挂了之后，$? 能捕获到错误码，而不是被 tee 的成功状态掩盖
# set -o pipefail

while [ $COUNT -le $MAX_RETRIES ]; do
    echo "=========================================================="
    echo "[INFO] Starting Attempt $COUNT / $MAX_RETRIES"
    echo "=========================================================="

    # 运行训练命令
    # 注意：这里去掉了 set -e 的影响，因为我们要自己处理错误
    python -m verl.trainer.main_ppo \
        actor_rollout_ref.model.path=$MODEL_PATH \
        data.max_prompt_length=$MAX_PROMPT_LEN \
        data.max_response_length=$MAX_RESPONSE_LEN \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32000 \
        critic.ppo_max_token_len_per_gpu=32000 \
        reward_model.forward_max_token_len_per_gpu=32000 \
        actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_MODEL_LEN \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32000 \
        actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
        actor_rollout_ref.rollout.max_num_batched_tokens=$BATCH_SIZE \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=True \
        reward_model.type=prm \
        reward_model.credit_assignment=0.1 \
        trainer.n_gpus_per_node=$GPU_COUNT \
        trainer.save_freq=10 \
        +reward_model.use_progress_rubric=False \
        +reward_model.use_progress_aggregation=True \
        +reward_model.progress_agg_method='min' \
        +reward_model.min_step_size=2 \
        +reward_model.max_progress=5 \
        +reward_model.debug_print=False \
        trainer.experiment_name="${EXP_NAME_PREFIX}-\${reward_model.type}_\${actor_rollout_ref.model.path}_prompts-\${data.train_batch_size}_n-\${actor_rollout_ref.rollout.n}" \
        2>&1 | tee -a $LOG_FILE  # 使用 -a 追加日志，保留历史报错信息

    # 捕获 Python 的退出码
    EXIT_CODE=$?

    # if [ $EXIT_CODE -eq 0 ]; then
    #     echo "[INFO] Training finished successfully!"
    #     break
    # else
    echo "[WARN] Training failed with exit code $EXIT_CODE."
    
    if [ $COUNT -eq $MAX_RETRIES ]; then
        echo "[ERROR] Maximum retries reached. Exiting."
        exit 1
    fi

    echo "[INFO] Cleaning up Ray processes and sleeping for 10s before retry..."
    
    # # 核心：必须清理 Ray，否则显存可能没释放，下次启动必挂
    # pkill -9 ray || true
    # # 也可以考虑清理临时文件，但要小心不要误删
    # # rm -rf /tmp/ray/* || true 

    sleep 10
    COUNT=$((COUNT+1))
    # fi
done


# python launch_shell.py --gpus '4,5,6,7' --command 'sh scripts/ppo_run_script/run_qwen2p5math1p5b_rlpgr.sh' --interval 1 --util-threshold 80 --mem-threshold 80