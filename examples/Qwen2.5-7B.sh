set -x
export PYTHONHASHSEED=0
export RAY_memory_monitor_refresh_ms=0
export WANDB_API_KEY=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

gsm8k_test_path=data/test_data/gsm8k_test.parquet
math_test_path=data/test_data/math_500.parquet
amc23_test_path=data/test_data/amc23_repeated_8_times.parquet
aime24_test_path=data/test_data/aime24_repeated_8_times.parquet
minerva_test_path=data/test_data/minerva_math.parquet
olympiadbench_test_path=data/test_data/olympiadbench.parquet

deepscaler_uniform_train_path=data/deepscaler_uniform_train.parquet

train_files="['$deepscaler_uniform_train_path']"
test_files="['$aime24_test_path']"

EXP_NAME='Qwen2.5-7B_GRPO'
OUTPUT_DIR="checkpoints/MATH/${EXP_NAME}"
mkdir -p ${OUTPUT_DIR}

VAL_OUTPUT_DIR="${OUTPUT_DIR}/validation"
mkdir -p ${VAL_OUTPUT_DIR}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=3000 \
    data.filter_overlong_prompts=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    data.truncation='error' \
    data.shuffle=False \
    data.save_dir="training_logs_valid/${EXP_NAME}" \
    actor_rollout_ref.model.path=/ceph_home/mazhengzhao2024/models/Qwen2.5-7B \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_dcpo' \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 $@ 2>&1 | tee ${OUTPUT_DIR}/training_process.log

    # actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    # data.train_batch_size=1024 \
    # trainer.n_gpus_per_node=8 \
    # actor_rollout_ref.model.use_shm=True \