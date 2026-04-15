#!/bin/bash
set -x


VERL_PATH=""

export PYTHONPATH=${VERL_PATH}:$PYTHONPATH


wandb offline
export WANDB_MODE=offline 

export MODEL_PATH=.../Qwen3-8B
export MODEL_NAME="unk/qwen3-8b/unk3-3"
export OUTPUT_DIR=./reasoning-unk/checkpoints/${MODEL_NAME}

use_dynamic_bsz=True

mkdir -p ${OUTPUT_DIR}

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files=['./data/unk_train_4number.parquet','./data/unk_train_6number.parquet','./data/unk_train_8number.parquet'] \
    data.val_files=['./data/unk_eval_4number.parquet','./data/unk_eval_6number.parquet','./data/unk_eval_8number.parquet'] \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.train_batch_size=32 \
    data.val_batch_size=512 \
    data.max_prompt_length=8192 \
    data.max_response_length=10240 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.model.use_remove_padding=True \
    algorithm.kl_ctrl.kl_coef=0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='deepscaler' \
    trainer.experiment_name=${MODEL_NAME} \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=5 \
    trainer.test_freq=-1 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=20 "${@:1}" \
    trainer.default_local_dir=${OUTPUT_DIR} \
    +trainer.wandb_save_dir=${OUTPUT_DIR} \
    +trainer.rollout_data_dir=${OUTPUT_DIR}/rollout_data \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    +reward_model.unk_reward_method=unk3 \
    2>&1 | tee ${OUTPUT_DIR}/log-$(date +%Y%m%d%H%M%S).log

