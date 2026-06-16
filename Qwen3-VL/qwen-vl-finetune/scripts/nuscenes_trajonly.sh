#!/usr/bin/env bash
#SBATCH --job-name=nuscenes-sft
#SBATCH --output=logs/nuscenessft-%j.out
#SBATCH --error=logs/nuscenessft-%j.err
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=400GB
#SBATCH --time=16:00:00
#SBATCH --exclude=h012

module load conda
module load modtree/gpu
module load cuda/12.0.1


echo $pwd

# Set CONDA_ENV_NAME to default if not set
if [ -z "$CONDA_ENV_NAME" ]; then
    CONDA_ENV_NAME="/path/to/conda/envs/qwen3-vl-new"
fi

conda activate $CONDA_ENV_NAME
echo "********************** Running with ${CONDA_ENV_NAME} ***********************"

export NUM_GPUS=4
########################################################

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs

export WANDB_PROJECT="nuscenes_qwen3_4b_experiments" #quadruped_qwen3_4b

# # DeepSpeed configuration
# deepspeed=./scripts/zero3.json

# Model configuration
llm=Qwen/Qwen3-VL-4B-Instruct  # Using HuggingFace model ID

# Training hyperparameters
lr=1e-5
batch_size=4
grad_accum_steps=4
num_train_epochs=30
max_model_length=2048

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
run_name=nuscenes_vqa_trajonly
datasets="${run_name}%100"

# Output configuration
# output_dir=./checkpoints/${run_name}_maxlen8192
# output_dir=./checkpoints/${run_name}_maxlen6144
output_dir=./checkpoints/${run_name}_maxlen${max_model_length}_${num_train_epochs}ep
# output_dir=./checkpoints/${run_name}_maxlen2048
# output_dir=./checkpoints/${run_name}_maxlen4096_bs32

# Training arguments #changed 2 to 0.25 epochs; 
args="
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 6 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length ${max_model_length} \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name}_maxlen${max_model_length}_${num_train_epochs}ep \
    --report_to wandb"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
