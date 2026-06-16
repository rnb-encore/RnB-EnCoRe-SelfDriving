#!/usr/bin/env bash
#SBATCH --job-name=nuscenes-rnbencore-inf
#SBATCH --output=logs/nuscrnbencore-inf-%j.out
#SBATCH --error=logs/nuscrnbencore-inf-%j.err
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=400GB
#SBATCH --time=8:00:00
#SBATCH --exclude=h012

module load conda
module load modtree/gpu
# module load cuda/12.0.1
module load cuda/12.8.0

echo $pwd

# Set CONDA_ENV_NAME to default if not set
if [ -z "$CONDA_ENV_NAME" ]; then
    CONDA_ENV_NAME="/path/to/conda/envs/qwen3_serve_new"
fi

conda activate $CONDA_ENV_NAME
echo "********************** Running with ${CONDA_ENV_NAME} ***********************"

export NUM_GPUS=1
########################################################

### some parameters
KGEN=8
if [[ "$1" == "--kgen" ]] && [[ -n "$2" ]]; then
    KGEN="$2"
fi
export KGEN

echo "Using KGEN=${KGEN} for inference generation"

# Path to the iteration-0 (Rnbencore it0) fine-tuned checkpoint used to generate samples.
export CHECKPOINT_PATH="/path/to/Qwen3-VL/qwen-vl-finetune/checkpoints/nuscenes_vqadriver_rnbencoreit0_dropout05_maxlen6144_2ep_lr5e-5/checkpoint-6000"
export OUTPUT_PATH="./results/nuscenes_vqadriver_rnbencore_do05_kgen${KGEN}/rnbencoreit0_generation.jsonl"
export MAX_MODEL_LEN=6144
export MAX_TOKENS=4096

python qwenvl/inference/rnbencore_generation_qwen.py \
    --model_name_or_path $CHECKPOINT_PATH \
    --annotation_path /path/to/Agent-Driver/data/finetune/nuscenes_trajonly_vqadriver_train.jsonl \
    --data_path /path/to/nuscenes \
    --output_path $OUTPUT_PATH \
    --max_tokens $MAX_TOKENS \
    --max_model_len $MAX_MODEL_LEN \
    --k_gen ${KGEN} \
    --temperature 1.0 \
    --k_sample 1 \
    --batch_size 1024

echo "Inference completed. Results saved to $OUTPUT_PATH"