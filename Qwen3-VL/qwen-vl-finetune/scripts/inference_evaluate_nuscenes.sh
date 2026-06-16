#!/usr/bin/env bash
#SBATCH --job-name=nuscenes-inference
#SBATCH --output=logs/nuscenesinference-%j.out
#SBATCH --error=logs/nuscenesinference-%j.err
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=400GB
#SBATCH --time=1:00:00
#SBATCH --exclude=h012

module load conda
module load modtree/gpu
# module load cuda/12.0.1
module load cuda/12.8.0


echo $PWD

# Set CHECKPOINT_PATH to a local fine-tuned checkpoint or a HuggingFace model ID,
# and EVAL_TAG to a short name used for the output/log filenames.
export CHECKPOINT_PATH="Qwen/Qwen3-VL-4B-Instruct"
export EVAL_TAG="qwen3vl4b_zeroshot"


echo "Evaluating model at ${CHECKPOINT_PATH}"

######## INFERENCE STEP ###########
echo "************ Running Inference ************"
# conda activate /path/to/conda/envs/qwen3_serve
conda activate /path/to/conda/envs/qwen3_serve_new

# Inference parameters
export TEMPERATURE=0.1
export MAX_TOKENS=6144  #4096
export INFERENCE_OUTPUT_PATH="results/${EVAL_TAG}_toks${MAX_TOKENS}.jsonl"

python qwenvl/inference/inference_qwen.py \
    --model_name_or_path $CHECKPOINT_PATH \
    --annotation_path /path/to/Agent-Driver/data/finetune/nuscenes_trajonly_vqadriver_val.jsonl \
    --data_path /path/to/nuscenes \
    --output_path $INFERENCE_OUTPUT_PATH \
    --temperature $TEMPERATURE \
    --max_tokens $MAX_TOKENS

# get absolute path of output file
INFERENCE_OUTPUT_PATH=$(realpath $INFERENCE_OUTPUT_PATH)
echo "Inference completed. Results saved to $INFERENCE_OUTPUT_PATH"

    # --annotation_path /path/to/Agent-Driver/data/finetune/nuscenes_trajonly_val.json \

######## EVALUATION STEP ###########
# Unload specific modules when you no longer need them
module unload cuda/12.0.1
module unload modtree/gpu

echo "************ Running Evaluation ************"
conda deactivate
# Evaluation scripts are at /path/to/Agent-Driver
cd ../../Agent-Driver

echo $PWD
conda activate /path/to/conda/envs/driveagent_new

export EVALUATION_LOG_FILE="evaluation_logs/${EVAL_TAG}_toks${MAX_TOKENS}.txt"
export PARSED_OUTPUT_PATH="results/${EVAL_TAG}.pkl"

python agentdriver/execution/parse_generations_to_pred.py \
    --input_file $INFERENCE_OUTPUT_PATH \
    --output_file $PARSED_OUTPUT_PATH

echo "Finished Parsing Generations. Beginning Evaluation..."

# Pipe evaluation output to a log file
export PYTHONPATH=$PWD
python agentdriver/evaluation/evaluation.py --metric uniad --result_file ${PARSED_OUTPUT_PATH} | tee $EVALUATION_LOG_FILE

echo "Evaluation completed. Results saved to ${EVALUATION_LOG_FILE}"
