#!/usr/bin/env python3
"""
Validation loss computation script for Qwen-VL models using vLLM.
Computes the loss on ground truth assistant responses instead of generating.
"""

import os
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

# Set environment variable for vLLM
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.inference.inference_utils import InferenceArguments
from qwenvl.inference.inference_utils import (
    load_dataset_from_config,
    build_vllm_messages,
    prepare_inputs_for_vllm
)


def extract_assistant_response(item: Dict) -> Optional[str]:
    """
    Extract the ground truth assistant response from the data item.
    
    Args:
        item: Data item containing conversations
        
    Returns:
        The assistant's response text, or None if not found
    """
    conversations = item.get("conversations", [])
    
    # Find the last assistant response (or the one we want to evaluate)
    for conv in reversed(conversations):
        if conv.get("from") == "assistant" or conv.get("role") == "assistant":
            return conv.get("value") or conv.get("content")
    
    return None


def prepare_prompt_and_response(
    item: Dict,
    processor: AutoProcessor,
    base_path: Path
) -> tuple:
    """
    Prepare prompt (user messages) and ground truth response separately.
    
    Args:
        item: Data item
        processor: AutoProcessor
        base_path: Base path for data files
        
    Returns:
        Tuple of (prompt_messages, ground_truth_response)
    """
    # Extract ground truth response
    ground_truth = extract_assistant_response(item)
    if ground_truth is None:
        return None, None
    
    # Build messages up to (but not including) the assistant response
    conversations = item.get("conversations", [])
    prompt_conversations = []
    
    for conv in conversations:
        role = conv.get("from") or conv.get("role")
        if role == "assistant":
            # Stop before the assistant response we want to evaluate
            break
        prompt_conversations.append(conv)
    
    # Create a temporary item with only prompt conversations
    prompt_item = item.copy()
    prompt_item["conversations"] = prompt_conversations
    
    # Build messages for the prompt
    messages = build_vllm_messages(prompt_item, base_path)
    
    return messages, ground_truth


def compute_validation_loss(args: InferenceArguments):
    """
    Compute validation loss on ground truth responses.
    
    Args:
        args: Inference arguments
    """
    print("=" * 80)
    print(f"Computing validation loss with model: {args.model_name_or_path}")
    print("=" * 80)

    # Load dataset
    print("\n[1/5] Loading dataset...")
    data_items = load_dataset_from_config(args)

    if len(data_items) == 0:
        print("No data loaded!")
        return

    if args.max_samples is not None:
        data_items = data_items[:args.max_samples]

    # Load processor and tokenizer
    print("\n[2/5] Loading processor and tokenizer...")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    print("Processor and tokenizer loaded.")

    # Initialize vLLM
    print("\n[3/5] Initializing vLLM model...")
    print(f"  - Model: {args.model_name_or_path}")
    print(f"  - Tensor parallel size: {args.tensor_parallel_size}")
    print(f"  - GPU memory utilization: {args.gpu_memory_utilization}")

    llm = LLM(
        model=args.model_name_or_path,
        max_model_len=args.max_model_len,
        mm_encoder_tp_mode="data",
        enable_expert_parallel=args.enable_expert_parallel,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    print("vLLM model initialized.")

    # Prepare data for loss computation
    print("\n[4/5] Preparing inputs for validation loss computation...")
    
    prompts_for_scoring = []
    prompt_lengths = []  # Store prompt-only token lengths
    
    for idx, item in enumerate(tqdm(data_items, desc="Processing samples")):
        base_path = Path(item.get("data_path", ""))
        
        # Get prompt and ground truth response
        messages, ground_truth = prepare_prompt_and_response(item, processor, base_path)
        
        if messages is None or ground_truth is None:
            print(f"Skipping sample {idx}: no valid assistant response")
            continue
        
        # Get prompt-only length by tokenizing the prompt part
        prompt_input = prepare_inputs_for_vllm(messages, processor)
        # Tokenize to get the actual token count
        prompt_tokens = tokenizer.encode(prompt_input['prompt'], add_special_tokens=False)
        prompt_lengths.append(len(prompt_tokens))
        
        # For vLLM scoring, we need the full prompt + completion
        # Add the assistant response to messages
        full_messages = messages + [{
            "role": "assistant",
            "content": ground_truth
        }]
        
        # Prepare the full input for scoring
        full_input = prepare_inputs_for_vllm(full_messages, processor)
        prompts_for_scoring.append(full_input)
    
    if len(prompts_for_scoring) == 0:
        print("No valid samples to process!")
        return

    print(f"Prepared {len(prompts_for_scoring)} samples for validation")

    # Compute log probabilities
    print(f"\n[5/5] Computing log probabilities on {len(prompts_for_scoring)} samples...")
    
    # Set sampling params to get logprobs
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding for evaluation
        max_tokens=1,  # We only need logprobs, not generation
        prompt_logprobs=1,  # Get logprobs for prompt tokens
    )
    
    try:
        # Generate with logprobs
        # Note: vLLM's generate will compute logprobs for the entire sequence
        outputs = llm.generate(prompts_for_scoring, sampling_params=sampling_params)
        print("Log probability computation completed!")
    except Exception as e:
        print(f"Error during log probability computation: {e}")
        print("\nNote: Computing validation loss with vLLM requires using the prompt logprobs.")
        print("Alternative approach: Use the model's native forward pass for exact loss computation.")
        return

    # Process results and compute losses
    print("\nProcessing results and computing losses...")
    
    total_loss = 0.0
    total_tokens = 0
    
    for prompt_len, output in zip(prompt_lengths, outputs):
        # Extract prompt logprobs
        prompt_token_ids = output.prompt_token_ids
        prompt_logprobs = output.prompt_logprobs
        
        if prompt_logprobs is None:
            raise ValueError("Prompt logprobs are None. Ensure that prompt_logprobs=1 is set in SamplingParams.")
        
        # The assistant response starts after the prompt tokens
        # Note: prompt_logprobs[i] contains the logprob for token_ids[i]
        # We want to compute loss only on assistant tokens
        assistant_start_idx = prompt_len
        
        # Compute negative log-likelihood (loss) for assistant tokens only
        token_losses = []
        for i in range(assistant_start_idx, len(prompt_token_ids)):
            token_id = prompt_token_ids[i]
            token_logprob_dict = prompt_logprobs[i]
            
            if token_logprob_dict is not None and token_id in token_logprob_dict:
                logprob = token_logprob_dict[token_id].logprob
                token_losses.append(-logprob)
        
        # Accumulate loss
        if token_losses:
            sample_loss = sum(token_losses)
            sample_tokens = len(token_losses)
            total_loss += sample_loss
            total_tokens += sample_tokens
    
    # Compute overall metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = np.exp(avg_loss)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"Validation Loss Computation Completed!")
    print(f"{'='*80}")
    print(f"Total samples: {len(outputs)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"{'='*80}")
    

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compute validation loss for Qwen-VL models using vLLM"
    )

    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to model checkpoint or HuggingFace model name"
    )

    # Data arguments
    parser.add_argument(
        "--dataset_use",
        type=str,
        default="",
        help="Comma-separated list of datasets (e.g., 'nuscenes_trajonly')"
    )
    parser.add_argument(
        "--annotation_path",
        type=str,
        default=None,
        help="Direct path to annotation file (overrides dataset_use)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Base path for data files"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="validation_loss_results.jsonl",
        help="Path to save results"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process"
    )

    # vLLM arguments
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=2048,
        help="Max model length"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization"
    )
    parser.add_argument(
        "--enable_expert_parallel",
        action="store_true",
        help="Enable expert parallelism for MoE models"
    )

    args = parser.parse_args()

    # Convert to InferenceArguments dataclass
    inference_args = InferenceArguments(
        model_name_or_path=args.model_name_or_path,
        dataset_use=args.dataset_use,
        annotation_path=args.annotation_path,
        data_path=args.data_path,
        output_path=args.output_path,
        max_samples=args.max_samples,
        max_model_len=args.max_model_len,
        temperature=0.7,  # Not used for loss computation
        top_p=0.9,  # Not used for loss computation
        max_tokens=512,  # Not used for loss computation
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_expert_parallel=args.enable_expert_parallel,
    )

    compute_validation_loss(inference_args)


if __name__ == "__main__":
    main()
