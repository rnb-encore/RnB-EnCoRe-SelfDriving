#!/usr/bin/env python3
"""
Script for generating Rnbencore data using Qwen-VL model inference. The process is as such:
1. Load the Qwen-VL model and processor.
2. For each datapoint in train dataset:
    a. Prepare prior and posterior prompt inputs.
    b. Run K_gen samples of generation for the posterior prompt (reasoning given task and action).
    c. For each K_gen, evaluate the likelihood of prior prompt + posterior generation + action. 
    d. Resample K generations from the posterior generations based on ELBO importance weights.
3. Save the generated data to output file.
"""
import os
import json
import argparse
import sys
import pickle
from pathlib import Path
from typing import Dict, List

from math import ceil
import numpy as np
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from tqdm import tqdm
import torch
import gc

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


class RnbencoreConfig:
    """Configuration constants for Rnbencore inference."""
    PLANNED_TRAJECTORY_PREFIX = "Planned Trajectory:"
    CONVERSATION_TYPE_PRIOR = "prior"
    CONVERSATION_TYPE_POSTERIOR = "posterior"
    POSTERIOR_PROB = 0.5  # Probability of selecting posterior conversation
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_K_GEN = 8
    DEFAULT_K_SAMPLE = 8


def extract_action_from_conversations(conversations: List[Dict]) -> str:
    """
    Extract action text from conversations.

    Args:
        conversations: List of conversation turns

    Returns:
        Action text string
    """
    assert len(conversations) == 2, "Expected exactly 2 conversation turns"
    # Extract ground truth action: predicted trajectory
    action_text = conversations[1]["value"]  # Assistant's trajectory response
    # assert that the action_text starts with "Planned Trajectory:"
    if not action_text.startswith(RnbencoreConfig.PLANNED_TRAJECTORY_PREFIX):
        action_text = action_text[action_text.find(RnbencoreConfig.PLANNED_TRAJECTORY_PREFIX):]
    return action_text


def build_posterior_messages(
    item: Dict,
    base_path: str,
) -> List[Dict]:
    """
    Build posterior prompt: prior (context + image) + action.

    Args:
        item: Dict with 'conversations', 'image', fields
        base_path: Base path for data

    Returns:
        Messages for posterior generation
    """
    # Extract ground truth action: predicted trajectory
    action_text = extract_action_from_conversations(item.get("conversations", []))
    
    # Copy a modified version of the item conversations to contain only user input, with the action appended
    posterior_item = item.copy()
    modified_conversations = [
        {
            "from": item["conversations"][0]["from"],
            "value": item["conversations"][0]["value"] + action_text
        }
    ]
    posterior_item["conversations"] = modified_conversations

    # Build messages
    messages = build_vllm_messages(posterior_item, base_path)

    return messages


def build_prior_messages(
    item: Dict,
    latent_texts: List[str],
    base_path: str,
) -> List[List[Dict]]:
    """
    Build prior messages: context + image + latent reasoning + action.

    Args:
        item: Dict with 'conversations', 'image', fields
        latent_texts: List of latent reasoning texts
        base_path: Base path for data

    Returns:
        List of num_latent messages for prior likelihood evaluation
    """
    action_text = extract_action_from_conversations(item.get("conversations", []))

    # Build messages for prior
    prior_messages = []
    for latent_text in latent_texts:
        prior_item = item.copy()
        modified_conversations = [
            {
                "from": prior_item["conversations"][0]["from"],
                "value": prior_item["conversations"][0]["value"]  
            },
            {
                "from": prior_item["conversations"][1]["from"],
                "value": latent_text + "\n\n" + action_text
            }
        ]
        prior_item["conversations"] = modified_conversations

        # Build vllm message with assistant message included
        messages = build_vllm_messages(prior_item, base_path, include_assistant=True)
        prior_messages.append(messages)

    return prior_messages


def generate_posterior_samples(
    llm: LLM,
    data_items: List[Dict],
    processor: AutoProcessor,
    args: InferenceArguments
) -> List[Dict]:
    """
    Generate K_gen posterior samples for each datapoint.
    
    Returns:
        List of dicts containing item, latent texts and posterior logprobs
    """
    # Set up sampling parameters for posterior generation
    posterior_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.k_gen,  # Generate K_gen samples per prompt
        logprobs=1,  # Get log probabilities for ELBO calculation
    )

    print(f"\nGenerating {args.k_gen} posterior samples per datapoint...")
    print(f"Processing in batches of {args.batch_size}...")
    num_batches = ceil(len(data_items) / args.batch_size)

    posterior_results = []
    
    for batch_idx in tqdm(range(num_batches), desc="Generating posterior samples"):
        # Get batch of items
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(data_items))
        batch_items = data_items[start_idx:end_idx]
        
        # Prepare all posterior inputs for this batch
        batch_posterior_inputs = []
        batch_base_paths = []
        
        for item in batch_items:
            base_path = Path(item.get("data_path", ""))
            batch_base_paths.append(base_path)
            
            # Build posterior prompt for this item
            posterior_messages = build_posterior_messages(item, base_path)
            posterior_input = prepare_inputs_for_vllm(posterior_messages, processor)
            batch_posterior_inputs.append(posterior_input)
        
        # Generate K_gen posterior samples for ALL items in batch
        batch_posterior_outputs = llm.generate(
            batch_posterior_inputs, 
            sampling_params=posterior_sampling_params
        )
        
        # Process outputs
        for item, posterior_output, base_path in zip(
            batch_items, batch_posterior_outputs, batch_base_paths
        ):
            # Extract latent texts and logprobs for this item
            latent_texts = [sample.text for sample in posterior_output.outputs]
            posterior_logprobs = [sample.cumulative_logprob for sample in posterior_output.outputs]
            
            posterior_results.append({
                'item': item,
                'base_path': base_path,
                'latent_texts': latent_texts,
                'posterior_logprobs': posterior_logprobs
            })
    
    print(f"Generated posterior samples for {len(posterior_results)} datapoints")
    return posterior_results


def evaluate_prior_likelihoods(
    llm: LLM,
    posterior_results: List[Dict],
    processor: AutoProcessor,
    args: InferenceArguments
) -> List[Dict]:
    """
    Evaluate prior likelihoods and compute ELBO weights for all posterior samples.
    
    Returns:
        List of dicts with all data needed for resampling
    """
    prior_sampling_params = SamplingParams(
        prompt_logprobs=1,  # Get log probs for the prompt tokens
        max_tokens=1,  # Generate just 1 token (we don't care about it)
        temperature=0,
    )
    
    print(f"\nEvaluating prior likelihoods...")
    print(f"Processing in batches of {args.batch_size}...")
    num_batches = ceil(len(posterior_results) / args.batch_size)
    
    latent_reasoning_data = []
    
    for batch_idx in tqdm(range(num_batches), desc="Evaluating prior likelihoods"):
        # Get batch of posterior results
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(posterior_results))
        batch_posterior_results = posterior_results[start_idx:end_idx]
        
        # Prepare all prior inputs for this batch
        all_prior_inputs = []  # Will hold batch_size * k_gen prior inputs
        batch_metadata = []  # Track which prior input belongs to which item/generation
        
        for post_result in batch_posterior_results:
            item = post_result['item']
            base_path = post_result['base_path']
            latent_texts = post_result['latent_texts']
            posterior_logprobs = post_result['posterior_logprobs']
            
            # Build prior messages for all k_gen latents of this item
            prior_messages = build_prior_messages(item, latent_texts, base_path)
            prior_inputs = [
                prepare_inputs_for_vllm(messages, processor, add_generation_prompt=False, truncate_prompt_tokens=args.max_model_len - 1) 
                for messages in prior_messages
            ]
            
            # Add to batch and track metadata
            all_prior_inputs.extend(prior_inputs)
            batch_metadata.append({
                'item': item,
                'latent_texts': latent_texts,
                'posterior_logprobs': posterior_logprobs,
                'num_generations': len(latent_texts),
                'prior_input_start_idx': len(all_prior_inputs) - len(prior_inputs)
            })
        
        # Evaluate prior likelihoods for ALL prior inputs in batch
        all_prior_outputs = llm.generate(all_prior_inputs, sampling_params=prior_sampling_params)
        
        # Process prior outputs and compute ELBO weights
        for meta in batch_metadata:
            item = meta['item']
            latent_texts = meta['latent_texts']
            posterior_logprobs = meta['posterior_logprobs']
            num_gens = meta['num_generations']
            prior_input_start_idx = meta['prior_input_start_idx']
            
            # Extract prior outputs for this item
            item_prior_outputs = all_prior_outputs[prior_input_start_idx:prior_input_start_idx + num_gens]
            
            # Compute offsets (using first sample)
            first_prior_input = all_prior_inputs[prior_input_start_idx]
            image_token_offset = len(item_prior_outputs[0].prompt_token_ids) - \
                len(processor.tokenizer.encode(first_prior_input['prompt'], add_special_tokens=False))
            prior_prompt_token_len = len(processor.tokenizer.encode(
                first_prior_input['prompt'][:first_prior_input['prompt'].find(latent_texts[0])], 
                add_special_tokens=False
            ))
            sample_prefix_len = image_token_offset + prior_prompt_token_len
            
            # Compute prior logprobs for each generation
            prior_logprobs = []
            for prior_output in item_prior_outputs:
                prior_token_ids = prior_output.prompt_token_ids
                prior_token_logprobs = prior_output.prompt_logprobs
                
                prior_latent_logprob = 0.0
                for token_id, token_logprob in zip(
                    prior_token_ids[sample_prefix_len:], 
                    prior_token_logprobs[sample_prefix_len:]
                ):
                    if token_id is not None and token_logprob is not None:
                        prior_latent_logprob += token_logprob.get(token_id).logprob
                
                prior_logprobs.append(prior_latent_logprob)
            
            # Compute importance weights
            importance_weights = []
            for k in range(args.k_gen):
                elbo_weight = prior_logprobs[k] - posterior_logprobs[k]
                importance_weights.append(elbo_weight)
            
            # Store results
            sample_data = {
                "original_item": item,
                "latent_texts": latent_texts,
                "posterior_logprobs": posterior_logprobs,
                "prior_logprobs": prior_logprobs,
                "importance_weights": importance_weights,
            }
            latent_reasoning_data.append(sample_data)
    
    print(f"Evaluated prior likelihoods for {len(latent_reasoning_data)} datapoints")
    return latent_reasoning_data


def resample_and_save(
    latent_reasoning_data: List[Dict],
    output_path: Path,
    k_sample: int
) -> Dict[str, int]:
    """
    Resample based on ELBO weights and save results.
    
    Returns:
        Statistics about saved conversations
    """
    print(f"\nResampling and saving results...")
    print(f"Saving to: {output_path}")
    
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Save statistics
    total_prior_conversations = 0
    total_posterior_conversations = 0
    
    with open(output_path, 'w') as f:
        for sample_data in tqdm(latent_reasoning_data, desc="Resampling and saving"):
            # Extract necessary info
            weights = sample_data["importance_weights"]
            posterior_generations = sample_data["latent_texts"]
            sample_item = sample_data["original_item"]
            previous_conversations = sample_item["conversations"]

            # Get likelihoods for saving
            posterior_logprobs = sample_data["posterior_logprobs"]
            prior_logprobs = sample_data["prior_logprobs"]

            # Convert logprobs to probabilities and normalize (importance weights)
            weights = np.array(weights)
            weights = weights - np.max(weights)  # Numerical stability
            weights = np.exp(weights)
            weights = weights / np.sum(weights)  # Normalize to probabilities

            # Resample K generations based on importance weights
            resampled_indices = np.random.choice(
                len(posterior_generations),
                size=k_sample,
                replace=True,
                p=weights
            )

            # Extract resampled generations and their logprobs
            resampled_latents = [posterior_generations[i] for i in resampled_indices]
            resampled_weights = [weights[i] for i in resampled_indices]
            resampled_posterior_logprobs = [posterior_logprobs[i] for i in resampled_indices]
            resampled_prior_logprobs = [prior_logprobs[i] for i in resampled_indices]

            # With 50% chance, make this sample a posterior conversation, 50% chance make it a prior conversation
            conv_type = RnbencoreConfig.CONVERSATION_TYPE_PRIOR  # RnbencoreConfig.CONVERSATION_TYPE_POSTERIOR if np.random.rand() < RnbencoreConfig.POSTERIOR_PROB else RnbencoreConfig.CONVERSATION_TYPE_PRIOR

            # Process all k_sample samples
            for latent_reasoning, prior_logprob, posterior_logprob, weight in zip(
                resampled_latents,
                resampled_prior_logprobs,
                resampled_posterior_logprobs,
                resampled_weights
            ):
                if conv_type == RnbencoreConfig.CONVERSATION_TYPE_POSTERIOR:
                    conversations = [
                        {
                            "from": "human",
                            "value": previous_conversations[0]["value"] + "\n\n" + extract_action_from_conversations(previous_conversations)
                        },
                        {
                            "from": "assistant",
                            "value": latent_reasoning
                        }
                    ]
                else:  # prior
                    conversations = [
                        {
                            "from": "human",
                            "value": previous_conversations[0]["value"]
                        },
                        {
                            "from": "assistant",
                            "value": latent_reasoning + "\n\n" + extract_action_from_conversations(previous_conversations)
                        }
                    ]

                # Save result
                result = {
                    # "token": sample_item.get("token", "unknown"),
                    "image": sample_item["image"],
                    "joint_prob": prior_logprob,  # log
                    "generation_prob": posterior_logprob,  # log
                    "importance_weights": weight,  #exponentiated
                    "conversation_type": conv_type,
                    "conversations": conversations,
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                
                if conv_type == RnbencoreConfig.CONVERSATION_TYPE_PRIOR:
                    total_prior_conversations += 1
                else:
                    total_posterior_conversations += 1

            f.flush()  # Ensure data is written to disk
    
    return {
        "total_prior": total_prior_conversations,
        "total_posterior": total_posterior_conversations,
        "total": total_prior_conversations + total_posterior_conversations
    }


def run_rnbencore_inference(args: InferenceArguments):
    """
    Run Rnbencore inference with 3 main phases:
    1. Generate posterior samples
    2. Evaluate prior likelihoods
    3. Resample and save

    Args:
        args: Inference arguments
    """
    print("=" * 80)
    print(f"Starting Rnbencore inference with model: {args.model_name_or_path}")
    print(f"K_gen (posterior samples per datapoint): {args.k_gen}")
    print(f"K_sample (resamples per datapoint): {args.k_sample}")
    print("=" * 80)

    # Load dataset
    print("\n[1/4] Loading dataset...")
    data_items = load_dataset_from_config(args)

    if len(data_items) == 0:
        print("No data loaded!")
        return

    # Load processor
    print("\n[2/4] Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    print("Processor loaded.")

    # Initialize vLLM
    print("\n[3/4] Initializing vLLM model...")
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

    print(f"\nSampling parameters:")
    print(f"  - Temperature: {args.temperature}")
    print(f"  - Top-p: {args.top_p}")
    print(f"  - Max tokens: {args.max_tokens}")
    print(f"  - Batch size: {args.batch_size}")

    # Phase 1: Generate posterior samples
    print("\n[4/4] Running Rnbencore inference...")
    print("\n--- Phase 1: Generate Posterior Samples ---")
    output_path = Path(args.output_path)
    os.makedirs(output_path.parent, exist_ok=True)

    # posterior_results = generate_posterior_samples(llm, data_items, processor, args)
    # if posterior results are already found in output path, load it instead
    if (output_path.parent / "posterior_results.pkl").exists():
        print(f"Loading existing posterior results from {output_path.parent / 'posterior_results.pkl'}")
        with open(output_path.parent / "posterior_results.pkl", "rb") as f:
            posterior_results = pickle.load(f)
    else:
        posterior_results = generate_posterior_samples(llm, data_items, processor, args)
        # Save posterior results for future use
        with open(output_path.parent / "posterior_results.pkl", "wb") as f:
            pickle.dump(posterior_results, f)
    
    torch.cuda.empty_cache()
    gc.collect()

    # Phase 2: Evaluate prior likelihoods
    print("\n--- Phase 2: Evaluate Prior Likelihoods ---")
    # latent_reasoning_data = evaluate_prior_likelihoods(llm, posterior_results, processor, args)
    # similarly, check if prior likelihoods are already evaluated
    if (output_path.parent / "latent_reasoning_data.pkl").exists():
        print(f"Loading existing latent reasoning data from {output_path.parent / 'latent_reasoning_data.pkl'}")
        with open(output_path.parent / "latent_reasoning_data.pkl", "rb") as f:
            latent_reasoning_data = pickle.load(f)
    else:
        latent_reasoning_data = evaluate_prior_likelihoods(llm, posterior_results, processor, args)
        # Save latent reasoning data for future use
        with open(output_path.parent / "latent_reasoning_data.pkl", "wb") as f:
            pickle.dump(latent_reasoning_data, f)

    # Phase 3: Resample and save
    print("\n--- Phase 3: Resample and Save ---")
    stats = resample_and_save(latent_reasoning_data, output_path, args.k_sample)

    # Print summary
    print(f"\n{'='*80}")
    print(f"Rnbencore inference completed successfully!")
    print(f"Total samples processed: {stats['total']}")
    print(f"  - Prior conversations: {stats['total_prior']}")
    print(f"  - Posterior conversations: {stats['total_posterior']}")
    print(f"Results saved to: {args.output_path}")
    print(f"{'='*80}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Inference with Qwen-VL models using vLLM batch generation"
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
        default="inference_results.jsonl",
        help="Path to save results"
    )

    # Generation arguments
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p sampling"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Max tokens to generate"
    )

    # vLLM arguments
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=4096,
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

    # Rnbencore-specific arguments
    parser.add_argument(
        "--k_gen",
        type=int,
        default=8,
        help="Number of posterior generations per sample for Rnbencore"
    )
    parser.add_argument(
        "--k_sample",
        type=int,
        default=8,
        help="Number of final samples to resample per datapoint"
    )

    # Batch size argument
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for vLLM inference"
    )

    args = parser.parse_args()

    # Convert to InferenceArguments dataclass
    inference_args = InferenceArguments(
        model_name_or_path=args.model_name_or_path,
        dataset_use=args.dataset_use,
        annotation_path=args.annotation_path,
        data_path=args.data_path,
        output_path=args.output_path,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_expert_parallel=args.enable_expert_parallel,
        k_gen=args.k_gen,
        k_sample=args.k_sample,
        batch_size=args.batch_size,
    )

    run_rnbencore_inference(inference_args)


if __name__ == "__main__":
    main()