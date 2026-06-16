#!/usr/bin/env python3
"""
Inference script for Qwen-VL models using vLLM offline batch generation.
Supports Qwen2-VL, Qwen2.5-VL, and Qwen3-VL models.
Loads data in the same format as the training script.
"""

import os
import json
import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from transformers import AutoProcessor
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


def prepare_batch_inputs(
    data_items: List[Dict],
    processor: AutoProcessor,
    max_samples: Optional[int] = None
) -> tuple:
    """
    Prepare all inputs for vLLM batch inference.

    Args:
        data_items: List of data items from dataset
        processor: AutoProcessor for the model
        max_samples: Maximum number of samples to process

    Returns:
        Tuple of (vllm_inputs, metadata_list)
    """
    if max_samples is not None:
        data_items = data_items[:max_samples]

    vllm_inputs = []
    metadata_list = []

    print("Preparing batch inputs...")
    for idx, item in enumerate(tqdm(data_items, desc="Processing samples")):
        try:
            base_path = Path(item.get("data_path", ""))

            # Build messages in vLLM format
            messages = build_vllm_messages(item, base_path)

            # Prepare inputs for vLLM using the prepare_inputs_for_vllm function
            vllm_input = prepare_inputs_for_vllm(messages, processor)
            vllm_inputs.append(vllm_input)

            # Store metadata for later analysis
            metadata = {
                "sample_id": idx,
                "original_item": item,
                # "token": item["token"],
                # "num_turns": len(item.get("conversations", [])),
                "has_image": "image" in item and item["image"],
                "has_video": "video" in item and item["video"],
            }
            metadata_list.append(metadata)

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

    print(f"Prepared {len(vllm_inputs)} samples for batch inference")
    return vllm_inputs, metadata_list


def run_inference(args: InferenceArguments):
    """
    Run inference using vLLM batch generation.

    Args:
        args: Inference arguments
    """
    print("=" * 80)
    print(f"Starting inference with model: {args.model_name_or_path}")
    print("=" * 80)

    # Load dataset
    print("\n[1/5] Loading dataset...")
    data_items = load_dataset_from_config(args)

    if len(data_items) == 0:
        print("No data loaded!")
        return

    # Load processor
    print("\n[2/5] Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    print("Processor loaded.")

    # Prepare batch inputs
    print("\n[3/5] Preparing batch inputs...")
    vllm_inputs, metadata_list = prepare_batch_inputs(
        data_items,
        processor,
        max_samples=args.max_samples
    )

    if len(vllm_inputs) == 0:
        print("No valid samples to process!")
        return

    # Initialize vLLM
    print("\n[4/5] Initializing vLLM model...")
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

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    print(f"\nSampling parameters:")
    print(f"  - Temperature: {args.temperature}")
    print(f"  - Top-p: {args.top_p}")
    print(f"  - Max tokens: {args.max_tokens}")

    # Run batch inference
    print(f"\n[5/5] Running batch inference on {len(vllm_inputs)} samples...")

    try:
        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
        print("Batch inference completed successfully!")
    except Exception as e:
        print(f"Error during inference: {e}")
        return

    # Save results
    print(f"\nSaving results to: {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    results = []
    with open(args.output_path, "w") as f:
        for output, metadata in zip(outputs, metadata_list):
            # Extract the generated text
            generated_text = output.outputs[0].text

            result = {
                "sample_id": metadata["sample_id"],
                "token": metadata["original_item"]["token"],
                "generated_text": generated_text,
                # "num_turns": metadata["num_turns"],
                "has_image": metadata["has_image"],
                "has_video": metadata["has_video"],
                "original_conversations": metadata["original_item"].get("conversations", []),
                "finish_reason": output.outputs[0].finish_reason,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            results.append(result)

    print(f"\n{'='*80}")
    print(f"Inference completed successfully!")
    print(f"Total samples processed: {len(results)}")
    print(f"Results saved to: {args.output_path}")
    print(f"{'='*80}")

    # Print sample results
    if results:
        print("\n--- Sample Result ---")
        sample = results[0]
        print(f"Sample ID: {sample['sample_id']}")
        # print(f"Num turns: {sample['num_turns']}")
        print(f"Has image: {sample['has_image']}")
        print(f"Has video: {sample['has_video']}")
        print(f"Generated text (first 300 chars):\n{sample['generated_text'][:300]}...")
        print("-" * 80)


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
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process"
    )

    # Generation arguments
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Max tokens to generate"
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
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_expert_parallel=args.enable_expert_parallel,
    )

    run_inference(inference_args)


if __name__ == "__main__":
    main()
