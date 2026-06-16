import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from tqdm import tqdm
from PIL import Image
from qwenvl.data import data_list


@dataclass
class InferenceArguments:
    """Arguments for inference."""
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        metadata={"help": "Path to the model checkpoint or HuggingFace model name"}
    )
    dataset_use: str = field(
        default="",
        metadata={"help": "Comma-separated list of datasets to use (e.g., 'nuscenes_trajonly')"}
    )
    annotation_path: Optional[str] = field(
        default=None,
        metadata={"help": "Direct path to annotation file (overrides dataset_use)"}
    )
    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Base path for data files"}
    )
    output_path: str = field(
        default="inference_results.jsonl",
        metadata={"help": "Path to save inference results"}
    )
    max_model_len: int = field(
        default=4096,
        metadata={"help": "Maximum model context length"}
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature"}
    )
    top_p: float = field(
        default=0.9,
        metadata={"help": "Top-p sampling parameter"}
    )
    max_tokens: int = field(
        default=2048,
        metadata={"help": "Maximum number of tokens to generate"}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples to process"}
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of GPUs for tensor parallelism"}
    )
    gpu_memory_utilization: float = field(
        default=0.9,
        metadata={"help": "GPU memory utilization (0.0-1.0)"}
    )
    enable_expert_parallel: bool = field(
        default=False,
        metadata={"help": "Enable expert parallelism for MoE models"}
    )
    k_gen: int = field(
        default=8,
        metadata={"help": "Number of posterior generations per sample for Rnbencore"}
    )
    k_sample: int = field(
        default=8,
        metadata={"help": "Number of final samples to resample per datapoint"}
    )
    batch_size: int = field(
        default=64,
        metadata={"help": "Batch size for vLLM inference"}
    )


def read_jsonl(path: str) -> List[Dict]:
    """Read JSONL file."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def read_json(path: str) -> List[Dict]:
    """Read JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
        # Handle both list and dict formats
        if isinstance(data, dict):
            return [data]
        return data


def _make_abs_paths(base: str, files: str) -> str:
    """Convert relative paths to absolute paths."""
    return str((Path(base) / files).resolve())


def prepare_inputs_for_vllm(messages: List[Dict], processor: AutoProcessor, add_generation_prompt: bool=True, truncate_prompt_tokens: Optional[int]=None) -> Dict:
    """
    Prepare inputs for vLLM inference using the same format as run_evaluation.py.

    Args:
        messages: List of message dictionaries with role and content
        processor: AutoProcessor for the model

    Returns:
        Dictionary with prompt and multi_modal_data for vLLM
    """
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    if truncate_prompt_tokens is not None:
        # Tokenize and truncate
        tokenized = processor.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=truncate_prompt_tokens
        )
        text = processor.tokenizer.decode(tokenized["input_ids"][0], skip_special_tokens=False)

    # qwen_vl_utils 0.0.14+ required
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


def build_vllm_messages(item: Dict[str, Any], base_path: Path, include_assistant: bool=False) -> List[Dict[str, Any]]:
    """
    Build vLLM-compatible messages from training data format.

    Args:
        item: Single data item with 'conversations', 'image', 'video' fields
        base_path: Base directory for resolving relative paths

    Returns:
        List of message dictionaries compatible with vLLM
    """
    # Extract and normalize images and videos
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    # Build media pools with absolute paths and load PIL Images
    image_pool = []
    for img_path in images:
        abs_path = _make_abs_paths(base_path, img_path)
        try:
            pil_image = Image.open(abs_path).convert("RGB")
            image_pool.append(pil_image)
        except Exception as e:
            print(f"Warning: Failed to load image {abs_path}: {e}")
            continue

    video_pool = []
    for vid_path in videos:
        abs_path = _make_abs_paths(base_path, vid_path)
        video_pool.append(abs_path)

    messages = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    # Use PIL Image directly (compatible with vLLM)
                    content.append({
                        "type": "image",
                        "image": image_pool.pop(0)
                    })
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError(
                            "Number of <video> placeholders exceeds the number of provided videos"
                        )
                    content.append({
                        "type": "video",
                        "video": video_pool.pop(0)
                    })
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        elif role == "assistant" and include_assistant:
            # Assistant messages contain only text
            messages.append({"role": role, "content": text})

    # Check for unused media files
    if image_pool:
        print(f"Warning: {len(image_pool)} image(s) remain unused")
    if video_pool:
        print(f"Warning: {len(video_pool)} video(s) remain unused")

    return messages


def load_dataset_from_config(args: InferenceArguments) -> List[Dict]:
    """
    Load dataset using the same configuration as training.

    Args:
        args: Inference arguments

    Returns:
        List of data items
    """

    if args.annotation_path is not None:
        # Direct annotation path provided
        annotation_paths = [args.annotation_path]
        data_paths = [args.data_path or ""]
    else:
        # Use dataset configuration
        dataset_names = args.dataset_use.split(",")
        dataset_list = data_list(dataset_names)

        annotation_paths = [d["annotation_path"] for d in dataset_list]
        data_paths = [d["data_path"] for d in dataset_list]

    all_data = []
    for ann_path, data_path in zip(annotation_paths, data_paths):
        print(f"Loading data from: {ann_path}")

        # Determine file format
        file_format = ann_path.split(".")[-1]
        if file_format == "jsonl":
            annotations = read_jsonl(ann_path)
        else:
            annotations = read_json(ann_path)

        # Add data_path to each annotation
        for ann in annotations:
            if isinstance(ann, list):
                for sub_ann in ann:
                    sub_ann["data_path"] = data_path
            else:
                ann["data_path"] = data_path

        all_data.extend(annotations)

    print(f"Total samples loaded: {len(all_data)}")
    return all_data
