import json
import ndjson
import numpy as np
import random
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from transformers import AutoTokenizer
from itertools import product

system_message = """**Autonomous Driving Planner**
Role: You're an autonomous vehicle's brain. Given an image of the current scenario, plan a 3-second safe trajectory to avoid obstacles.

Context:
- Coordinates: X-axis is perpendicular, and Y-axis is parallel to the direction you're facing. You're at point (0,0). Units: meters.
- Goal: Plan a 3-second route using 6 waypoints (0.5s intervals).

Task:
- Based on inputs, plan a safe, feasible 3-second trajectory of 6 waypoints.

Output:
Planned Trajectory:\n[(x1,y1), (x2,y2), ... , (x6,y6)]
"""

def generate_all_conversations_for_sample(data_sample, post_dropout_rate=0.2, num_samples=64):
    """
    Generate 128 conversations for a single data sample:
    - 64 prior conversations (power set of reasoning types)
    - 64 posterior conversations (with 20% dropout of reasoning traces)
    
    Returns:
        List of tuples: [(token, user_message, assistant_message), ...]
    """
    all_conversations = []
    
    # Part 1: Generate 64 prior conversations (power set)
    # These are all combinations of including/not including each reasoning type
    reasoning_types = ['use_goal', 'use_perception', 'use_commonsense', 
                      'use_experience', 'use_cot', 'use_reasoning']
    
    # Generate prior conversations with 50% dropout for each reasoning type
    for i in range(num_samples):
        # Independently sample each reasoning type with 50% probability
        dropout_kwargs = {}
        for reasoning_type in reasoning_types:
            # 50% chance to include this reasoning type (50% dropout)
            dropout_kwargs[reasoning_type] = random.random() < 0.5
        
        dropout_kwargs['construct_posterior'] = False  # Priors don't use posterior
        
        token, user_message, assistant_message = generate_conversation(
            data_sample,
            **dropout_kwargs
        )
        all_conversations.append((token, user_message, assistant_message))
    
    # Part 2: Generate 64 posterior conversations with dropout
    # Each reasoning type is independently included with 80% probability (20% dropout)
    for i in range(num_samples):
        # Independently sample each reasoning type with 80% probability
        dropout_kwargs = {}
        for reasoning_type in reasoning_types:
            # 80% chance to include this reasoning type (20% dropout)
            dropout_kwargs[reasoning_type] = random.random() < (1 - post_dropout_rate)
        
        dropout_kwargs['construct_posterior'] = True  # Posteriors use construct_posterior
        
        token, user_message, assistant_message = generate_conversation(
            data_sample,
            **dropout_kwargs
        )
        all_conversations.append((token, user_message, assistant_message))
    
    return all_conversations


def generate_conversation(
        data_sample, 
        use_goal=True, 
        use_perception=True,
        use_commonsense=True,
        use_experience=True,
        use_cot=True,
        use_reasoning=True,
        construct_posterior=False, 
    ):
    token = data_sample["token"]
    ego = data_sample["ego"]
    perception = data_sample["perception"]
    commonsense = data_sample["commonsense"]
    experiences =  data_sample["experiences"]
    reasoning = data_sample["reasoning"]
    # long_experiences = data_sample["long_experiences"] if "long_experiences" in data_sample else None
    chain_of_thoughts = data_sample["chain_of_thoughts"] if "chain_of_thoughts" in data_sample else ""
    planning_target = data_sample["planning_target"] if "planning_target" in data_sample else None

    ego_info = ego.split("\nMission Goal:")[0]
    mission_goal = "Mission Goal:" + ego.split("\nMission Goal:")[1]

    user_message = "<image>\n" + system_message + ego_info

    assistant_message = ""
    if use_goal:
        assistant_message += mission_goal
    if use_perception:
        assistant_message += perception
    if use_commonsense:
        assistant_message += commonsense
    if use_experience:
        if experiences:
            assistant_message += experiences
        else:
            assistant_message += "\nExperience: None.\n"
    if use_cot:
        assistant_message += chain_of_thoughts
    if use_reasoning:
        assistant_message += reasoning
    
    if construct_posterior:
        user_message += planning_target
    else:
        assistant_message += planning_target
    
    return token, user_message, assistant_message

# mission goal, perception, commonsense, experiences, chain_of_thoughts, reasoning


def get_nuscenes_image_path(nusc, sample_token):
    sample = nusc.get('sample', sample_token)
    camera_data_token = sample['data']['CAM_FRONT']
    camera_data = nusc.get('sample_data', camera_data_token)
    image_path = camera_data['filename']
    return image_path


def generate_drivevla_traj_finetune_data(
    data_path, 
    data_file, 
    image_data_path, 
    sample_ratio=1.0, 
    use_gt_cot=False,
    model_name="Qwen/Qwen3-VL-4B-Instruct",  # Can be changed to other Qwen VL models
    max_input_tokens=32768,  # Qwen3 VL supports much longer context
    verbose_progress=True,
    split="train",
    post_dropout_rate=0.2,
    seed=42,
    num_samples=64,
):
    """
    Generate fine-tuning data for Qwen3 VL model.
    Saves data incrementally as JSONL format, one line per conversation.
    
    Args:
        data_path: Path to data directory
        data_file: Name of the data file
        image_data_path: Path to NuScenes image data
        sample_ratio: Ratio of samples to use
        use_gt_cot: Whether to use ground truth chain of thoughts
        model_name: Name of the Qwen model for tokenizer
        max_input_tokens: Maximum number of input tokens (Qwen supports up to 32k)
        verbose_progress: Whether to print progress
        split: Data split (train/val/test)
        post_dropout_rate: Dropout rate for posterior conversations
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Initialize Qwen tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='left'
        )
    except Exception as e:
        print(f"Error loading tokenizer for {model_name}: {e}")
        print("Using fallback tokenizer configuration...")
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            trust_remote_code=True,
            padding_side='left'
        )
    
    # Load data samples
    data_samples = json.load(open(Path(data_path) / Path(data_file), 'r'))

    # Create nuscenes object
    nusc = NuScenes(version='v1.0-trainval', dataroot=image_data_path, verbose=True)
    
    # Prepare output file path
    saved_file_name = f"nuscenes_agentdriver_rnbencoreit0_{split}.jsonl"
    output_path = Path(data_path) / Path(saved_file_name)
    
    # Create a temporary file for atomic writes
    temp_output_path = output_path.with_suffix('.jsonl.tmp')
    
    # Statistics tracking
    num_user_tokens = 0
    num_assistant_tokens = 0
    max_assistant_tokens = 0
    conversation_type_counts = {
        'prior': 0,
        'posterior': 0
    }
    
    # Sample data if needed
    sample_size = int(len(data_samples) * sample_ratio)
    if sample_ratio < 1.0:
        data_samples = random.sample(data_samples, sample_size)
    
    invalid_tokens = []
    total_conversations = 0
    
    print(f"Processing {len(data_samples)} data samples...")
    print(f"Each sample will generate 128 conversations (64 priors + 64 posteriors)")
    print(f"Total conversations to generate: {len(data_samples) * 128}")
    print(f"Output file: {output_path}")
    print("=" * 60)
    
    # Open file for incremental writing
    with open(temp_output_path, 'w') as f:
        for idx, data_sample in enumerate(data_samples):
            if verbose_progress and idx % 10 == 0:
                print(f"Processing data sample {idx}/{len(data_samples)}... "
                      f"({total_conversations} conversations saved so far)")
            
            try:
                # Generate all 128 conversations for this sample
                sample_conversations = generate_all_conversations_for_sample(
                    data_sample, 
                    post_dropout_rate=post_dropout_rate,
                    num_samples=num_samples
                )
                
                # Get image path once for this sample (same for all conversations)
                token = data_sample["token"]
                image_path = get_nuscenes_image_path(nusc, token)
                
                # Process each of the 128 conversations
                for conv_idx, (token, user_message, assistant_message) in enumerate(sample_conversations):
                    
                    # Track conversation type
                    if conv_idx < num_samples:
                        conversation_type_counts['prior'] += 1
                        conv_type = "prior"
                    else:
                        conversation_type_counts['posterior'] += 1
                        conv_type = "posterior"
                    
                    assert assistant_message is not None, f"Assistant message is None for token {token}"
                    
                    # Print first few samples for verification
                    if verbose_progress and idx == 0 and conv_idx < 3:
                        print(f"\n********** Sample {idx}, Conversation {conv_idx} ({conv_type}) **********")
                        print("Token:", token)
                        print("User message preview:", user_message[:200] + "...")
                        print("Assistant message preview:", assistant_message[:200] + "...")
                        print("Image path:", image_path)
                        print("=" * 50)
                    
                    # Format for Qwen VL fine-tuning
                    conversation = {
                        "token": token,
                        "image": image_path,
                        "conversation_type": conv_type,  # Add metadata about conversation type
                        "conversation_index": conv_idx,  # Index within the 128 conversations
                        "sample_index": idx,  # Original sample index
                        "conversations": [
                            {
                                "from": "human",
                                "value": user_message
                            },
                            {
                                "from": "assistant",
                                "value": assistant_message
                            }
                        ]
                    }
                    
                    # Tokenize to count tokens
                    user_tokens = tokenizer.encode(user_message, add_special_tokens=False)
                    assistant_tokens = tokenizer.encode(assistant_message, add_special_tokens=False)
                    
                    # Account for image tokens
                    image_token_count = 256
                    
                    num_user_tokens += len(user_tokens)
                    num_assistant_tokens += len(assistant_tokens)
                    if len(assistant_tokens) > max_assistant_tokens:
                        max_assistant_tokens = len(assistant_tokens)
                    
                    # Calculate total input tokens
                    num_input_tokens = len(user_tokens) + image_token_count
                    
                    if num_input_tokens > max_input_tokens:
                        if len(invalid_tokens) < 100:  # Track more invalid samples for better debugging
                            invalid_tokens.append({
                                "token": token,
                                "sample_idx": idx,
                                "conv_idx": conv_idx,
                                "conv_type": conv_type,
                                "token_count": num_input_tokens
                            })
                    
                    # Write each conversation as a separate line in JSONL
                    f.write(json.dumps(conversation) + '\n')
                    total_conversations += 1
                
                # Flush to disk after each sample (128 conversations)
                f.flush()
                    
            except Exception as e:
                print(f"Error processing sample {idx} (token: {data_sample.get('token', 'unknown')}): {e}")
                continue
    
    # Atomic move: rename temp file to final output file
    temp_output_path.rename(output_path)
    
    # Calculate total tokens
    num_language_tokens = num_user_tokens + num_assistant_tokens
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("#### Rnbencore It0 Data Generation Summary ####")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Maximum input tokens: {max_input_tokens}")
    print(f"Number of original data samples: {len(data_samples)}")
    print(f"Conversations per sample: {num_samples * 2} ({num_samples} prior + {num_samples} posterior)")
    print(f"Total conversations generated: {total_conversations}")
    print(f"  - Prior conversations: {conversation_type_counts['prior']}")
    print(f"  - Posterior conversations: {conversation_type_counts['posterior']}")
    print(f"\nToken Statistics:")
    print(f"  - Total user tokens: {num_user_tokens:,}")
    print(f"  - Total assistant tokens: {num_assistant_tokens:,}")
    print(f"  - Total language tokens: {num_language_tokens:,}")
    print(f"  - Max assistant tokens in single sample: {max_assistant_tokens}")
    if total_conversations > 0:
        print(f"  - Avg tokens per conversation: {num_language_tokens / total_conversations:.2f}")
    
    if invalid_tokens:
        print(f"\nConversations exceeding token limit ({max_input_tokens} tokens): {len(invalid_tokens)}")
        # Group by sample index to show which samples have issues
        print(invalid_tokens)
    
    return total_conversations, invalid_tokens


if __name__ == "__main__":
    # Example usage with Qwen3 VL
    split = "train"
    seed = 42
    num_samples = 32
    
    # Generate the data
    total_convs, invalid = generate_drivevla_traj_finetune_data(
        data_path="data/finetune",
        data_file=f"data_samples_{split}.json",
        image_data_path='/path/to/nuscenes',
        sample_ratio=1.0,  # Use all data
        use_gt_cot=True,
        model_name="Qwen/Qwen3-VL-4B-Instruct",  # Specify the Qwen model you're using
        max_input_tokens=32768,  # Qwen3 VL can handle much longer context
        verbose_progress=True,
        split=split,
        post_dropout_rate=0.2,
        seed=seed,
        num_samples=num_samples
    )
