import json
import random
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from transformers import AutoTokenizer

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


def generate_vqa_conversation(
        data_sample, 
        use_goal=False, 
        use_perception=False,
        use_experience=False, 
        use_cot_collidable=False,
        use_cot_plan=False,
        use_reasoning_objs=False,
    ):
    token = data_sample["token"]
    ego = data_sample["ego"]
    perception = data_sample["perception"]
    commonsense = data_sample["commonsense"]
    experiences =  data_sample["experiences"]
    long_experiences = data_sample["long_experiences"] if "long_experiences" in data_sample else None
    reasoning = data_sample["reasoning"]
    chain_of_thoughts = data_sample["chain_of_thoughts"] if "chain_of_thoughts" in data_sample else ""
    planning_target = data_sample["planning_target"] if "planning_target" in data_sample else None

    # Extract ego information separately from the mission goal
    split_ego = ego.split("\nMission Goal:")
    ego_info = split_ego[0]

    user_message = "<image>\n" + system_message + commonsense + ego_info
    assistant_message = ""

    if use_goal:
        mission_goal = "<What is the mission goal?>" + split_ego[1]
        assistant_message += mission_goal

    if use_perception:

        perception_string = perception.split("*****Perception Results:*****")[1]
        if len(perception_string.strip()) == 0:
            print(f"Warning: Perception string for token {token} is empty.")
            perception_string = "\nNone."
        perception_string = "<What do you perceive in the scene?>" + perception_string
        assistant_message += perception_string

    if use_experience:
        if experiences:
            experience_string = experiences.split("*****Past Driving Experience for Reference:*****")[1]
            if len(experience_string.strip()) == 0:
                print(f"Warning: Experience string for token {token} is empty.")
                experience_string = "\nNone."
            experience_string = "<What is a similar past experience?>" + experience_string
        elif long_experiences:
            experience_string = "<What is a similar past experience?>" + long_experiences.split("*****Past Driving Experience for Reference:*****")[1]
        else:
            experience_string = "<What is a similar past experience?>\nNo similar past experience available."
        assistant_message += experience_string
    
    # Extract collidable objects and driving plan from chain of thoughts
    if use_cot_collidable:
        # pull text between "\nThoughts:" and "\nDriving Plan:"
        collidable_objects = chain_of_thoughts.split("\nThoughts:")[1].split("\nDriving Plan:")[0]
        if len(collidable_objects.strip()) == 0:
            print(f"Warning: Collidable objects string for token {token} is empty.")
            collidable_objects = "\nNone."
        collidable_string = "<What are the collidable objects?>" + collidable_objects
        assistant_message += collidable_string

    if use_cot_plan:
    # pull text after "\nDriving Plan:"
        driving_plan = chain_of_thoughts.split("\nDriving Plan:")[1]
        if len(driving_plan.strip()) == 0:
            print(f"Warning: Driving plan string for token {token} is empty.")
            driving_plan = "\nNone."
        driving_plan_string = "<What is the driving plan?>" + driving_plan
        assistant_message += driving_plan_string

    # extract notable objects from reasoning
    if use_reasoning_objs:
        # pull text between "\nThoughts:" and "\nDriving Plan:"
        notable_objects = reasoning.split("\nThoughts:")[1].split("\nDriving Plan:")[0]
        if len(notable_objects.strip()) == 0:
            print(f"Warning: Notable objects string for token {token} is empty.")
            notable_objects = "\nNone."
        notable_string = "<What are the notable objects?>" + notable_objects
        assistant_message += notable_string


    assistant_message += "\n\n" + planning_target
    
    return token, user_message, assistant_message


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
    model_name="Qwen/Qwen3-VL-4B-Instruct",  # Can be changed to other Qwen VL models
    max_input_tokens=32768,  # Qwen3 VL supports much longer context
    verbose_progress=True,
    split="train",
    save_file_name="nuscenes_fulltrace_vqadriver_train.jsonl"
):
    """
    Generate fine-tuning data for Qwen3 VL model.
    
    Args:
        data_path: Path to data directory
        data_file: Name of the data file
        image_data_path: Path to NuScenes image data
        sample_ratio: Ratio of samples to use
        use_gt_cot: Whether to use ground truth chain of thoughts
        model_name: Name of the Qwen model for tokenizer
        max_input_tokens: Maximum number of input tokens (Qwen supports up to 32k)
        verbose_progress: Whether to print progress
    """
    
    # Initialize Qwen tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='left'  # Important for VL models
        )
    except Exception as e:
        print(f"Error loading tokenizer for {model_name}: {e}")
        print("Using fallback tokenizer configuration...")
        # Fallback to a basic Qwen tokenizer if specific model not available
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            trust_remote_code=True,
            padding_side='left'
        )
    
    # Load data samples
    data_samples = json.load(open(Path(data_path) / Path(data_file), 'r'))

    # Create nuscenes object
    nusc = NuScenes(version='v1.0-trainval', dataroot=image_data_path, verbose=True)
    
    # Statistics tracking
    num_user_tokens = 0
    num_assistant_tokens = 0
    max_assistant_tokens = 0
    
    # Sample data if needed
    sample_size = int(len(data_samples) * sample_ratio)
    if sample_ratio < 1.0:
        data_samples = random.sample(data_samples, sample_size)
    
    invalid_tokens = []
    conversations_trainval_set = []
    
    for idx, data_sample in enumerate(data_samples):
        if verbose_progress and idx % 100 == 0:
            print(f"Processing sample {idx}/{len(data_samples)}...")
        
        try:
            token, user_message, assistant_message = generate_vqa_conversation(
                data_sample, 
            )

            image_path = get_nuscenes_image_path(nusc, token)

            assert assistant_message is not None, f"Assistant message is None for token {token}"

            if verbose_progress and idx < 5:  # Print first 5 samples for verification
                print("********** Sample Verification **********")
                print("Token:", token)
                print("user_message:", user_message)
                print("assistant_message:", assistant_message)
                print("image_path:", image_path)
                print("===========================================\n")
            
            # Format for Qwen VL fine-tuning
            # Note: Qwen VL may use different format, adjust if needed
            conversations = {
                "token": token,
                "image": image_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": user_message
                    },
                    {
                        "from": "assistant",  # Qwen might prefer "assistant" over "gpt"
                        "value": assistant_message
                    }
                ]
            }
            
            # Tokenize to count tokens
            # For Qwen VL, we need to handle image tokens specially
            # The tokenizer might add special tokens for images
            user_tokens = tokenizer.encode(user_message, add_special_tokens=False)
            assistant_tokens = tokenizer.encode(assistant_message, add_special_tokens=False)
            
            # Account for image tokens (Qwen VL typically uses special tokens for images)
            # This is an approximation - actual image token count may vary
            image_token_count = 256  # Typical for vision transformers, adjust based on Qwen's documentation
            
            num_user_tokens += len(user_tokens)
            num_assistant_tokens += len(assistant_tokens)
            if len(assistant_tokens) > max_assistant_tokens:
                max_assistant_tokens = len(assistant_tokens)
            
            # Calculate total input tokens (including image tokens)
            num_input_tokens = len(user_tokens) + image_token_count
            
            if num_input_tokens > max_input_tokens:
                print(f"Warning: Token {token} has {num_input_tokens} tokens, "
                      f"which exceeds the limit of {max_input_tokens} tokens.")
                invalid_tokens.append({
                    "token": token,
                    "token_count": num_input_tokens
                })
                
                # Optionally, you can try to truncate or skip this sample
                # For now, we'll include it but mark it as potentially problematic
            
            conversations_trainval_set.append(conversations)
            
        except Exception as e:
            print(f"Error processing token {data_sample.get('token', 'unknown')}: {e}")
            continue
    
    # Calculate total tokens
    num_language_tokens = num_user_tokens + num_assistant_tokens
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("#### Fine-tuning Data Generation Summary ####")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Maximum input tokens: {max_input_tokens}")
    print(f"Number of total samples: {len(conversations_trainval_set)}")
    print(f"Number of user tokens: {num_user_tokens:,}")
    print(f"Number of assistant tokens: {num_assistant_tokens:,}")
    print(f"Number of total language tokens: {num_language_tokens:,}")
    print(f"Maximum assistant tokens in a single sample: {max_assistant_tokens}")
    print(f"Average tokens per sample: {num_language_tokens / len(conversations_trainval_set):.2f}")
    
    if invalid_tokens:
        print(f"\nSamples exceeding token limit ({max_input_tokens} tokens): {len(invalid_tokens)}")
        for invalid in invalid_tokens[:5]:  # Show first 5
            print(f"  - Token: {invalid['token']}, Count: {invalid['token_count']}")
        if len(invalid_tokens) > 5:
            print(f"  ... and {len(invalid_tokens) - 5} more")
    
    # Save the data
    jsonl_output_path = Path(data_path) / Path(save_file_name)
    
    with open(jsonl_output_path, "w") as f:
        for conversation in conversations_trainval_set:
            f.write(json.dumps(conversation) + "\n")
    
    print(f"JSONL data saved to: {jsonl_output_path}")
    
    return conversations_trainval_set, invalid_tokens


if __name__ == "__main__":
    # Example usage with Qwen3 VL
    split = "val" #"train"
    generate_drivevla_traj_finetune_data(
        data_path="data/finetune",
        data_file=f"data_samples_{split}.json",
        image_data_path='/path/to/nuscenes',
        sample_ratio=1.0,  # Use all data
        model_name="Qwen/Qwen3-VL-4B-Instruct",  # Specify the Qwen model you're using
        max_input_tokens=32768,  # Qwen3 VL can handle much longer context
        verbose_progress=True,
        split=split,
        # save_file_name=f"nuscenes_fulltrace_vqadriver_{split}.jsonl"
        # save_file_name=f"nuscenes_noexp_nonotableobj_vqadriver_{split}.jsonl"
        save_file_name=f"nuscenes_trajonly_vqadriver_{split}.jsonl"
    )
