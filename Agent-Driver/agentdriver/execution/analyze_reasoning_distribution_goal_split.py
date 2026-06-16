import json
from typing import List, Dict

# NUSCENES_REASONING_TYPES = {
#     "goal": "Mission Goal:",
#     "perception": "*****Perception Results:*****",
#     "common_sense": "*****Traffic Rules:*****",
#     "experience": "*****Past Driving Experience for Reference:*****",
#     "chain_of_thought": "*****Chain of Thoughts Reasoning:*****",
#     "cot_no_objects": "Notable Objects: None",
#     "double_count": "*****Chain of Thoughts Reasoning:*****"
# }
NUSCENES_REASONING_TYPES = {
    "What is the mission goal?": "<What is the mission goal?>",
    "What do you perceive in the scene?": "<What do you perceive in the scene?>",
    "What is a similar past experience?": "<What is a similar past experience?>",
    "What are the collidable objects?": "<What are the collidable objects?>",
    "What is the driving plan?": "<What is the driving plan?>",
    "What are the notable objects?": "<What are the notable objects?>",
}


def load_jsonl_file(file_path: str) -> List[Dict]:
    """
    Load a JSONL file and return a list of dictionaries.

    Args:
        file_path: Path to the JSONL file
    Returns:
        List of dictionaries loaded from the file
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def analyze_reasoning_distribution(data: List[Dict]) -> Dict[str, int]:
    """
    Analyze the distribution of reasoning types in the dataset.

    Args:
        data: List of data items, each containing reasoning information
    Returns:
        Dictionary with counts of each reasoning type
    """
    data_samples = json.load(open("/path/to/Agent-Driver/data/finetune/data_samples_train.json", 'r'))

    # counts = {key: 0 for key in NUSCENES_REASONING_TYPES.keys()}
    # save counts by goal type
    counts = {}
    total_counts = {}
    # total_count = 0
    for i, item in enumerate(data):
        assistant_message = item["conversations"][1]["value"]
        token = data_samples[i]["token"]
        mission_goal = data_samples[i]["ego"].split("\nMission Goal:")[1].strip()
        # if mission_goal == "LEFT" or mission_goal == "RIGHT":
        #     mission_goal = "TURN"
        if mission_goal not in counts:
            counts[mission_goal] = {key: 0 for key in NUSCENES_REASONING_TYPES.keys()}
            total_counts[mission_goal] = 0
        total_counts[mission_goal] += 1

        # total_count += 1
        for key, marker in NUSCENES_REASONING_TYPES.items():
            if marker in assistant_message:
                counts[mission_goal][key] += 1
            # count number of times the marker occurs in the assistant message
        

    
    # distribution = {key: counts[key] / float(total_count) for key in counts}
    # normalize counts by goal type
    distribution = {}
    for goal, goal_counts in counts.items():
        total_goal_count = total_counts[goal]
        distribution[goal] = {key: goal_counts[key] / float(total_goal_count) for key in goal_counts}
    return distribution


if __name__ == "__main__":
    # Example usage
    # file_path = "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_rnbencoreit0_generation_resampling.jsonl"
    # file_path = "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_vqadriver_rnbencore/rnbencoreit0_generation.jsonl"
    # file_path = "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_vqadriver_rnbencore_rerun/rnbencoreit0_generation.jsonl"
    file_path = "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_vqadriver_rnbencore_do05_validation/rnbencoreit0_generation.jsonl"
    # file_path = "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_vqadriver_rnbencore_do03_prioronly/rnbencoreit0_generation.jsonl"
    # file_path = "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_vqadriver_rnbencore_do05_prioronly/rnbencoreit0_generation.jsonl"
    # file_path = "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_vqadriver_rnbencore_do05_kgen16/rnbencoreit0_generation.jsonl"
    data = load_jsonl_file(file_path)
    distribution = analyze_reasoning_distribution(data)
    print("Reasoning Type Distribution:")
    for goal, value in distribution.items():
        print(f"{goal}:")
        for key, percentage in value.items():
            print(f"  {key}: {percentage:.2%}")
        print()
