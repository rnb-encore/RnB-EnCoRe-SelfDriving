import json
from typing import List, Dict, Tuple
from itertools import combinations

# NUSCENES_REASONING_TYPES = {
#     "goal": "Mission Goal:",
#     "perception": "*****Perception Results:*****",
#     "common_sense": "*****Traffic Rules:*****",
#     "experience": "*****Past Driving Experience for Reference:*****",
#     "chain_of_thought": "*****Chain of Thoughts Reasoning:*****",
#     "cot_no_objects": "Notable Objects: None"
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


def analyze_reasoning_distribution(data: List[Dict]) -> Dict[str, float]:
    """
    Analyze the distribution of reasoning types in the dataset.

    Args:
        data: List of data items, each containing reasoning information
    Returns:
        Dictionary with proportions of each reasoning type
    """
    counts = {key: 0 for key in NUSCENES_REASONING_TYPES.keys()}
    total_count = 0
    for item in data:
        assistant_message = item["conversations"][1]["value"]
        total_count += 1
        for key, marker in NUSCENES_REASONING_TYPES.items():
            if marker in assistant_message:
                counts[key] += 1
    
    distribution = {key: counts[key] / float(total_count) for key in counts}
    return distribution


def analyze_pairwise_cooccurrence(data: List[Dict]) -> Dict[Tuple[str, str], float]:
    """
    Analyze the pairwise co-occurrence of reasoning types in the dataset.
    
    Args:
        data: List of data items, each containing reasoning information
    Returns:
        Dictionary with proportions of each pair of reasoning types co-occurring
    """
    # Initialize counts for all possible pairs
    pair_counts = {}
    all_types = list(NUSCENES_REASONING_TYPES.keys())
    for i in range(len(all_types)):
        for j in range(i + 1, len(all_types)):
            pair = (all_types[i], all_types[j])
            pair_counts[pair] = 0
    
    total_count = len(data)
    
    # Count co-occurrences
    for item in data:
        assistant_message = item["conversations"][1]["value"]
        
        # Find which reasoning types are present in this message
        present_types = []
        for key, marker in NUSCENES_REASONING_TYPES.items():
            if marker in assistant_message:
                present_types.append(key)
        
        # Count all pairs of present types
        for i in range(len(present_types)):
            for j in range(i + 1, len(present_types)):
                type1 = present_types[i]
                type2 = present_types[j]
                
                # Find the pair in our pair_counts dictionary
                # Try both orderings since we need to match the key exactly
                if (type1, type2) in pair_counts:
                    pair_counts[(type1, type2)] += 1
                elif (type2, type1) in pair_counts:
                    pair_counts[(type2, type1)] += 1
    
    # Convert to proportions
    pair_distribution = {
        pair: count / float(total_count) 
        for pair, count in pair_counts.items()
    }
    
    return pair_distribution
    # return pair_counts


def create_cooccurrence_matrix(pair_distribution: Dict[Tuple[str, str], float]) -> Dict[str, Dict[str, float]]:
    """
    Create a symmetric co-occurrence matrix from pairwise distribution.
    
    Args:
        pair_distribution: Dictionary of pairwise co-occurrence proportions
    Returns:
        Nested dictionary representing the co-occurrence matrix
    """
    # Get all unique reasoning types
    types = sorted(NUSCENES_REASONING_TYPES.keys())
    
    # Initialize matrix as nested dictionary
    matrix = {}
    for type1 in types:
        matrix[type1] = {}
        for type2 in types:
            matrix[type1][type2] = 0.0
    
    # Fill in the matrix
    for (type1, type2), value in pair_distribution.items():
        matrix[type1][type2] = value
        matrix[type2][type1] = value  # Make it symmetric
    
    return matrix


def print_matrix(matrix: Dict[str, Dict[str, float]], format_percentage: bool = True, normalization: str = 'none'):
    """
    Print a formatted co-occurrence matrix.

    Args:
        matrix: Nested dictionary representing the matrix
        format_percentage: Whether to format values as percentages
        normalization: Type of normalization ('none', 'total', 'row', 'column')
                      WARNING: 'row' and 'column' normalization will break symmetry
    """

    # Create a deep copy to avoid modifying the original matrix
    import copy
    display_matrix = copy.deepcopy(matrix)

    # normalize it according to total sum, row sum, or column sum
    if normalization == 'total':
        total_sum = sum(display_matrix[type1][type2] for type1 in display_matrix for type2 in display_matrix[type1])
        if total_sum > 0:
            for type1 in display_matrix:
                for type2 in display_matrix[type1]:
                    display_matrix[type1][type2] /= total_sum
    elif normalization == 'row':
        for type1 in display_matrix:
            row_sum = sum(display_matrix[type1][type2] for type2 in display_matrix[type1])
            if row_sum > 0:
                for type2 in display_matrix[type1]:
                    display_matrix[type1][type2] /= row_sum
    elif normalization == 'column':
        for type2 in display_matrix:
            col_sum = sum(display_matrix[type1][type2] for type1 in display_matrix)
            if col_sum > 0:
                for type1 in display_matrix:
                    display_matrix[type1][type2] /= col_sum

    types = sorted(display_matrix.keys())

    # Calculate column widths
    col_width = 12
    first_col_width = max(len(t) for t in types) + 2

    # Print header
    print(" " * first_col_width, end="")
    for type2 in types:
        print(f"{type2[:col_width-2]:>{col_width}}", end="")
    print()

    # Print separator
    print("-" * (first_col_width + len(types) * col_width))

    # Print rows
    for type1 in types:
        print(f"{type1:<{first_col_width}}", end="")
        for type2 in types:
            value = display_matrix[type1][type2]
            if format_percentage:
                formatted = f"{value:.2%}"
            else:
                formatted = f"{value:.4f}"
            print(f"{formatted:>{col_width}}", end="")
        print()


def print_top_cooccurrences(pair_distribution: Dict[Tuple[str, str], float], top_n: int = 10):
    """
    Print the top N most frequent co-occurrences.
    
    Args:
        pair_distribution: Dictionary of pairwise co-occurrence proportions
        top_n: Number of top pairs to display
    """
    # Sort pairs by frequency
    sorted_pairs = sorted(pair_distribution.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {top_n} Co-occurring Reasoning Type Pairs:")
    print("-" * 50)
    for i, ((type1, type2), freq) in enumerate(sorted_pairs[:top_n], 1):
        print(f"{i}. {type1} + {type2}: {freq:.2%}")


def save_matrix_to_csv(matrix: Dict[str, Dict[str, float]], filename: str):
    """
    Save the co-occurrence matrix to a CSV file.
    
    Args:
        matrix: Nested dictionary representing the matrix
        filename: Output CSV filename
    """
    types = sorted(matrix.keys())
    
    with open(filename, 'w') as f:
        # Write header
        f.write("," + ",".join(types) + "\n")
        
        # Write rows
        for type1 in types:
            row_values = [str(matrix[type1][type2]) for type2 in types]
            f.write(type1 + "," + ",".join(row_values) + "\n")
    
    print(f"Matrix saved to {filename}")


def debug_sample(data: List[Dict], sample_index: int = 0):
    """
    Debug function to show which reasoning types are present in a specific sample.
    
    Args:
        data: List of data items
        sample_index: Index of the sample to examine
    """
    if sample_index >= len(data):
        print(f"Sample index {sample_index} out of range")
        return
    
    assistant_message = data[sample_index]["conversations"][1]["value"]
    
    print(f"\nDebug: Sample {sample_index}")
    print("-" * 50)
    print("Present reasoning types:")
    for key, marker in NUSCENES_REASONING_TYPES.items():
        if marker in assistant_message:
            print(f"  ✓ {key}")
    print("\nFirst 500 characters of message:")
    print(assistant_message[:500])


if __name__ == "__main__":
    # Example usage
    # file_path = "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_rnbencoreit0_generation_resampling.jsonl"
    file_path = "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_vqadriver_rnbencore_do05_validation/rnbencoreit0_generation.jsonl"

    # Load data
    data = load_jsonl_file(file_path)
    
    # Debug: Check a few samples to see what's present
    print("Debugging first few samples:")
    for i in range(min(3, len(data))):
        debug_sample(data, i)
    
    # Single occurrence distribution
    print("\n" + "=" * 50)
    print("Single Reasoning Type Distribution:")
    print("-" * 50)
    single_distribution = analyze_reasoning_distribution(data)
    for key, value in single_distribution.items():
        print(f"{key}: {value:.2%}")
    
    # Pairwise co-occurrence distribution
    pair_distribution = analyze_pairwise_cooccurrence(data)
    
    # Print top co-occurrences
    print_top_cooccurrences(pair_distribution)
    
    # Create and display co-occurrence matrix
    print("\nCo-occurrence Matrix (as percentages):")
    print("-" * 50)
    cooccurrence_matrix = create_cooccurrence_matrix(pair_distribution)
    print_matrix(cooccurrence_matrix, format_percentage=True, normalization='column')
    
    # Optional: Save matrix to CSV
    # save_matrix_to_csv(cooccurrence_matrix, "cooccurrence_matrix.csv")
    
    # Print some statistics
    print("\nStatistics:")
    print("-" * 50)
    print(f"Total samples: {len(data)}")
    print(f"Number of possible pairs: {len(pair_distribution)}")
    non_zero_pairs = sum(1 for v in pair_distribution.values() if v > 0)
    print(f"Number of pairs that co-occur at least once: {non_zero_pairs}")
    
    # Calculate average without numpy
    total_cooccurrence = sum(pair_distribution.values())
    avg_cooccurrence = total_cooccurrence / len(pair_distribution) if pair_distribution else 0
    print(f"Average co-occurrence rate: {avg_cooccurrence:.2%}")
