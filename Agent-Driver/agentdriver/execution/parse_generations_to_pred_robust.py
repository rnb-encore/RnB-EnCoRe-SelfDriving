import json
import ast
import numpy as np
import pickle
import re
import argparse


def extract_trajectory(generated_text: str):
    """
    Extract trajectory from text after "Planned Trajectory:" marker.
    Handles various formats: [(x,y), ...], [[x,y], ...], etc.
    """
    # Split on "Planned Trajectory:" and take everything after
    parts = generated_text.split("Planned Trajectory:")
    if len(parts) < 2:
        raise ValueError("Generated text does not contain 'Planned Trajectory:'")
    
    traj_text = parts[1].strip()
    
    # Try to find the trajectory list/array
    patterns = [
        r'(\[[\s\S]*?\])',  # Matches [...] including nested
        r'(\([\s\S]*?\))',  # Matches (...) including nested
    ]
    
    for pattern in patterns:
        match = re.search(pattern, traj_text)
        if match:
            traj_str = match.group(1)
            
            # First, try to fix malformed numbers like "11.  .6" -> "11.6"
            cleaned_traj_str = fix_malformed_numbers(traj_str)
            
            try:
                # Try ast.literal_eval with cleaned string
                traj = ast.literal_eval(cleaned_traj_str)
                
                # Convert to numpy array
                traj_array = np.array(traj)
                
                # Ensure it's 2D with shape (n, 2)
                if traj_array.ndim == 2 and traj_array.shape[1] == 2:
                    return traj_array
                    
            except (SyntaxError, ValueError) as e:
                print(f"ast.literal_eval failed: {e}")
                # If ast.literal_eval fails, try the improved parse approach
                traj_points = parse_trajectory_pairs(traj_str)
                if traj_points:
                    return np.array(traj_points)
    
    raise ValueError("Could not parse trajectory from generated text")
    

def fix_malformed_numbers(text: str):
    """
    Fix malformed numbers like "11.  .6" -> "11.6"
    """
    # Pattern to match malformed decimal numbers with spaces
    # Matches: digit(s) + dot + space(s) + dot + digit(s)
    pattern = r'(\d+)\.\s+\.(\d+)'
    fixed_text = re.sub(pattern, r'\1.\2', text)
    return fixed_text


def parse_trajectory_pairs(text: str):
    """
    Parse trajectory by extracting coordinate pairs.
    Handles multiple formats and malformed numbers.
    """
    # First fix malformed numbers
    text = fix_malformed_numbers(text)
    
    # List of patterns to try, in order of preference
    patterns = [
        # Format: (x,y) or (x, y)
        (r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)', 'parentheses with comma'),
        
        # Format: [x,y] or [x, y]
        (r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]', 'brackets with comma'),
        
        # Format: ['x' 'y'] or ["x" "y"] (space-separated with quotes)
        (r'\[\s*[\'"](-?\d+\.?\d*)[\'\"]\s+[\'"](-?\d+\.?\d*)[\'\"]\s*\]', 'brackets with quotes and space'),
        
        # Format: [x y] (space-separated without quotes)
        # Note: This needs to be last as it's the most permissive
        (r'\[\s*(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s*\]', 'brackets with space'),
    ]
    
    for pattern, description in patterns:
        matches = re.findall(pattern, text)
        if matches:
            print(f"Matched format: {description}")  # Debug info, can remove
            points = [[float(x), float(y)] for x, y in matches]
            return points
    
    # If no patterns matched, try one more approach:
    # Look for any sequence of number pairs (most permissive)
    # This handles cases where the outer brackets might be malformed
    if "Trajectory" in text or "[" in text:
        # Remove all quotes and extra brackets
        cleaned = re.sub(r'[\'"]', '', text)
        # Find consecutive number pairs
        number_pattern = r'(-?\d+\.?\d*)\s+(-?\d+\.?\d*)'
        matches = re.findall(number_pattern, cleaned)
        
        # Filter out unlikely coordinate pairs (e.g., from other parts of text)
        # by checking if they appear after "Trajectory" or within brackets
        if matches and len(matches) >= 2:  # Expect at least 2 points in a trajectory
            points = [[float(x), float(y)] for x, y in matches]
            # Basic sanity check - trajectories usually have reasonable values
            if all(-1000 < p[0] < 1000 and -1000 < p[1] < 1000 for p in points):
                print(f"Matched format: fallback number pairs")  # Debug info
                return points
    
    return None
            

def parse_jsonl_to_trajectories(jsonl_file_path, save_file_name):
    """
    Parse JsonL file and save trajectories as pickle.
    
    Args:
        jsonl_file_path: Path to input JsonL file
        save_file_name: Path to output pickle file
    
    Returns:
        Dictionary mapping tokens to trajectory numpy arrays
    """
    pred_trajs_dict = {}
    invalid_tokens = []
    
    # Load and process JsonL file
    with open(jsonl_file_path, 'r') as f:
        for line in f:
            token = None  # Initialize token to None for error handling
            try:
                # Parse JSON line
                data = json.loads(line.strip())
                
                # Check for required fields
                token = data.get('token')
                generated_text = data.get('generated_text')
                
                # if not token or not generated_text:
                #     raise ValueError(f"Missing required fields - token: {bool(token)}, generated_text: {bool(generated_text)}")
                
                # # Use regex to get only text after "Planned Trajectory:"
                # match = re.search(r'Planned Trajectory:\s*(\[.*?\])', generated_text, re.DOTALL)
                # if not match:
                #     raise ValueError("Could not find 'Planned Trajectory:' followed by trajectory list")
                
                # # Extract the trajectory list string
                # traj_str = match.group(1)
                # traj = ast.literal_eval(traj_str)
                # traj = np.array(traj)
                traj = extract_trajectory(generated_text)
                
                # Store in dictionary
                pred_trajs_dict[token] = traj
            
            except ValueError as ve:
                print(f"Parsing error: {ve}")
                invalid_tokens.append(token)
                print(f"Unparse token: {token}")
                # # Add default in-place trajectory (e.g., zeros)
                # pred_trajs_dict[token] = np.zeros((6, 2))
                continue
                
            except Exception as e:
                print(f"An error occurred: {e}")
                
                invalid_tokens.append(token)
                print(f"Invalid token: {token}")
                # # Add default in-place trajectory (e.g., zeros)
                # pred_trajs_dict[token] = np.zeros((6, 2))
                continue
    
    print("#### Invalid Tokens ####")
    # print(f"{invalid_tokens}")
    print(f"Total invalid tokens: {len(invalid_tokens)}")
    
    # Save to pickle
    with open(save_file_name, "wb") as f:
        pickle.dump(pred_trajs_dict, f)
    
    return pred_trajs_dict


# Example usage
if __name__ == "__main__":
    # Specify your file paths
    # input_file = "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_trajonly.jsonl"
    # output_file = "./results/nuscenes_trajonly.pkl"
    # input_file = "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_fulltrace_agentdriver.jsonl"
    # output_file = "./results/nuscenes_fulltrace_agentdriver.pkl"
    # input_file = "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_fulltrace_agentdriver_maxlen4096.jsonl"
    # output_file = "./results/nuscenes_fulltrace_agentdriver_maxlen4096.pkl"
    # input_file = "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_fulltrace_agentdriver_maxlen6000.jsonl"
    # output_file = "results/nuscenes_fulltrace_agentdriver_maxlen6000.pkl"
    # input_file = "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_agentdriver_rnbencoreit0_maxlen4096.jsonl"
    # output_file = "./results/nuscenes_agentdriver_rnbencoreit0_maxlen4096.pkl"
    # input_file = "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_agentdriver_rnbencoreit1_maxlen4096.jsonl"
    # output_file = "./results/nuscenes_agentdriver_rnbencoreit1_maxlen4096.pkl"
    # input_file = "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_agentdriver_rnbencoreit1_maxlen4096_ckpt13500.jsonl"
    # output_file = "./results/nuscenes_agentdriver_rnbencoreit1_maxlen4096_ckpt13500.pkl"
    # input_file = "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_agentdriver_rnbencoreit1_maxlen4096.jsonl"
    # output_file = "./results/nuscenes_agentdriver_rnbencoreit1_maxlen4096_v2.pkl"
    # input_file = "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_agentdriver_rnbencoreit1_maxlen4096_ckpt37000.jsonl"
    # output_file = "./results/nuscenes_agentdriver_rnbencoreit1_maxlen4096_ckpt37000.pkl"
    # input_file = "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_agentdriver_rnbencoreit1_maxlen4096_5ep_ckpt2000.jsonl"
    # output_file = "./results/nuscenes_agentdriver_rnbencoreit1_maxlen4096_5ep_ckpt2000.pkl"
    # input_file = "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_agentdriver_rnbencoreit1_maxlen4096_5ep_ckpt6000.jsonl"
    # output_file = "results/nuscenes_agentdriver_rnbencoreit1_maxlen4096_5ep_ckpt6000.pkl"
    # input_file = "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_fulltrace_vqadriver_maxlen8192.jsonl"
    # output_file = "results/nuscenes_fulltrace_vqadriver_maxlen8192.pkl"
    # input_file = "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_fulltrace_vqadriver_maxlen8192_ckp3000.jsonl"
    # output_file = "results/nuscenes_fulltrace_vqadriver_maxlen8192_ckp3000.pkl"
    # input_file = "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_fulltrace_vqadriver_maxlen8192_ckp1500.jsonl"
    # output_file = "results/nuscenes_fulltrace_vqadriver_maxlen8192_ckp1500.pkl"

    # Argparse to get input and output file paths
    parser = argparse.ArgumentParser(description="Parse JsonL file to trajectories pickle.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JsonL file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output pickle file")
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file

    # Process the file
    trajectories = parse_jsonl_to_trajectories(input_file, output_file)
    
    print(f"Successfully processed {len(trajectories)} trajectories to {output_file}")
