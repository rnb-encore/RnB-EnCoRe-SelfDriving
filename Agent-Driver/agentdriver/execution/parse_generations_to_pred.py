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
    # This pattern captures various list/tuple formats
    patterns = [
        r'(\[[\s\S]*?\])',  # Matches [...] including nested
        r'(\([\s\S]*?\))',  # Matches (...) including nested
    ]
    
    for pattern in patterns:
        match = re.search(pattern, traj_text)
        if match:
            traj_str = match.group(1)
            
            try:
                # Try ast.literal_eval first (handles tuples and lists)
                traj = ast.literal_eval(traj_str)
                
                # Convert to numpy array
                traj_array = np.array(traj)
                
                # Ensure it's 2D with shape (n, 2)
                if traj_array.ndim == 2 and traj_array.shape[1] == 2:
                    return traj_array
                    
            except (SyntaxError, ValueError) as e:
                print(f"ast.literal_eval failed: {e}")
                # If ast.literal_eval fails, try the parse_trace approach
                traj_points = parse_trace_fallback(traj_str)
                if traj_points:
                    return np.array(traj_points)
    raise ValueError("Could not parse trajectory from generated text")
    

def parse_trace_fallback(text: str):
    """
    Fallback parser for cases where ast.literal_eval doesn't work.
    Extracts pairs of numbers from the text.
    """
    # Extract all numbers
    numbers = re.findall(r'-?\d+\.?\d*', text)
    
    # Ensure we have an even number of values
    if len(numbers) >= 2 and len(numbers) % 2 == 0:
        points = []
        for i in range(0, len(numbers), 2):
            points.append([float(numbers[i]), float(numbers[i + 1])])
        return points
            

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
