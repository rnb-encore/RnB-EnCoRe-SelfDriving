import json
import numpy as np
import re

def parse_trajectory(planning_target_str):
    """
    Parse the planning_target string to extract trajectory points.
    
    Args:
        planning_target_str: String containing "Planned Trajectory:\n[(x1,y1), (x2,y2), ...]"
    
    Returns:
        numpy array of shape (6, 2) containing the trajectory points
    """
    # Extract the list portion from the string
    # Look for pattern between '[' and ']'
    match = re.search(r'\[(.*?)\]', planning_target_str)
    
    if not match:
        raise ValueError(f"Could not find trajectory list in: {planning_target_str}")
    
    # Get the matched string containing coordinates
    coords_str = match.group(1)
    
    # Parse individual coordinate pairs
    # Split by '), (' to separate coordinate pairs
    coord_pairs = re.findall(r'\(([^,]+),([^)]+)\)', coords_str)
    
    if len(coord_pairs) != 6:
        raise ValueError(f"Expected 6 coordinate pairs, got {len(coord_pairs)}")
    
    # Convert to numpy array
    trajectory = np.array([[float(x), float(y)] for x, y in coord_pairs])
    
    return trajectory

def compute_average_trajectory(json_file_path):
    """
    Compute the average trajectory from all samples in the JSON file.
    
    Args:
        json_file_path: Path to the JSON file containing the data
    
    Returns:
        numpy array of shape (6, 2) containing the averaged trajectory
    """
    # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # List to store all trajectories
    trajectories = []
    
    # Process each element
    for i, element in enumerate(data):
        try:
            if 'planning_target' in element:
                trajectory = parse_trajectory(element['planning_target'])
                trajectories.append(trajectory)
            else:
                print(f"Warning: Element {i} missing 'planning_target' field")
        except Exception as e:
            print(f"Error processing element {i}: {e}")
    
    if not trajectories:
        raise ValueError("No valid trajectories found in the data")
    
    # Stack all trajectories and compute the mean
    all_trajectories = np.stack(trajectories, axis=0)  # Shape: (n_samples, 6, 2)
    average_trajectory = np.mean(all_trajectories, axis=0)  # Shape: (6, 2)
    
    return average_trajectory

def main():
    # Path to your JSON file
    json_file_path = '/path/to/Agent-Driver/data/finetune/data_samples_train.json'
    
    try:
        # Compute the average trajectory
        avg_trajectory = compute_average_trajectory(json_file_path)
        
        print("Average Trajectory (6x2 array):")
        print(avg_trajectory)
        print(f"\nShape: {avg_trajectory.shape}")
        
        # Optional: Display statistics
        print("\nTrajectory Statistics:")
        print(f"Mean X displacement: {avg_trajectory[:, 0].mean():.3f}")
        print(f"Mean Y displacement: {avg_trajectory[:, 1].mean():.3f}")
        print(f"Total distance traveled: {np.sum(np.sqrt(np.sum(np.diff(avg_trajectory, axis=0)**2, axis=1))):.3f}")
        
        # Optional: Save to file
        np.save('average_trajectory.npy', avg_trajectory)
        print("\nAverage trajectory saved to 'average_trajectory.npy'")
        
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file '{json_file_path}'")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
