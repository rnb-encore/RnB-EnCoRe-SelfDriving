import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import json
import ast
import re
import textwrap
from PIL import Image, ImageDraw, ImageFont

from cam_render import CameraRender
from utils import AgentPredictionData
# from visual_tokens import tokens_for_viz, tokens_for_main, viz_scenes
from nuscenes.nuscenes import NuScenes
# from nuscenes.utils.data_classes import LidarPointCloud, Box
# from pyquaternion import Quaternion
import imageio

# FIG_X = 53.3333
FIG_X = 17.0
FIG_Y = 10.0


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
    raise ValueError("Could not parse trajectory from generated text")


def parse_front_objects(reasoning):
    """
    Parse front object detections and their future trajectories from generated_text.
    
    Returns:
        dict: Dictionary with object_id as key, mapped to dict containing:
              - 'object_type': str
              - 'position': tuple
              - 'size': tuple
              - 'future_waypoints': numpy array
    """
    front_objects = {}
    
    if 'Front object detected' not in reasoning:
        return front_objects
    
    lines = reasoning.split('\n')
    
    # Parse front object detections
    for line in lines:
        if 'Front object detected' in line:
            # Extract object type
            obj_type_match = re.search(r'object type:\s*(\w+)', line)
            # Extract object id
            obj_id_match = re.search(r'object id:\s*(\d+)', line)
            # Extract position
            position_match = re.search(r'position:\s*\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)', line)
            # Extract size
            size_match = re.search(r'size:\s*\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)', line)
            
            if obj_id_match:
                obj_id = int(obj_id_match.group(1))
                front_objects[obj_id] = {
                    'object_type': obj_type_match.group(1) if obj_type_match else None,
                    'position': (float(position_match.group(1)), float(position_match.group(2))) if position_match else None,
                    'size': (float(size_match.group(1)), float(size_match.group(2))) if size_match else None,
                    'future_waypoints': None
                }
                # print(f"Detected front object ID {obj_id} of type {front_objects[obj_id]['object_type']}")
                    
    
    # Parse future trajectories
    for i, line in enumerate(lines):
        if 'Future trajectories for specific objects:' in line:
            # Look at subsequent lines for trajectory data
            j = i + 1
            while j < len(lines) and lines[j].strip():
                traj_line = lines[j]
                
                # Extract object id from trajectory line
                obj_id_match = re.search(r'object id:\s*(\d+)', traj_line)
                # Extract waypoint coordinates
                waypoints_match = re.search(r'future waypoint coordinates.*?:\s*(\[.*?\])', traj_line)
                
                if obj_id_match and waypoints_match:
                    obj_id = int(obj_id_match.group(1))
                    try:
                        waypoints_list = ast.literal_eval(waypoints_match.group(1))
                        waypoints_array = np.array(waypoints_list)
                        
                        # Add to existing object if it exists
                        if obj_id in front_objects:
                            front_objects[obj_id]['future_waypoints'] = waypoints_array
                    except:
                        pass
                
                j += 1
                # Break if we hit an empty line or new section
                if j < len(lines) and not lines[j].strip():
                    break
                if j < len(lines) and lines[j].startswith('<'):
                    break
    
    return front_objects



def draw_model_with_text(sample_token, nusc, samples, plan_trajs_dict, gt_trajs_dict=None, visualization_tag='default', experiment_tag='default'):
    sdc_pred_color = np.array([180, 120, 200]) / 255.0
    # det_obj_color = np.array([51, 153,255]) / 255.0
    # tool_obj_color = np.array([255, 153, 51]) / 255.0
    # cot_obj_color = np.array([255, 51, 51]) / 255.0
    sdc_gt_color = np.array([6, 214, 160]) / 255.0
    

    cam_render = CameraRender(show_gt_boxes=False, figsize=(FIG_X, FIG_Y))

    planner_input = None
    for sample in samples:
        if sample['token'] == sample_token:
            planner_input = sample
            break
    
    if planner_input is None:
        return

    # reasoning = planner_input['reasoning']
    obj_heights = {
        'car': 1.5,
        'truck': 3.5,
        'bus': 3.5,
        'bicycle': 1.5,
        'motorcycle': 1.5,
        'pedestrian': 1.7,
        'cone': 0.5,
        'barrier': 1.0
    }
    reasoning = planner_input['generated_text']
    # print(reasoning)
    front_objects = parse_front_objects(reasoning)
    # print("front objects:", front_objects)

    # if len(notable_objects) != len(notable_coords):
        # return
    det_agent_list = []
    for obj_id, obj_info in front_objects.items():
        pred_center = [obj_info['position'][0], obj_info['position'][1], 0.0] if obj_info['position'] is not None else [0.0, 0.0, 0.0]
        obj_height = obj_heights[obj_info['object_type'].lower()] if obj_info['object_type'] in obj_heights else 2.0
        pred_dim = [obj_info['size'][0], obj_info['size'][1], obj_height] if obj_info['size'] is not None else [1.0, 1.0, 1.0]
        pred_traj = obj_info['future_waypoints']
        if obj_info['future_waypoints'] is None:
            pred_traj = np.zeros((6, 2))
        pred_traj = np.concatenate([pred_traj, np.zeros((pred_traj.shape[0], 1))], axis=-1)
        det_agent_list.append(
            AgentPredictionData(
                pred_score=1.0,
                pred_label=0,
                pred_center=pred_center,
                pred_dim=pred_dim,
                pred_yaw=0.0,
                pred_vel=0.0,
                pred_traj=pred_traj,
                is_sdc = False
            )
        )

    cam_render.reset_canvas(dx=1, dy=1, tight_layout=True)
    cam_render.render_image_data(sample_token, nusc)

    # det_obj_color = np.array([51, 153,255]) / 255.0

    cam_render.render_pred_track_bbox_2D(
        det_agent_list, sample_token, nusc, box_color=sdc_pred_color)
    # cam_render.render_pred_traj(
    #     det_agent_list, sample_token, nusc, sdc_color=sdc_pred_color, render_sdc=False, box_color=det_obj_color, points_per_step=1)
    if gt_trajs_dict is not None and sample_token in gt_trajs_dict:
        gt_traj = gt_trajs_dict[sample_token][0]
        # filter out values where both are 0s in the -1 dim
        gt_traj = gt_traj[~((gt_traj[:,0]==0) & (gt_traj[:,1]==0))]
        gt_traj = np.concatenate([gt_traj, np.ones((gt_traj.shape[0],1))], axis=-1)
        # gt_traj = np.array(gt_traj)[1:]

        gt_agent_list = [
            AgentPredictionData(
            pred_score=1.0,
            pred_label=0,
            pred_center=[0, 0, 0],
            pred_dim=[4.5, 2.0, 2.0],
            pred_yaw=0,
            pred_vel=0,
            pred_traj=gt_traj,
            is_sdc = True
            )
        ]

        cam_render.render_pred_traj(
            gt_agent_list, sample_token, nusc, sdc_color=sdc_gt_color, render_sdc=True, points_per_step=5)
    
    # if sample_token not in plan_trajs_dict:
    #     return
    # plan_traj = plan_trajs_dict[sample_token]
    # don't rely on given plan traj dict, extract from generated text
    import pdb; pdb.set_trace()
    try:
        plan_traj = extract_trajectory(planner_input['generated_text'])
    except ValueError:
        print("Could not extract trajectory for token:", sample_token)
        print(planner_input['generated_text'])
        plan_traj = plan_trajs_dict[sample_token]
    plan_traj = np.concatenate([plan_traj, np.ones((plan_traj.shape[0],1))], axis=-1)

    pred_agent_list = [
        AgentPredictionData(
            pred_score=1.0,
            pred_label=0,
            pred_center=[0, 0, 0],
            pred_dim=[4.5, 2.0, 2.0],
            pred_yaw=0,
            pred_vel=0,
            pred_traj=plan_traj,
            is_sdc = True
        )
    ]
    cam_render.render_pred_traj(
        pred_agent_list, sample_token, nusc, sdc_color=sdc_pred_color, render_sdc=True, points_per_step=5)
    
    save_dir = Path(f"experiments/{visualization_tag}/{experiment_tag}")
    # save_dir = Path(f"experiments/{visualization_tag}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / Path(sample_token + '.jpg')
    cam_render.save_fig(save_path)
    
    # import pdb; pdb.set_trace()
    reasoning = planner_input['generated_text']
    reasoning = reasoning.split("\n")
    # text = ""
    # for line in reasoning:
    #     text += line + "\n"
    # cat_text(save_path, text)
    text = ""
    for line in reasoning:
        if len(line) > 55:
            # Wrap line: break after 50+ chars at last whitespace before 55
            wrapped = textwrap.fill(line, width=55, break_long_words=False, break_on_hyphens=False)
            text += wrapped + "\n"
        else:
            text += line + "\n"
    if len(reasoning) > 1:
        # cat_text(save_path, text)

        # save text in a txt file for easy retrieval
        save_text_path = save_dir / Path(sample_token + '.txt')
        with open(save_text_path, "w") as f:
            f.write(planner_input['generated_text'])
    image = Image.open(save_path)
    return image

def create_text_image(text, img_size=(5333, 900)):
    image = Image.new('RGB', img_size, color='white')
    draw = ImageDraw.Draw(image)
    # font = ImageFont.truetype(font='/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf', size=50)
    # font = ImageFont.load_default()
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSansMono.ttf", size=50)
    x = 80
    y = 0
    draw.text((x, y), text, font=font, fill='black')
    
    return image

def cat_text(save_path, text):
    text_height = 1800
    image1 = Image.open(save_path)
    image2 = create_text_image(text, img_size=(int(FIG_X*100), text_height))

    new_image = Image.new('RGB', (int(FIG_X*100), int(FIG_Y*100 + text_height)), (255, 255, 255))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (0, int(FIG_Y * 100)))

    new_image.save(save_path)
    return

if __name__ == "__main__":
    nusc = NuScenes(version="v1.0-trainval", dataroot="/path/to/nuscenes", verbose=True)
    # samples = json.load(open('data/finetune/data_samples_val.json', 'r'))
    # plan_trajs_dict = pickle.load(open('pred_trajs_dict.pkl', 'rb')) # your pred traj dict here
    
    visualization_tag = "visualization_8e-5"
    sample_json_paths = [
        '/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_vqa_trajonly_maxlen2048_toks2048.jsonl',
        '/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_fulltrace_vqadriver_maxlen6144_30ep_lr5e-5_toks6144.jsonl',
        # '/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_rnbencorekgen_16_maxlen6144_30ep_lr6e-5_toks6144.jsonl',
        "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_vqadriver_rnbencoreit1_do05_maxlen6144_30ep_lr8e-5_toks6144.jsonl",
        "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_vqadriver_rnbencoreit0_dropout05_maxlen6144_2ep_lr5e-5_checkpoint-6000_toks6144.jsonl",
    ]
    planning_pickle_paths = [
        '/path/to/Agent-Driver/results/nuscenes_vqa_trajonly_maxlen2048.pkl',
        '/path/to/Agent-Driver/results/nuscenes_fulltrace_vqadriver_maxlen6144_30ep_lr5e-5.pkl',
        # '/path/to/Agent-Driver/results/nuscenes_rnbencorekgen_16_maxlen6144_30ep_lr6e-5.pkl',
        '/path/to/Agent-Driver/results/nuscenes_vqadriver_rnbencoreit1_do05_maxlen6144_30ep_lr8e-5.pkl',
        '/path/to/Agent-Driver/results/nuscenes_vqadriver_rnbencoreit0_dropout05_maxlen6144_2ep_lr5e-5_checkpoint-6000.pkl',
    ]
    experiment_tags = [
        'nuscenes_vqa_trajonly_maxlen2048',
        'nuscenes_fulltrace_vqadriver_maxlen6144_30ep_lr5e-5',
        # 'nuscenes_rnbencorekgen_16_maxlen6144_30ep_lr6e-5',
        'nuscenes_vqadriver_rnbencoreit1_do05_maxlen6144_30ep_lr8e-5',
        'nuscenes_vqadriver_rnbencoreit0_dropout05_maxlen6144_2ep_lr5e-5_checkpoint-6000',
    ]

    # visualization_tag = "kgen_comparison"
    # sample_json_paths = [
    #     "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_rnbencorekgen_4_maxlen6144_30ep_lr6e-5_toks6144.jsonl",
    #     '/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_rnbencorekgen_16_maxlen6144_30ep_lr6e-5_toks6144.jsonl',
    #     "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_rnbencorekgen_32_maxlen6144_30ep_lr6e-5_toks6144.jsonl",
    # ]
    # planning_pickle_paths = [
    #     "/path/to/Agent-Driver/results/nuscenes_rnbencorekgen_4_maxlen6144_30ep_lr6e-5.pkl",
    #     '/path/to/Agent-Driver/results/nuscenes_rnbencorekgen_16_maxlen6144_30ep_lr6e-5.pkl',
    #     "/path/to/Agent-Driver/results/nuscenes_rnbencorekgen_32_maxlen6144_30ep_lr6e-5.pkl",
    # ]
    # experiment_tags = [
    #     "nuscenes_rnbencorekgen_4_maxlen6144_30ep_lr6e-5",
    #     'nuscenes_rnbencorekgen_16_maxlen6144_30ep_lr6e-5',
    #     "nuscenes_rnbencorekgen_32_maxlen6144_30ep_lr6e-5",
    # ]


    output_samples = []
    for sample_json_path in sample_json_paths:
        with open(sample_json_path, 'r') as f:
            samples = [json.loads(line) for line in f]
        output_samples.append(samples)
    
    all_plan_trajs_dicts = []
    for planning_pickle_path in planning_pickle_paths:
        plan_trajs_dict = pickle.load(open(planning_pickle_path, 'rb'))
        all_plan_trajs_dicts.append(plan_trajs_dict)

    gt_trajs_dict = pickle.load(open('data/metrics/gt_traj.pkl', 'rb'))
    sample_lde_diff = {}
    TOKENS_TO_AVOID = ["d763ff87e53e4b4f967ee4dc4193dda9"]
    # import pdb; pdb.set_trace()
    for sample_token in gt_trajs_dict.keys():
        gt_traj = gt_trajs_dict[sample_token][0]
        # filter out values where both are 0s in the -1 dim
        gt_traj = gt_traj[~((gt_traj[:,0]==0) & (gt_traj[:,1]==0))]
        if gt_traj.shape[0] < 2 or sample_token not in all_plan_trajs_dicts[1] or sample_token not in all_plan_trajs_dicts[2]:
            continue
        gt_last_timestep = gt_traj[-1, :]
        if np.linalg.norm(gt_last_timestep - gt_traj[0, :]) < 0.1:
            # skip ones that are stopped
            TOKENS_TO_AVOID.append(sample_token)

        # our method is the last one in the list
        ours_traj = all_plan_trajs_dicts[2][sample_token]
        ours_last_timestep = ours_traj[-1, :]

        # full reasoning
        full_traj = all_plan_trajs_dicts[1][sample_token]
        full_last_timestep = full_traj[-1, :]

        difference = np.linalg.norm(gt_last_timestep - full_last_timestep) - np.linalg.norm(gt_last_timestep - ours_last_timestep)
        sample_lde_diff[sample_token] = difference
    # import pdb; pdb.set_trace()
    # tokens = pd.Series(sample_lde_diff).nlargest(90).index.tolist()
    tokens = ["56a8812e2fae4b13960c853cebe58374"]
    # tokens = [sample['token'] for sample in samples]
    for sample_token in tokens:
        if sample_token in TOKENS_TO_AVOID:
            continue
        sample_images = []
        # gt_img = draw_ground_truth(sample_token, nusc, gt_trajs_dict)
        # sample_images.append(gt_img)
        for output_sample, plan_trajs_dict, experiment_tag in zip(output_samples, all_plan_trajs_dicts, experiment_tags):
            if "rnbencoreit0" not in experiment_tag:
                continue  # only visualize it0
            # print("Visualizing sample token:", sample_token, "for experiment:", experiment_tag)
            model_img = draw_model_with_text(sample_token, nusc, output_sample, plan_trajs_dict, gt_trajs_dict=gt_trajs_dict, visualization_tag=visualization_tag, experiment_tag=experiment_tag)
            sample_images.append(model_img)

        # # Combine images horizontally
        # widths = [img.width for img in sample_images]
        # heights = [img.height for img in sample_images]

        # total_width = sum(widths)
        # max_height = max(heights)

        # # Create new image with combined width
        # new_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))

        # # Paste images horizontally
        # x_offset = 0
        # for img in sample_images:
        #     new_image.paste(img, (x_offset, 0))
        #     x_offset += img.width

        # save_path = Path(f"experiments/{visualization_tag}/combined_{sample_token}.jpg")
        # new_image.save(save_path)
