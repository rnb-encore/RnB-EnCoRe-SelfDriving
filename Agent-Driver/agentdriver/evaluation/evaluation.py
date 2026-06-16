# Evaluation of planning
# Written by Junjie Ye

import torch
from torch import Tensor
from tqdm import tqdm
import pickle
import json
from pathlib import Path
import os
import argparse

def planning_evaluation(pred_trajs_dict, config):
    future_second = 3
    ts = future_second * 2
    device = torch.device('cpu')

    if config.metric=="uniad":
        from agentdriver.evaluation.metric_uniad import PlanningMetric
        with open(Path(os.path.join(config.gt_folder, 'uniad_gt_seg.pkl')),'rb') as f:
            gt_occ_map = pickle.load(f)
        for token in gt_occ_map.keys():
            if not isinstance(gt_occ_map[token], torch.Tensor):
                gt_occ_map[token] = torch.tensor(gt_occ_map[token])
    elif config.metric=="stp3":
        from agentdriver.evaluation.metric_stp3 import PlanningMetric
        with open(Path(os.path.join(config.gt_folder, 'stp3_gt_seg.pkl')),'rb') as f:
            gt_occ_map = pickle.load(f)
        for token in gt_occ_map.keys():
            if not isinstance(gt_occ_map[token], torch.Tensor):
                gt_occ_map[token] = torch.tensor(gt_occ_map[token])
            gt_occ_map[token] = torch.flip(gt_occ_map[token], [-1])
            gt_occ_map[token] = torch.flip(gt_occ_map[token], [-2])
    else:
        raise ValueError(f"Invalid metric: {config.metric}")
    
    metric_planning_val = PlanningMetric(ts).to(device)     

    with open(Path(os.path.join(config.gt_folder, 'gt_traj.pkl')),'rb') as f:
        gt_trajs_dict = pickle.load(f)

    with open(Path(os.path.join(config.gt_folder, 'gt_traj_mask.pkl')),'rb') as f:
        gt_trajs_mask_dict = pickle.load(f)

    invalid_tokens = []
    # set default trajectory to zeros
    default_traj = torch.zeros((6, 2))
    # set default trajectory to average trajectory
    # default_traj = torch.tensor([[ 0.02084402,  2.50936506],
    #                              [ 0.04402258,  5.01318497],
    #                              [ 0.06945399,  7.50744185],
    #                              [ 0.09701727,  9.9884894 ],
    #                              [ 0.12654053, 12.45306781],
    #                              [ 0.15814563, 14.89885027]])
    for index, token in enumerate(tqdm(gt_trajs_dict.keys())):
        gt_trajectory =  torch.tensor(gt_trajs_dict[token])
        gt_trajectory = gt_trajectory.to(device)

        gt_traj_mask = torch.tensor(gt_trajs_mask_dict[token])
        gt_traj_mask = gt_traj_mask.to(device)

        if token not in pred_trajs_dict:
            # print(f"Token {token} not in prediction results, skipping...")
            # continue
            print(f"Token {token} not in prediction results, using zeros trajectory...")
            # pred_trajs_dict[token] = torch.zeros_like(gt_trajectory).cpu().numpy()
            pred_trajs_dict[token] = default_traj

            # save the invalid token
            invalid_tokens.append(token)
        try:
            output_trajs =  torch.tensor(pred_trajs_dict[token], dtype=torch.float32)
        except Exception as e:
            print(f"Error converting predicted trajectory to tensor for token {token}: {e}")
            print(f"*********** Using zeros trajectory for trajectory {pred_trajs_dict[token]}...")
            output_trajs = default_traj
            invalid_tokens.append(token)
        if output_trajs.shape[0] != gt_traj_mask.shape[1]:
            print(f"Token {token} has different trajectory length between prediction and ground truth, appending missing items to the end...")
            diff_len = gt_traj_mask.shape[1] - output_trajs.shape[0]
            if diff_len < 0:
                print(f"Token {token} has longer predicted trajectory than ground truth, truncating...")
                output_trajs = output_trajs[:gt_traj_mask.shape[1], :]
            else:
                pad_trajs = torch.zeros((diff_len, output_trajs.shape[1]), dtype=output_trajs.dtype, device=output_trajs.device)
                output_trajs = torch.cat((output_trajs, torch.tensor(pad_trajs)), dim=0)
        output_trajs = output_trajs.reshape(gt_traj_mask.shape)
        output_trajs = output_trajs.to(device)

        occupancy: Tensor = gt_occ_map[token]
        occupancy = occupancy.to(device)

        if output_trajs.shape[1] % 2: # in case the current timestep is inculded
            output_trajs = output_trajs[:, 1:]

        if occupancy.shape[1] % 2: # in case the current timestep is inculded
            occupancy = occupancy[:, 1:]
        
        if gt_trajectory.shape[1] % 2: # in case the current timestep is inculded
            gt_trajectory = gt_trajectory[:, 1:]

        if gt_traj_mask.shape[1] % 2:  # in case the current timestep is inculded
            gt_traj_mask = gt_traj_mask[:, 1:]
        
        metric_planning_val(output_trajs[:, :ts], gt_trajectory[:, :ts], occupancy[:, :ts], token, gt_traj_mask)
          
    results = {}
    scores = metric_planning_val.compute()
    for i in range(future_second):
        for key, value in scores.items():
            results['plan_'+key+'_{}s'.format(i+1)]=value[:(i+1)*2].mean()
    
    print("#################### Invalid Tokens ####################")
    print(invalid_tokens)
    print("Total invalid tokens: ", len(invalid_tokens))
    print("########################################################")
    print("#################### Evaluation Results ####################")
    headers = ["Method", "L2 (m)", "Collision (%)"]
    sub_headers = ["1s", "2s", "3s", "Avg."]
    if config.metric=="uniad":
        method = (config.method, "{:.2f}".format(scores["L2"][1]), "{:.2f}".format(scores["L2"][3]), "{:.2f}".format(scores["L2"][5]),\
                "{:.2f}".format((scores["L2"][1]+ scores["L2"][3]+ scores["L2"][5]) / 3.), \
                "{:.2f}".format(scores["obj_box_col"][1]*100), \
                "{:.2f}".format(scores["obj_box_col"][3]*100), \
                "{:.2f}".format(scores["obj_box_col"][5]*100), \
                "{:.2f}".format(100*(scores["obj_box_col"][1]+ scores["obj_box_col"][3]+ scores["obj_box_col"][5]) / 3.))
        print("{:<15} {:<20} {:<20}".format(*headers))
        print("{:<15},{:<5},{:<5},{:<5},{:<5},{:<5},{:<5},{:<5},{:<5}".format("", *sub_headers, *sub_headers))
        print("{:<15},{:<5},{:<5},{:<5},{:<5},{:<5},{:<5},{:<5},{:<5}".format(*method))

    elif config.metric=="stp3":
        method = (config.method, "{:.2f}".format(results["plan_L2_1s"]), "{:.2f}".format(results["plan_L2_2s"]), "{:.2f}".format(results["plan_L2_3s"]), \
                    "{:.2f}".format((results["plan_L2_1s"]+results["plan_L2_2s"]+results["plan_L2_3s"])/3.), \
                    "{:.2f}".format(results["plan_obj_box_col_1s"]*100), "{:.2f}".format(results["plan_obj_box_col_2s"]*100), "{:.2f}".format(results["plan_obj_box_col_3s"]*100), \
                        "{:.2f}".format(((results["plan_obj_box_col_1s"] + results["plan_obj_box_col_2s"] + results["plan_obj_box_col_3s"])/3)*100))
        print("{:<15} {:<20} {:<20}".format(*headers))
        print("{:<15},{:<5},{:<5},{:<5},{:<5},{:<5},{:<5},{:<5},{:<5}".format("", *sub_headers, *sub_headers))
        print("{:<15},{:<5},{:<5},{:<5},{:<5},{:<5},{:<5},{:<5},{:<5}".format(*method))

def load_pred_trajs_from_file(path):
    with open(path, "rb") as f:
        pred_trajs_dict = pickle.load(f)
    return pred_trajs_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation of planning')
    parser.add_argument('--method', type=str, help='name of the method being evaluated, used for table print', default='Agent-Driver')
    parser.add_argument('--result_file', type=str, help='path to the result file', default='temp_results/refined_trajs_dict_0.0_5.0_1.265_7.89.pkl')
    parser.add_argument('--metric', type=str, default='uniad', help='metric to evaluate, either uniad or stp3')
    parser.add_argument('--gt_folder', type=str, default='data/metrics')
    config = parser.parse_args()

    result_file = Path(config.result_file)
    pred_trajs_dict = load_pred_trajs_from_file(result_file)
    planning_evaluation(pred_trajs_dict, config)