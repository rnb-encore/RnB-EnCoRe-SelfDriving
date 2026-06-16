# Agent-Driver

Based on the arXiv pre-print [Agent-Driver](https://arxiv.org/abs/2311.10813)
[[Project Page](https://usc-gvl.github.io/Agent-Driver/)].

> **Note for this release.** In this project we use Agent-Driver only for its **nuScenes
> data** and **planning evaluation** (UniAD / STP3 metrics) — see
> [`agentdriver/execution`](agentdriver/execution) (data generation, trajectory parsing) and
> [`agentdriver/evaluation`](agentdriver/evaluation). The original LLM-agent pipeline
> (OpenAI GPT-based reasoning engine, cognitive memory, tool library) is not used here. See
> the [top-level README](../README.md) for the end-to-end workflow. The upstream OpenAI
> fine-tuning / inference instructions have been removed from this file; refer to the
> [original repository](https://github.com/USC-GVL/Agent-Driver) for them.

## Installation

a. Install the dependent libraries as follows:

```
pip install -r requirements.txt
```

## Data Preparation

a. We used pre-cached data from the nuScenes dataset. The data can be downloaded at
[Google Drive](https://drive.google.com/drive/folders/1BjCYr0xLTkLDN9DrloGYlerZQC1EiPie?usp=sharing).

b. You can put the downloaded data here:
```
Agent-Driver
├── data
│   ├── finetune
|   |   |── data_samples_train.json
|   |   |── data_samples_val.json
│   ├── memory
|   |   |── database.pkl
│   ├── metrics
|   |   |── gt_traj.pkl
|   |   |── gt_traj_mask.pkl
|   |   |── stp3_gt_seg.pkl
|   |   |── uniad_gt_seg.pkl
│   ├── train
|   |   |── [token].pkl
|   |   |── ...
│   ├── val
|   |   |── [token].pkl
|   |   |── ...
│   ├── split.json
├── agentdriver
├── scripts
```

The `data/metrics/{uniad,stp3}_gt_seg.pkl`, `gt_traj.pkl`, and `gt_traj_mask.pkl` files are
the ground truth required by the planning evaluation in
[`agentdriver/evaluation/evaluation.py`](agentdriver/evaluation/evaluation.py).

## Citation

If you find this project useful in your research, please consider citing:

```
@article{agentdriver,
  title={A Language Agent for Autonomous Driving},
  author={Mao, Jiageng and Ye, Junjie and Qian, Yuxi and Pavone, Marco and Wang, Yue},
  year={2023}
}
```
