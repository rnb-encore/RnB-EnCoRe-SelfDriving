# R&B-EnCoRe for Self-Driving Trajectory Planning

[![arXiv](https://img.shields.io/badge/arXiv-2602.08167-b31b1b.svg)](https://arxiv.org/abs/2602.08167) [![Conference](https://img.shields.io/badge/RSS-2026-blue)](https://roboticsconference.org/)


## Self-Refining and Bootstrapping from Action-Predictive Embodied Reasoning for Autonomous Vehicle Trajectory Planning on nuScenes using Qwen3-VL

This repository implements **R&B-EnCoRe**, a self-refining and bootstrapping pipeline that learns *which* reasoning actually helps a **Qwen3-VL** vision-language model produce better autonomous-vehicle trajectories on the [nuScenes](https://www.nuscenes.org/) benchmark. The model takes a front-camera image plus ego state and produces a 3-second planned trajectory (6 waypoints at 0.5 s intervals), optionally accompanied by intermediate reasoning (perception, common-sense, experience, and chain-of-thought).

The core idea: instead of hand-picking what a model should "reason about" before acting, we generate many candidate reasoning traces, score how much each one improves the likelihood of the correct trajectory (via an ELBO importance weight against a prior), resample toward the useful reasoning, and bootstrap a stronger model on the refined data.

This release combines two upstream codebases:

| Subdirectory   | Role | Upstream |
|----------------|------|----------|
| [`Qwen3-VL/`](Qwen3-VL/)       | VLM fine-tuning, vLLM inference, and R&B-EnCoRe sampling | [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) |
| [`Agent-Driver/`](Agent-Driver/) | Driving data generation, trajectory parsing, and nuScenes (UniAD / STP3) evaluation | [Agent-Driver](https://github.com/USC-GVL/Agent-Driver) |

The codebase covers:
1.  **Data Generation** (`Agent-Driver`): Building conversation-format fine-tuning data from Agent-Driver planner inputs — single-trace SFT data and R&B-EnCoRe iteration-0 data (prior + posterior reasoning traces).
2.  **Supervised Fine-Tuning** (`Qwen3-VL`): Training Qwen3-VL as an end-to-end trajectory planner — a trajectory-only baseline and an iteration-0 model that produces latent reasoning + trajectory.
3.  **Refining (R&B-EnCoRe)**: Sampling posterior reasoning traces from the iteration-0 model, scoring them with ELBO importance weights, and resampling toward the reasoning that helps.
4.  **Bootstrapping**: Retraining (iteration 1) on the refined reasoning to produce the final model.
5.  **Inference & Evaluation**: Running vLLM inference, parsing trajectories, and computing nuScenes planning metrics (UniAD / STP3 L2 + collision).

> **Note on paths.** Absolute cluster paths have been replaced with placeholders such as
> `/path/to/Agent-Driver`, `/path/to/Qwen3-VL`, `/path/to/nuscenes`, and
> `/path/to/conda/envs/<env>`. Update these to your environment before running. The training
> SLURM scripts (`qwen-vl-finetune/scripts/*.sh`) and `qwenvl/data/__init__.py` are the main
> places that reference data and checkpoint locations.

---

## Setup

### 1. Environments

The scripts reference several conda environments; create them to match your cluster:

- **Training** (`qwen3-vl-new`): PyTorch, `transformers`, DeepSpeed, and the
  Qwen3-VL fine-tuning dependencies in [`Qwen3-VL/qwen-vl-finetune`](Qwen3-VL/qwen-vl-finetune).
- **Inference / R&B-EnCoRe sampling** (`qwen3_serve_new`): `vllm` plus the Qwen3-VL
  multimodal processor stack.
- **Evaluation** (`driveagent_new`): the Agent-Driver dependencies (`torch`,
  `nuscenes-devkit`, `numpy`) used by `agentdriver/evaluation`.

See the upstream READMEs ([`Qwen3-VL/README.md`](Qwen3-VL/README.md),
[`Agent-Driver/README.md`](Agent-Driver/README.md)) for the full dependency lists.

### 2. Data

- **nuScenes images.** Download the nuScenes `v1.0-trainval` set and point
  `data_path` / `image_data_path` (placeholder `/path/to/nuscenes`) at the dataroot.
- **Agent-Driver planner inputs & metric ground truth.** The `Agent-Driver/data/`
  directory (planner inputs, `data/metrics` ground-truth pickles) is **not** bundled in
  this release — see the [Agent-Driver instructions](Agent-Driver/README.md) to download
  it. Generated fine-tuning files are recreated by the data-generation scripts above.
  Evaluation expects `data/metrics/{uniad,stp3}_gt_seg.pkl`, `gt_traj.pkl`, and
  `gt_traj_mask.pkl`.

### 3. Model checkpoints

Fine-tuned checkpoints are released on the HuggingFace Hub:

| Checkpoint | Description | Link |
|------------|-------------|------|
| **No-reasoning (trajectory-only)** | Trajectory-prediction baseline trained with `nuscenes_trajonly.sh` — predicts waypoints directly, with no intermediate reasoning. | [stanfordasl/nuscenes-waypoints-model](https://huggingface.co/stanfordasl/nuscenes-waypoints-model) |
| **Full Agent Driver Reasoning** | Checkpoint trained with the full reasoning traces for self-driving, estimated by the Agent Driver work. | [stanfordasl/nuscenes-full-reasoning-waypoints](https://huggingface.co/stanfordasl/nuscenes-full-reasoning-waypoints) |
| **R&B-EnCoRe (iteration 1)** | Final checkpoint after the R&B-EnCoRe self-training loop (`nuscenes_rnbencore_it1.sh`), with the refined latent reasoning. | [stanfordasl/nuscenes-rnbencore-reasoning-waypoints](https://huggingface.co/stanfordasl/nuscenes-rnbencore-reasoning-waypoints) |

Point `CHECKPOINT_PATH` in `inference_evaluate_nuscenes.sh` at the HuggingFace
model ID (or a local download) to reproduce the reported metrics.

---

## Repository layout

```
Code-Release/
├── Qwen3-VL/
│   └── qwen-vl-finetune/
│       ├── qwenvl/
│       │   ├── train/train_qwen.py             # SFT entry point
│       │   ├── inference/inference_qwen.py     # vLLM batch inference (evaluation)
│       │   ├── inference/rnbencore_generation_qwen.py # R&B-EnCoRe posterior sampling + ELBO resampling
│       │   ├── inference/inference_utils.py    # shared data loading / vLLM input prep
│       │   └── data/__init__.py                # dataset registry (annotation/data paths)
│       └── scripts/                            # SLURM launch scripts (see below)
└── Agent-Driver/
    └── agentdriver/
        ├── execution/
        │   ├── gen_drive_vla_data.py           # build SFT data (single trace per sample)
        │   ├── gen_drive_vla_rnbencoreit0_data.py   # build R&B-EnCoRe it0 data (prior+posterior traces)
        │   └── parse_generations_to_pred.py    # parse generated text -> trajectory pickle
        └── evaluation/
            ├── evaluation.py                   # planning evaluation entry point
            ├── metric_uniad.py                 # UniAD-style L2 / collision metrics
            └── metric_stp3.py                  # STP3-style L2 / collision metrics
```

---

## Workflow & Usage

The pipeline runs end-to-end: generate data → fine-tune → sample & refine (R&B-EnCoRe) → bootstrap → evaluate.

```
data_samples_*.json                 (Agent-Driver planner inputs)
        │
        │  gen_drive_vla_data.py  /  gen_drive_vla_rnbencoreit0_data.py
        ▼
nuscenes_*_{train,val}.jsonl        (conversation-format SFT data)
        │
        │  train_qwen.py  (nuscenes_trajonly.sh / nuscenes_rnbencore_it0.sh)
        ▼
fine-tuned checkpoint
        │
        │  rnbencore_generation_qwen.py  (nuscenes_rnbencore_inference.sh)   ── R&B-EnCoRe loop
        ▼
rnbencoreit0_generation.jsonl  ──►  train_qwen.py  (nuscenes_rnbencore_it1.sh)
        │
        │  inference_qwen.py  ──►  parse_generations_to_pred.py  ──►  evaluation.py
        ▼  (inference_evaluate_nuscenes.sh)
L2 / collision metrics (UniAD or STP3)
```

### 1. Data generation (`Agent-Driver`)

Starting from Agent-Driver planner inputs (`data/finetune/data_samples_{train,val}.json`),
build conversation-format fine-tuning data:

```bash
cd Agent-Driver
# Single-trace SFT data
python agentdriver/execution/gen_drive_vla_data.py
# R&B-EnCoRe iteration-0 data (64 prior + 64 posterior traces per sample, with dropout)
python agentdriver/execution/gen_drive_vla_rnbencoreit0_data.py
```

Each script writes `.json` / `.jsonl` files under `data/finetune/`. Edit the `image_data_path`
and output names in the `__main__` block to match your nuScenes location. The resulting
datasets are registered in [`Qwen3-VL/qwen-vl-finetune/qwenvl/data/__init__.py`](Qwen3-VL/qwen-vl-finetune/qwenvl/data/__init__.py)
under names like `nuscenes_vqa_trajonly`, `nuscenes_vqadriver_rnbencoreit0`, etc.

### 2. Supervised fine-tuning (`Qwen3-VL`)

```bash
cd Qwen3-VL/qwen-vl-finetune
# Trajectory-only SFT baseline
sbatch scripts/nuscenes_trajonly.sh
# R&B-EnCoRe iteration 0 (train the model to produce latent reasoning + trajectory)
sbatch scripts/nuscenes_rnbencore_it0.sh
```

Each script selects a dataset by name (`--dataset_use`) from the registry and writes a
checkpoint under `checkpoints/`.

### 3. R&B-EnCoRe sampling (`Qwen3-VL`)

Use the iteration-0 checkpoint to sample reasoning traces for the training set. For each
example, `K_gen` posterior generations are drawn and resampled by their ELBO importance
weights, producing the iteration-1 training data:

```bash
sbatch scripts/nuscenes_rnbencore_inference.sh --kgen 8
```

### 4. R&B-EnCoRe iteration 1 (`Qwen3-VL`)

Fine-tune again on the resampled traces:

```bash
sbatch scripts/nuscenes_rnbencore_it1.sh
```

---

## Evaluation

The end-to-end evaluation script runs vLLM inference, parses trajectories, and computes
nuScenes planning metrics:

```bash
cd Qwen3-VL/qwen-vl-finetune
sbatch scripts/inference_evaluate_nuscenes.sh
```

Internally this:
1. runs `qwenvl/inference/inference_qwen.py` to generate predictions (`results/*.jsonl`);
2. parses them with `agentdriver/execution/parse_generations_to_pred.py` into a trajectory
   pickle (`results/*.pkl`);
3. evaluates with `agentdriver/evaluation/evaluation.py --metric uniad` (or `stp3`).

Set `CHECKPOINT_PATH` to a local checkpoint or a HuggingFace model ID and `EVAL_TAG` to a
short label for the output files.

---

## Citation

If you find this code useful in your research, please cite our paper:

```bibtex
@INPROCEEDINGS{GanaiLuoEtAl-2026RnB-EnCoRe,
    AUTHOR    = {Milan Ganai AND Katie Luo AND Jonas Frey AND Clark Barrett AND Marco Pavone},
    TITLE     = {{Self-Supervised Bootstrapping of Action-Predictive Embodied Reasoning}},
    BOOKTITLE = {Proceedings of Robotics: Science and Systems},
    YEAR      = {2026},
    ADDRESS   = {Sydney, Australia},
    MONTH     = {July}
}
```

## Acknowledgements & licenses

This release builds directly on two open-source projects, each retaining its original license:

- **Qwen3-VL** — see [`Qwen3-VL/LICENSE`](Qwen3-VL/LICENSE).
- **Agent-Driver** — see [`Agent-Driver/LICENSE`](Agent-Driver/LICENSE).
