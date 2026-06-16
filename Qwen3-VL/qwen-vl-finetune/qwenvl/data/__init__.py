import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

NUSCENES_FULL_AGENTDRIVER_COT = {
    "annotation_path": "/path/to/Agent-Driver/data/finetune/nuscenes_fulltrace_agentdriver_train.json",
    "data_path": "/path/to/nuscenes",
}

NUSCENES_FULL_VQADRIVER_COT = {
    "annotation_path": "/path/to/Agent-Driver/data/finetune/nuscenes_fulltrace_vqadriver_train.jsonl",
    "data_path": "/path/to/nuscenes",
}

NUSCENES_SUBSET_VQADRIVER_COT = {
    "annotation_path": "/path/to/Agent-Driver/data/finetune/nuscenes_noexp_nonotableobj_vqadriver_train.jsonl",
    "data_path": "/path/to/nuscenes",
}

NUSCENES_TRAGONLY = {
    "annotation_path": "/path/to/Agent-Driver/data/finetune/nuscenes_trajonly_train.json",
    "data_path": "/path/to/nuscenes",
} 

NUSCENES_VQA_TRAJONLY = {
    "annotation_path": "/path/to/Agent-Driver/data/finetune/nuscenes_trajonly_vqadriver_train.jsonl",
    "data_path": "/path/to/nuscenes",
}

NUSCENES_AGENTDRIVER_RNBENCOREIT0 = {
    "annotation_path": "/path/to/Agent-Driver/data/finetune/nuscenes_agentdriver_rnbencoreit0_train.jsonl",
    "data_path": "/path/to/nuscenes",
}

NUSCENES_VQADRIVER_RNBENCOREIT0 = {
    "annotation_path": "/path/to/Agent-Driver/data/finetune/nuscenes_vqadriver_rnbencoreit0_train.jsonl",
    "data_path": "/path/to/nuscenes",
}

NUSCENES_VQADRIVER_RNBENCOREIT0_DROPOUT05 = {
    "annotation_path": "/path/to/Agent-Driver/data/finetune/nuscenes_vqadriver_rnbencoreit0_train_dropout0.5.jsonl",
    "data_path": "/path/to/nuscenes",
}

NUSCENES_VQADRIVER_RNBENCOREIT0_DROPOUT03 = {
    "annotation_path": "/path/to/Agent-Driver/data/finetune/nuscenes_vqadriver_rnbencoreit0_train_dropout0.3.jsonl",
    "data_path": "/path/to/nuscenes",
}

NUSCENES_AGENTDRIVER_RNBENCOREIT1 = {
    "annotation_path": "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_rnbencoreit0_generation_resampling.jsonl",
    "data_path": "/path/to/nuscenes",
}

NUSCENES_VQADRIVER_RNBENCOREIT1_V1 = {
    "annotation_path": "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_vqadriver_rnbencore/rnbencoreit0_generation.jsonl",
    "data_path": "/path/to/nuscenes",
}

NUSCENES_VQADRIVER_RNBENCOREIT1_V2 = {
    "annotation_path": "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_vqadriver_rnbencore_rerun/rnbencoreit0_generation.jsonl",
    "data_path": "/path/to/nuscenes",
}

NUSCENES_VQADRIVER_RNBENCOREIT1_03 = {
    "annotation_path": "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_vqadriver_rnbencore_do03_prioronly/rnbencoreit0_generation.jsonl",
    "data_path": "/path/to/nuscenes",
}

NUSCENES_VQADRIVER_RNBENCOREIT1_05 = {
    "annotation_path": "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_vqadriver_rnbencore_do05_prioronly/rnbencoreit0_generation.jsonl",
    "data_path": "/path/to/nuscenes",
}

NUSCENES_RNBENCOREKGEN_4 = {
    "annotation_path": "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_vqadriver_rnbencore_do05_kgen4/rnbencoreit0_generation.jsonl",
    "data_path": "/path/to/nuscenes",
}
NUSCENES_RNBENCOREKGEN_12 = {
    "annotation_path": "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_vqadriver_rnbencore_do05_kgen12/rnbencoreit0_generation.jsonl",
    "data_path": "/path/to/nuscenes",
}
NUSCENES_RNBENCOREKGEN_16 = {
    "annotation_path": "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_vqadriver_rnbencore_do05_kgen16/rnbencoreit0_generation.jsonl",
    "data_path": "/path/to/nuscenes",
}
NUSCENES_RNBENCOREKGEN_32 = {
    "annotation_path": "/path/to/Qwen3-VL/qwen-vl-finetune/results/nuscenes_vqadriver_rnbencore_do05_kgen32/rnbencoreit0_generation.jsonl",
    "data_path": "/path/to/nuscenes",
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "nuscenes_fulltrace_agentdriver": NUSCENES_FULL_AGENTDRIVER_COT,
    "nuscenes_fulltrace_vqadriver": NUSCENES_FULL_VQADRIVER_COT,
    "nuscenes_subset_vqadriver": NUSCENES_SUBSET_VQADRIVER_COT,
    "nuscenes_vqadriver_rnbencoreit0": NUSCENES_VQADRIVER_RNBENCOREIT0,
    "nuscenes_vqadriver_rnbencoreit0_dropout05": NUSCENES_VQADRIVER_RNBENCOREIT0_DROPOUT05,
    "nuscenes_vqadriver_rnbencoreit0_dropout03": NUSCENES_VQADRIVER_RNBENCOREIT0_DROPOUT03,
    "nuscenes_vqadriver_rnbencoreit1_v1": NUSCENES_VQADRIVER_RNBENCOREIT1_V1,
    "nuscenes_vqadriver_rnbencoreit1_v2": NUSCENES_VQADRIVER_RNBENCOREIT1_V2,
    "nuscenes_vqadriver_rnbencoreit1_do03": NUSCENES_VQADRIVER_RNBENCOREIT1_03,
    "nuscenes_vqadriver_rnbencoreit1_do05": NUSCENES_VQADRIVER_RNBENCOREIT1_05,
    "nuscenes_trajonly": NUSCENES_TRAGONLY,
    "nuscenes_vqa_trajonly": NUSCENES_VQA_TRAJONLY,
    "nuscenes_agentdriver_rnbencoreit0": NUSCENES_AGENTDRIVER_RNBENCOREIT0,
    "nuscenes_agentdriver_rnbencoreit1": NUSCENES_AGENTDRIVER_RNBENCOREIT1,
    "nuscenes_rnbencorekgen_4": NUSCENES_RNBENCOREKGEN_4,
    "nuscenes_rnbencorekgen_12": NUSCENES_RNBENCOREKGEN_12,
    "nuscenes_rnbencorekgen_16": NUSCENES_RNBENCOREKGEN_16,
    "nuscenes_rnbencorekgen_32": NUSCENES_RNBENCOREKGEN_32,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
