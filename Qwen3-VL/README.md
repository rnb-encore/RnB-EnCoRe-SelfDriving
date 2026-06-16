# Qwen3-VL

<p align="center">
    <img src="https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-VL/qwen3vllogo.png" width="400"/>
<p>

<p align="center">
        💜 <a href="https://chat.qwenlm.ai/"><b>Qwen Chat</b></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/collections/Qwen/qwen3-vl-68d2a7c1b8a8afce4ebd2dbe">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/collections/Qwen3-VL-5c7a94c8cb144b">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://qwen.ai/blog?id=99f0335c4ad9ff6153e517418d48535ab6d8afef&from=research.latest-advancements-list">Blog</a>&nbsp&nbsp | &nbsp&nbsp📚 <a href="https://github.com/QwenLM/Qwen3-VL/tree/main/cookbooks">Cookbooks</a>
</p>

> **Note for this release.** This is a trimmed copy of the upstream
> [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) repository. We use it as the base
> vision-language model and its [`qwen-vl-finetune`](qwen-vl-finetune) stack for
> fine-tuning, vLLM inference, and R&B-EnCoRe sampling in our self-driving experiments — see the
> [top-level README](../README.md) for the end-to-end workflow. The extensive upstream usage
> guide (news, benchmarks, cookbooks, per-modality inference snippets, web UI) has been
> removed; refer to the [original repository](https://github.com/QwenLM/Qwen3-VL) for the
> full documentation.

## Introduction

Qwen3-VL is the latest vision-language model in the Qwen series, delivering upgrades across
text understanding & generation, visual perception & reasoning, extended context length,
spatial/video comprehension, and agent interaction. It is available in Dense and MoE
architectures (Instruct and Thinking editions) scaling from edge to cloud.

## Model Download

Pretrained checkpoints are hosted on Hugging Face and ModelScope:

- 🤗 [Hugging Face collection](https://huggingface.co/collections/Qwen/qwen3-vl-68d2a7c1b8a8afce4ebd2dbe)
- 🤖 [ModelScope collection](https://modelscope.cn/collections/Qwen3-VL-5c7a94c8cb144b)

Sizes range from 2B to 235B-A22B (e.g. `Qwen/Qwen3-VL-4B-Instruct`,
`Qwen/Qwen3-VL-30B-A3B-Instruct`), each in Instruct and Thinking variants, plus FP8 versions.

## Setup

### Transformers (training / quick inference)

```bash
# The Qwen3-VL model requires transformers >= 4.57.0
pip install "transformers>=4.57.0"
pip install qwen-vl-utils==0.0.14
```

For fine-tuning dependencies (DeepSpeed, etc.), see [`qwen-vl-finetune`](qwen-vl-finetune).

### vLLM (deployment / batch inference / R&B-EnCoRe sampling)

We recommend vLLM for fast Qwen3-VL inference. You need `vllm>=0.11.0` to enable Qwen3-VL
support.

```bash
pip install accelerate
pip install qwen-vl-utils==0.0.14
# Install the latest version of vLLM 'vllm>=0.11.0'
uv pip install -U vllm
```

Please check the [vLLM official documentation](https://docs.vllm.ai/en/latest/serving/multimodal_inputs.html)
for online serving and offline inference details.

### Docker

Pre-built images with the environment ready are available at
[qwenllm/qwenvl](https://hub.docker.com/r/qwenllm/qwenvl):

```bash
docker run --gpus all --ipc=host --network=host --rm --name qwen3vl -it qwenllm/qwenvl:qwen3vl-cu128 bash
```

## Citation

If you find Qwen3-VL useful in your research, please consider citing:

```BibTeX
@article{Qwen2.5-VL,
  title={Qwen2.5-VL Technical Report},
  author={Bai, Shuai and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Song, Sibo and Dang, Kai and Wang, Peng and Wang, Shijie and Tang, Jun and Zhong, Humen and Zhu, Yuanzhi and Yang, Mingkun and Li, Zhaohai and Wan, Jianqiang and Wang, Pengfei and Ding, Wei and Fu, Zheren and Xu, Yiheng and Ye, Jiabo and Zhang, Xi and Xie, Tianbao and Cheng, Zesen and Zhang, Hang and Yang, Zhibo and Xu, Haiyang and Lin, Junyang},
  journal={arXiv preprint arXiv:2502.13923},
  year={2025}
}

@article{Qwen2-VL,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng and Bai, Shuai and Tan, Sinan and Wang, Shijie and Fan, Zhihao and Bai, Jinze and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Fan, Yang and Dang, Kai and Du, Mengfei and Ren, Xuancheng and Men, Rui and Liu, Dayiheng and Zhou, Chang and Zhou, Jingren and Lin, Junyang},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}
```
