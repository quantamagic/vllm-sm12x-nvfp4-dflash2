# vLLM All-NVFP4 DFlash2 — AutoDL 镜像说明

## 镜像构建

### 基本环境

- Ubuntu 22.04 / CUDA 13.0 / cuDNN 9.2
- Python 3.12
- 显卡：RTX 5090 (SM120) 32GB

### 框架及版本

| 组件 | 版本 |
|---|---|
| PyTorch | 2.13.0+cu130 |
| vLLM | 0.27.2.dev0（v0.27.1 + SM120/NVFP4 补丁）|
| Triton | 3.7.1 |
| FlashInfer | 0.6.16.post3 |
| transformers / safetensors | 随 vLLM 依赖安装 |

## 构建过程

### 1. 代码 Clone

```bash
cd /root
git clone https://github.com/quantamagic/vllm-sm12x-nvfp4-dflash2.git
cd vllm-sm12x-nvfp4-dflash2
```

### 2. 依赖安装

```bash
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
pip install flashinfer==0.6.16.post3 -i https://flashinfer.ai/whl/cu130
pip install triton==3.7.1
pip install vllm
```

模型权重（HF 下载，约 22 GB）：

```bash
huggingface-cli download gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 \
  --revision 69274a0d8dff5dd35bcee8290612f71e03b6e981 --local-dir /root/models/target
huggingface-cli download YourHighnessLA/Qwen3.8-27B-DFlash2-NVFP4 --local-dir /root/models/draft
```

### 3. 环境验证代码

执行命令：

```bash
python /root/vllm-sm12x-nvfp4-dflash2/verify_env.py
```

验证脚本内容：

```python
# verify_env.py
import torch, vllm, triton
print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
print("vllm:", vllm.__version__, "| triton:", triton.__version__)
assert torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0), "| SM:", torch.cuda.get_device_capability(0))
print("ALL OK")
```

预期输出：

```
torch: 2.13.0+cu130 | cuda: 13.0
vllm: 0.27.2.dev0+g6e448d0ea.d20260822 | triton: 3.7.1
GPU: NVIDIA GeForce RTX 5090 | SM: (12, 0)
ALL OK
```
