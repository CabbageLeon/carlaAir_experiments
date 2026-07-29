# OpenFly-Agent 推理部署

## 1. 模型

[HuggingFace: IPEC-COMMUNITY/openfly-agent-7b](https://huggingface.co/IPEC-COMMUNITY/openfly-agent-7b)

> 如果下载时触发 LLaMA-2 权限检查，需在 HF 申请 [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) 授权并 `huggingface-cli login`。

## 2. 安装依赖

```bash
conda create -n openfly python=3.10 -y && conda activate openfly

# PyTorch (按你的 CUDA 版本选)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 核心依赖 (版本必须精确)
pip install transformers==4.48.1 tokenizers==0.21.1 "timm>=0.9.10,<1.0.0" accelerate

# flash-attention
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation

# 工具
pip install numpy Pillow opencv-python
```

## 3. 部署

**Step 1** — 从本项目复制 4 个 HuggingFace 适配文件到你的目录：

```
本项目路径: train/extern/hf/
├── __init__.py
├── configuration_prismatic.py
├── modeling_prismatic.py
└── processing_prismatic.py
```

这 4 个文件完全自包含，无项目内部依赖。

**Step 2** — 创建推理脚本 `infer.py`：

```python
import argparse, numpy as np, torch, cv2
from PIL import Image
from transformers import (AutoConfig, AutoImageProcessor,
                          AutoModelForVision2Seq, AutoProcessor)
from extern.hf.configuration_prismatic import OpenFlyConfig
from extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from extern.hf.processing_prismatic import (PrismaticImageProcessor,
                                            PrismaticProcessor)

# 注册自定义模型类
AutoConfig.register("openvla", OpenFlyConfig)
AutoImageProcessor.register(OpenFlyConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenFlyConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenFlyConfig, OpenVLAForActionPrediction)

# 10 种离散动作模板
TEMPLATES = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],   # 0: stop
    [0, 3, 0, 0, 0, 0, 0, 0],   # 1: forward 3m
    [0, 0,15, 0, 0, 0, 0, 0],   # 2: turn left 15°
    [0, 0, 0,15, 0, 0, 0, 0],   # 3: turn right 15°
    [0, 0, 0, 0, 2, 0, 0, 0],   # 4: up 2m
    [0, 0, 0, 0, 0, 2, 0, 0],   # 5: down 2m
    [0, 0, 0, 0, 0, 0, 5, 0],   # 6: move left 5m
    [0, 0, 0, 0, 0, 0, 0, 5],   # 7: move right 5m
    [0, 6, 0, 0, 0, 0, 0, 0],   # 8: forward 6m
    [0, 9, 0, 0, 0, 0, 0, 0],   # 9: forward 9m
], dtype=np.float32)

NAMES = ["STOP", "前进3m", "左转15°", "右转15°", "上升2m",
         "下降2m", "左移5m", "右移5m", "快速前进6m", "最快前进9m"]


def load(model_path="IPEC-COMMUNITY/openfly-agent-7b", device="cuda:0"):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path, attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(device).eval()
    return model, processor


def predict(model, processor, image, instruction, unnorm_key="vln_norm"):
    # OpenFly 固定 3 帧输入，单步推理时复制同一张
    inputs = processor(instruction, [image, image, image]).to(
        model.device, dtype=torch.bfloat16)
    return model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--instruction", required=True)
    p.add_argument("--model", default="IPEC-COMMUNITY/openfly-agent-7b")
    args = p.parse_args()

    image = Image.fromarray(cv2.imread(args.image)).convert("RGB")
    model, processor = load(args.model)
    action = predict(model, processor, image, args.instruction)

    best = int(np.argmin(np.linalg.norm(TEMPLATES - action, axis=1)))
    print(f"动作向量: {np.array2string(action, precision=2)}")
    print(f"离散动作: {best} ({NAMES[best]})")


if __name__ == "__main__":
    main()
```

**Step 3** — 运行：

```bash
python infer.py --image drone.png --instruction "Go straight pass the river"
# 输出: 动作向量: [0.00 3.01 ...]  离散动作: 1 (前进3m)
```
