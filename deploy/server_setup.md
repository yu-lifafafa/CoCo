# CoCo 烟草 Task 6B：Linux 服务器部署与 GPU Smoke 操作手册

本手册只执行部署核验与有硬上限的 GPU Smoke，不启动正式训练、完整 Epoch、val 推理或 test 推理。所有服务器实测结果必须来自实际日志；当前交付状态仍为 `SERVER GPU TEST: NOT RUN`。

## 1. 设置服务器变量

以下路径都是占位符，必须改成服务器上的真实绝对路径。使用全新的工作目录，不覆盖任何已有代码或未保存修改。

```bash
COCO_BASELINE_DIR="/srv/coco-official-baseline"
COCO_WORKDIR="/srv/coco-tobacco-task6b"
PACKAGE_DIR="/srv/coco-task6a-package"
MANIFEST_PATH="/data/tobacco/image_manifest_updated.csv"
DATA_ROOT="/data/tobacco"
BACKBONE_WEIGHTS="/srv/weights/resnet101-63fe2227.pth"
BACKBONE_SHA256="<fill-after-independent-verification>"
SMOKE_OUTPUT="/srv/coco-smoke-results"
CUDA_DEVICE="cuda:0"
EXPECTED_MANIFEST_SHA256="ce51205b537c3d1eee28b3d8112ce1edb6454c6f16cd3b441387bff2be2c56b5"
BASELINE_COMMIT="342c4e35b19fc773ae71565dbe17be70d56bfcdc"
```

先拒绝会覆盖现有内容的目标：

```bash
test ! -e "$COCO_BASELINE_DIR" || { echo "STOP: baseline directory exists"; exit 1; }
test ! -e "$COCO_WORKDIR" || { echo "STOP: work directory exists"; exit 1; }
test -d "$PACKAGE_DIR" || { echo "STOP: package directory missing"; exit 1; }
test -f "$PACKAGE_DIR/coco_tobacco_deploy.zip"
test -f "$PACKAGE_DIR/deployment_manifest.json"
test -f "$PACKAGE_DIR/verify_deployment.py"
```

## 2. 核验官方基线，不覆盖脏工作树

```bash
git clone https://github.com/jasonseu/CoCo.git "$COCO_BASELINE_DIR"
cd "$COCO_BASELINE_DIR"
git checkout "$BASELINE_COMMIT"
test "$(git rev-parse HEAD)" = "$BASELINE_COMMIT"
git diff --quiet
git diff --cached --quiet
python "$PACKAGE_DIR/verify_deployment.py" baseline \
  --repo-root "$COCO_BASELINE_DIR" \
  --manifest "$PACKAGE_DIR/deployment_manifest.json"
```

如果 clone、commit 或 clean 检查失败，停止。部署快照写入另一个全新目录，官方基线目录只用于来源核验和回退证据。

## 3. 部署包 SHA、成员与安全解压核验

记录外层文件 SHA：

```bash
cd "$PACKAGE_DIR"
sha256sum coco_tobacco_deploy.zip deployment_manifest.json verify_deployment.py
python verify_deployment.py archive \
  --archive "$PACKAGE_DIR/coco_tobacco_deploy.zip" \
  --manifest "$PACKAGE_DIR/deployment_manifest.json"
python verify_deployment.py extract \
  --archive "$PACKAGE_DIR/coco_tobacco_deploy.zip" \
  --manifest "$PACKAGE_DIR/deployment_manifest.json" \
  --destination "$COCO_WORKDIR"
python "$COCO_WORKDIR/deploy/verify_deployment.py" tree \
  --root "$COCO_WORKDIR" \
  --manifest "$PACKAGE_DIR/deployment_manifest.json"
```

校验器会拒绝 ZIP SHA 不符、缺失/额外/损坏成员、绝对路径、`..` 穿越、重复规范化路径和符号链接。不要用 `unzip -o` 覆盖已有目录。

## 4. 创建隔离 Conda 环境并核验 GPU 栈

`deploy/environment.yml` 固定 Python 3.10 和已确认的非 CUDA 依赖，但故意不猜测服务器的 torch、torchvision 或 CUDA 版本。先采集驱动证据：

```bash
nvidia-smi
conda env create -f "$COCO_WORKDIR/deploy/environment.yml"
conda activate coco-tobacco
python --version
```

随后根据服务器 `nvidia-smi` 的驱动能力，使用 PyTorch 官方安装选择器安装相互匹配的 torch 与 torchvision。把实际安装命令和版本保存在 Task 6B 日志中；不要沿用未知的全局环境，也不要盲目安装或升级系统 CUDA。

安装完成后执行只读依赖审计：

```bash
python -c 'import torch, torchvision; print("torch", torch.__version__); print("torchvision", torchvision.__version__); print("torch CUDA", torch.version.cuda); print("available", torch.cuda.is_available()); print("count", torch.cuda.device_count()); print("device0", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'
python -c 'import numpy, scipy, PIL, sklearn, yaml, tqdm, timm, randaugment, tensorboard; print("non-CUDA imports PASS")'
cd "$COCO_WORKDIR"
python -c 'import models; from models.factory import backbone_entrypoint, model_entrypoint; print(backbone_entrypoint("resnet101")[1]); print(model_entrypoint("mlic"))'
```

默认 ResNet101 路径不需要外部 `inplace_abn`。只有选择 TResNet 时才需要另行安装并验证该编译依赖；本次 Smoke 禁止切换 Backbone。

## 5. 核验 manifest 与 train/val 数据（绝不使用 test 调试训练）

```bash
test -f "$MANIFEST_PATH"
test -d "$DATA_ROOT/train"
test -d "$DATA_ROOT/val"
printf '%s  %s\n' "$EXPECTED_MANIFEST_SHA256" "$MANIFEST_PATH" | sha256sum --check --strict
cd "$COCO_WORKDIR"
python - "$MANIFEST_PATH" "$DATA_ROOT" <<'PY'
import sys
from lib.tobacco_dataset import TobaccoManifestDataset

manifest, root = sys.argv[1:]
for split, expected in (("train", 10050), ("val", 2866)):
    dataset = TobaccoManifestDataset(manifest, root, split=split, transform=None)
    if len(dataset) != expected:
        raise SystemExit(f"{split}: expected {expected}, got {len(dataset)}")
    print(split, len(dataset), "paths exist")
PY
```

上面的 Dataset 构造检查 manifest 协议和每个 train/val 文件是否存在。Smoke 只会根据 train 标签确定性选择健康、单病、多病与混合样本；不会构造 val/test Dataset。若需要在 Smoke 前全面检查图像解码，可在不修改文件的前提下对 train/val 的 `record.image_path` 逐个运行 `PIL.Image.open(path).verify()`，任何失败都必须停止并报告，不能跳过。

已知本机 test 集的 `images/task_16/IMG_7950.png` 曾无法由 Pillow 解码；本阶段不读取 test，也不能据此宣称服务器 test 已通过。

## 6. 核验 ImageNet ResNet101 初始化证据

官方 `models/resnet.py` 会通过 `torch.utils.model_zoo.load_url` 请求 `resnet101-63fe2227.pth`。这是 ImageNet Backbone 权重，不是官方完整 CoCo checkpoint，也不是烟草 18 类最佳 EMA checkpoint。

推荐 Task 6B 使用显式离线文件：

```bash
test -f "$BACKBONE_WEIGHTS"
test "$BACKBONE_SHA256" != "<fill-after-independent-verification>"
printf '%s  %s\n' "$BACKBONE_SHA256" "$BACKBONE_WEIGHTS" | sha256sum --check --strict
```

`--backbone-init local-file` 会把该文件复制到 Smoke 输出目录内隔离的 Torch Hub cache，再核对源/目标 SHA。`torch-cache` 只接受已经存在的标准缓存并在缺失时停止；`allow-download` 是唯一允许联网的显式模式。随机初始化不属于成功模式。

## 7. 运行有硬上限的 GPU Smoke

先创建独立输出根，不要指向 `experiments/`、`checkpoints/` 或 Task 5 的正式 calibration/evaluation 目录：

```bash
mkdir -p "$SMOKE_OUTPUT"
cd "$COCO_WORKDIR"
```

Batch Size 1，只测全健康单步：

```bash
python scripts/smoke_tobacco_coco.py \
  --manifest-path "$MANIFEST_PATH" \
  --data-root "$DATA_ROOT" \
  --img-size 384 \
  --batch-size 1 \
  --device "$CUDA_DEVICE" \
  --amp \
  --seed 42 \
  --backbone-init local-file \
  --backbone-weights "$BACKBONE_WEIGHTS" \
  --output-dir "$SMOKE_OUTPUT/bs1_384_amp" \
  --max-steps 1 \
  --scenarios healthy \
  --expected-manifest-sha256 "$EXPECTED_MANIFEST_SHA256"
```

Batch Size 2，分别测单病与多病，共两步：

```bash
python scripts/smoke_tobacco_coco.py \
  --manifest-path "$MANIFEST_PATH" \
  --data-root "$DATA_ROOT" \
  --img-size 384 \
  --batch-size 2 \
  --device "$CUDA_DEVICE" \
  --amp \
  --seed 42 \
  --backbone-init local-file \
  --backbone-weights "$BACKBONE_WEIGHTS" \
  --output-dir "$SMOKE_OUTPUT/bs2_384_amp" \
  --max-steps 2 \
  --scenarios single multi \
  --expected-manifest-sha256 "$EXPECTED_MANIFEST_SHA256"
```

Batch Size 4，测健康、单病、多病和混合 Batch，共四步：

```bash
python scripts/smoke_tobacco_coco.py \
  --manifest-path "$MANIFEST_PATH" \
  --data-root "$DATA_ROOT" \
  --img-size 384 \
  --batch-size 4 \
  --device "$CUDA_DEVICE" \
  --amp \
  --seed 42 \
  --backbone-init local-file \
  --backbone-weights "$BACKBONE_WEIGHTS" \
  --output-dir "$SMOKE_OUTPUT/bs4_384_amp" \
  --max-steps 4 \
  --scenarios healthy single multi mixed \
  --expected-manifest-sha256 "$EXPECTED_MANIFEST_SHA256"
```

每个输出目录只应出现 `smoke_result.json`、`smoke_result.log` 和隔离的 `torch_home`；不会生成 `best_ema_model.pth`。JSON 记录 GPU 型号、实际 Batch、输入尺寸、AMP、分类/对比/总 Loss、反向传播、有限梯度、optimizer/scheduler/EMA 更新、同步单步时间和显存峰值。

## 8. 停止条件与 Task 6B 交接

遇到下列任一情况立即停止，不自动降到 224，也不改模型结构：

- manifest SHA、数据数量、路径或图像解码失败；
- ResNet101 权重来源或 SHA 无法核验；
- torch/torchvision/CUDA 不匹配，或 CUDA 不可用；
- 任一 Loss/梯度出现 NaN/Inf；
- 全健康 Batch 的 Contrastive Loss 非零；
- backward、optimizer、scheduler 或 EMA 更新失败；
- CUDA OOM。先记录 384 下的真实 Batch Size、错误和显存，再从 4 降到 2、1；只有 384×384 的 Batch Size 1 仍明确不可行，才讨论 224×224。

Task 6B 应回传：`nvidia-smi`、Python/torch/torchvision/CUDA 版本、依赖 import 结果、manifest SHA、train/val 路径检查、Backbone 文件 SHA，以及三个 Smoke 输出目录的 JSON/log。不要启动正式训练。
