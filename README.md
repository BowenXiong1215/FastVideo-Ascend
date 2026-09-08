# FastVideo-Ascend：MiniMax-H3 Dense 四步 DMD2 训练指南

本仓库提供一个轻量补丁包，用于在 8 张昇腾 910B 上，为指定版本的 FastVideo 增加
MiniMax-H3 Dense T2VA 训练、Dense DMD2 四步蒸馏、student 导出、严格四步推理，以及
原始模型与四步模型的同条件性能对比能力。

本仓库不复制 FastVideo 源码，不包含 MiniMax-H3 权重或训练数据，也不包含 VSA。当前
Dense DMD2 配置是一套可以直接运行和继续调参的工程起点；它不等同于未公开的最终质量
recipe。

完整流程如下：

```text
基础镜像 + FastVideo 固定版本
              ↓
          应用补丁包
              ↓
    安装昇腾训练依赖并检查权重
              ↓
       准备有声视频训练数据
              ↓
        Dense SFT 单步验收
              ↓
 student + teacher + critic Dense DMD2
              ↓
      导出独立四步 student
              ↓
  原始 49 forward / student 4 forward 对比
```

## 1. 版本与资源要求

| 项目 | 推荐值 |
| --- | --- |
| NPU | 8 × Ascend 910B（单卡 64 GB） |
| 基础镜像 | `quay.io/ascend/triton:3.2.1-cann9.0.0-torch_npu2.7.1.post4-910b-ubuntu22.04-py3.11` |
| CANN | 9.0.0 |
| PyTorch | 2.7.1 |
| torch_npu | 2.7.1.post4 |
| Python | 3.11（最低 3.10） |
| FastVideo revision | `7bb76b5ec99807a66aa3047b901f15019abe0f00` |

MiniMax-H3 完整权重约占数百 GiB。DMD2 会同时构建 student、teacher、critic 三个 33B
Transformer，并通过 FSDP CPU offload 控制 NPU 显存，因此还需要充足的主机内存、交换
空间和磁盘容量。模型、预处理数据和 checkpoint 最好放在本机 NVMe。

## 2. 宿主机目录

先在宿主机设置路径。下面只是通用示例，可以替换成自己的绝对路径：

```bash
export FASTVIDEO_HOST=/path/to/FastVideo
export PATCH_REPO_HOST=/path/to/FastVideo-Ascend
export MINIMAX_H3_HOST=/path/to/MiniMax-H3
export TRAINING_MEDIA_HOST=/path/to/training-media
export OUTPUT_HOST=/path/to/fastvideo-output

mkdir -p "${TRAINING_MEDIA_HOST}"
mkdir -p "${OUTPUT_HOST}/runs" "${OUTPUT_HOST}/outputs"
```

获取固定版本源码和补丁仓库：

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git "${FASTVIDEO_HOST}"
git -C "${FASTVIDEO_HOST}" checkout 7bb76b5ec99807a66aa3047b901f15019abe0f00

git clone https://github.com/BowenXiong1215/FastVideo-Ascend.git \
  "${PATCH_REPO_HOST}"
```

训练服务器不能联网时，可在联网机器上下载两个仓库，再完整上传目录。模型权重也可以手工
下载后上传，但必须保留原始目录层级和文件名。

## 3. 获取和启动基础镜像

### 3.1 直接拉取

```bash
docker pull quay.io/ascend/triton:3.2.1-cann9.0.0-torch_npu2.7.1.post4-910b-ubuntu22.04-py3.11
```

### 3.2 在机器之间传输镜像

在有镜像的机器上导出：

```bash
docker save \
  quay.io/ascend/triton:3.2.1-cann9.0.0-torch_npu2.7.1.post4-910b-ubuntu22.04-py3.11 \
  | gzip > ascend-triton-fastvideo.tar.gz
```

上传到目标机器后可以直接加载 gzip 压缩的 Docker 镜像归档：

```bash
docker load -i ascend-triton-fastvideo.tar.gz
```

只有 `docker save` 生成的归档才能使用 `docker load`；普通目录通过 `tar -czf` 压缩后并不
会变成 Docker 镜像。

### 3.3 启动容器

```bash
docker run -it \
  --name fastvideo-h3-ascend \
  --network host \
  --ipc host \
  --privileged \
  --shm-size 256g \
  -v "${FASTVIDEO_HOST}:/workspace/FastVideo" \
  -v "${PATCH_REPO_HOST}:/workspace/FastVideo-Ascend:ro" \
  -v "${MINIMAX_H3_HOST}:/models/MiniMax-H3:ro" \
  -v "${TRAINING_MEDIA_HOST}:/data/media:ro" \
  -v "${OUTPUT_HOST}/runs:/workspace/FastVideo/runs" \
  -v "${OUTPUT_HOST}/outputs:/workspace/FastVideo/outputs" \
  quay.io/ascend/triton:3.2.1-cann9.0.0-torch_npu2.7.1.post4-910b-ubuntu22.04-py3.11 \
  bash
```

这里没有使用 `--rm`，退出容器后可以用以下命令再次进入：

```bash
docker start fastvideo-h3-ascend
docker exec -it fastvideo-h3-ascend bash
```

如果集群要求显式挂载 Ascend 驱动和设备，请按本机驱动安装方式补充挂载。容器的 CANN
userspace 版本应与宿主机驱动兼容。

## 4. 初始化容器环境

以下命令均在容器内运行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=1800
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

| 环境变量 | 含义 |
| --- | --- |
| `ASCEND_RT_VISIBLE_DEVICES` | 暴露给容器进程的 NPU 编号；单机八卡应包含 `0` 到 `7` |
| `PYTORCH_NPU_ALLOC_CONF` | 启用可扩展显存段，降低大模型运行时的显存碎片 |
| `HCCL_CONNECT_TIMEOUT` | 分布式进程建立 HCCL 连接的超时时间，单位为秒 |
| `HF_HUB_OFFLINE` | 禁止 Hugging Face Hub 联网，全部资源从本地模型目录读取 |
| `TRANSFORMERS_OFFLINE` | 禁止 Transformers 自动联网下载配置或权重 |

检查八张 NPU 和 Python 组件：

```bash
npu-smi info

python - <<'PY'
import torch
import torch_npu

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("NPU available:", torch.npu.is_available())
print("NPU count:", torch.npu.device_count())
PY
```

安装视频保存和检查工具：

```bash
apt-get update
apt-get install -y ffmpeg
ffmpeg -version
ffprobe -version
```

## 5. 应用补丁

安装脚本以 `sed -i` 更新指定 FastVideo 文件，并把新增文件复制到源码树。它可以重复执行。

```bash
cd /workspace/FastVideo-Ascend

bash patches/fastvideo-ascend-910b-patch-20260904/install.sh \
  /workspace/FastVideo

bash patches/fastvideo-ascend-910b-patch-20260904/verify.sh \
  /workspace/FastVideo
```

预期看到：

```text
FastVideo Ascend 910B Dense DMD2 bring-up patch: PASS
FastVideo Ascend 910B Dense DMD2 bring-up tree: PASS
```

也可以只上传发布包：

```text
patches/fastvideo-ascend-910b-patch-20260904.tar.gz
patches/fastvideo-ascend-910b-patch-20260904.tar.gz.sha256
```

校验并解压：

```bash
sha256sum -c fastvideo-ascend-910b-patch-20260904.tar.gz.sha256
tar -xzf fastvideo-ascend-910b-patch-20260904.tar.gz

bash fastvideo-ascend-910b-patch-20260904/install.sh \
  /workspace/FastVideo
```

## 6. 安装训练依赖

```bash
cd /workspace/FastVideo
bash scripts/install_ascend_dependencies.sh
```

该安装器固定以下核心版本：

```text
torch==2.7.1
torch_npu==2.7.1.post4
torchvision==0.22.1
torchaudio==2.7.1
```

同时安装 H3 训练路径需要的 `ftfy`、`remote-pdb`、PyAV、Diffusers、PyArrow 等模块。
不要再执行无版本约束的 `pip install -r requirements/ascend-training.txt`，否则 pip 可能重新
解析并替换镜像内的 PyTorch/torch_npu 组合。

检查依赖：

```bash
python - <<'PY'
import av
import diffusers
import ftfy
import pyarrow
import remote_pdb
import torch
import torch_npu
import torchaudio
import torchvision

print("Ascend training imports: PASS")
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
PY
```

## 7. 检查 MiniMax-H3 权重

模型目录需要保持 Diffusers 组件布局，至少包括：

```text
MiniMax-H3/
├── transformer/
├── vae/
├── audio_vae/
├── text_encoder/
├── tokenizer/
├── processor/
├── scheduler/
└── audio_scheduler/
```

执行完整性检查：

```bash
cd /workspace/FastVideo

python scripts/verify_minimax_h3_checkpoint.py \
  /models/MiniMax-H3 \
  2>&1 | tee /tmp/minimax_h3_verify.log
```

检查器会验证组件目录、JSON、safetensors 索引和分片、空文件、Git LFS 指针以及
safetensors header，不会把全部权重加载进内存。继续前应看到：

```text
MiniMax-H3 checkpoint verification: PASS
```

如果训练机能够访问 ModelScope，可在确认仓库提供相同 Diffusers 目录结构后下载：

```bash
pip install modelscope
modelscope download --model MiniMax/MiniMax-H3 \
  --local_dir /models/MiniMax-H3
```

下载完成后仍要运行上述检查器。面向 ComfyUI 的扁平化权重不能直接代替当前 loader 所需的
Diffusers 组件目录。

## 8. 准备训练数据

### 8.1 检查原始视频

第一轮可使用任意一条本地 MP4，但视频必须含音轨，并且按 24 FPS 处理后不少于 124 帧。

```bash
ffprobe -v error \
  -show_entries stream=index,codec_type,codec_name,width,height,r_frame_rate,duration \
  -of default=noprint_wrappers=1 \
  /data/media/sample-with-audio.mp4
```

### 8.2 生成单样本预处理数据

```bash
cd /workspace/FastVideo

export MINIMAX_H3_MODEL_PATH=/models/MiniMax-H3
export TRAINING_VIDEO_PATH=/data/media/sample-with-audio.mp4
export TRAINING_CAPTION='Describe the visible action, scene, camera motion, speech, music, and environmental sounds precisely.'

bash examples/train/prepare_minimax_h3_ascend.sh \
  2>&1 | tee /tmp/minimax_h3_preprocess.log
```

预期输出：

```text
Validated one MiniMax H3 training row in ...
```

默认数据目录是：

```text
/workspace/FastVideo/data/crush-smol_h3_t2va_single_sample_preprocessed
```

### 8.3 扩展到正式数据集

正式训练时，每条记录仍需提供与单样本产物相同的预处理字段，包括文本条件、视频 latent、
音频 latent 以及对应元数据。最稳妥的做法是先保留单样本目录作为 schema 模板，再通过
FastVideo 的预处理入口批量生成 Parquet；不要手工猜测 tensor shape 或字段 dtype。

质量主要取决于视频和音频的清晰度、caption 对画面与声音的覆盖程度、数据多样性，以及
数据分布是否接近目标场景。正式训练前应抽样解码预处理结果，并统计损坏样本、时长、尺寸、
帧率和有无音轨。

## 9. 先运行 Dense SFT 验收

只构建配置、模型和数据加载器：

```bash
cd /workspace/FastVideo

NUM_NPUS=8 bash examples/train/run_ascend.sh \
  examples/train/configs/ascend/minimax_h3_t2va_sft_smoke.yaml \
  --dry-run
```

`--dry-run` 的 `Training completed` 表示运行时构建完成，不包含参数更新。

执行一个真实 optimizer step：

```bash
NUM_NPUS=8 bash examples/train/run_ascend.sh \
  examples/train/configs/ascend/minimax_h3_t2va_sft_smoke.yaml
```

这一步用于确认 Dense SDPA、BF16、HCCL、FSDP/HSDP、sequence parallel、视频/音频联合
前后向和 optimizer 更新能够共同运行。完成标准是所有 rank 正常退出且 loss 为有限值。
SFT smoke 的 `training_state_checkpointing_steps` 默认为 `0`，因此这一步不保存训练状态。

## 10. Dense DMD2 四步训练

### 10.1 检查配置和路径

```bash
cd /workspace/FastVideo

python scripts/verify_minimax_h3_dmd2_config.py \
  examples/train/configs/ascend/minimax_h3_t2va_dmd2_4step_smoke.yaml \
  --check-paths
```

预期：

```text
MiniMax-H3 Dense DMD2 four-step config: PASS
```

### 10.2 DMD2 dry-run

```bash
NUM_NPUS=8 bash examples/train/run_ascend.sh \
  examples/train/configs/ascend/minimax_h3_t2va_dmd2_4step_smoke.yaml \
  --dry-run
```

### 10.3 单步真实训练

```bash
NUM_NPUS=8 bash examples/train/run_ascend.sh \
  examples/train/configs/ascend/minimax_h3_t2va_dmd2_4step_smoke.yaml
```

一次 DMD2 训练迭代包含：student 四步 rollout、teacher score、critic score、student
backward、critic backward、两个 optimizer 更新和 checkpoint 保存。rank 0 会输出当前
阶段和运行时长；长阶段每 60 秒输出一次心跳。

### 10.4 正式训练配置

保留 smoke 配置作为可复现基线，复制一份再修改：

```bash
cp examples/train/configs/ascend/minimax_h3_t2va_dmd2_4step_smoke.yaml \
  examples/train/configs/ascend/minimax_h3_t2va_dmd2_4step_train.yaml
```

至少调整数据路径、输出路径和训练步数，例如：

```bash
NUM_NPUS=8 bash examples/train/run_ascend.sh \
  examples/train/configs/ascend/minimax_h3_t2va_dmd2_4step_train.yaml \
  --training.data.data_path /workspace/FastVideo/data/my_h3_preprocessed \
  --training.checkpoint.output_dir runs/minimax_h3_dense_dmd2_4step \
  --training.loop.max_train_steps 1000 \
  --training.checkpoint.training_state_checkpointing_steps 50 \
  2>&1 | tee /tmp/minimax_h3_dmd2_train.log
```

命令行使用点号路径覆盖 YAML，适合临时实验；稳定实验应把最终值写入独立 YAML 并和训练
产物一起保存。

### 10.5 续训

从输出目录里最新的完整 checkpoint 继续：

```bash
NUM_NPUS=8 bash examples/train/run_ascend.sh \
  examples/train/configs/ascend/minimax_h3_t2va_dmd2_4step_train.yaml \
  --training.checkpoint.resume_from_checkpoint latest
```

也可指定 checkpoint 路径：

```bash
NUM_NPUS=8 bash examples/train/run_ascend.sh \
  examples/train/configs/ascend/minimax_h3_t2va_dmd2_4step_train.yaml \
  --training.checkpoint.resume_from_checkpoint \
  runs/minimax_h3_dense_dmd2_4step/checkpoint-500
```

## 11. 参数说明与调参方向

以下名称对应补丁提供的 SFT/DMD2 YAML 和 `run_ascend.sh`。不同版本 FastVideo 的字段层级
可能变化，因此应以本仓库固定 revision 为准。

### 11.1 `models`

`models` 下包含 `student`、`teacher`、`critic` 三个角色；每个角色使用同一组模型字段：

| 参数 | 含义 | 调整建议 |
| --- | --- | --- |
| `_target_` | FastVideo 模型包装类 | 保持 `fastvideo.train.models.minimax_h3.MiniMaxH3Model` |
| `init_from` | 本地 MiniMax-H3 根目录 | 离线训练必须指向完整本地目录 |
| `trainable` | 该角色是否参与反向和参数更新 | student/critic 为 `true`，teacher 为 `false` |
| `disable_custom_init_weights` | 是否跳过为新模型执行的自定义初始化 | teacher/critic 从基础权重加载时保持配置值 |
| `enable_gradient_checkpointing_type` | activation checkpoint 类型 | student/critic 为 `full`，以重算换显存 |
| `attention_backend` | 注意力实现 | 当前路线固定 Dense `TORCH_SDPA` |

student 是最终导出的模型；teacher 是冻结的真实分布 score 模型；critic 是可训练的 fake-score
模型。

### 11.2 `method`

| 参数 | 含义 | 调整建议 |
| --- | --- | --- |
| `_target_` | 训练方法类 | DMD2 保持 `MiniMaxH3DMD2Method`；SFT 使用 `FineTuneMethod` |
| `rollout_mode` | student rollout 的生成模式 | 当前使用可训练的模拟 rollout 路径，保持默认 |
| `dmd_denoising_steps` | student rollout 的四个训练时间点 | 默认 `[999, 749, 500, 250]`；它直接定义四步蒸馏轨迹 |
| `generator_update_interval` | 每隔多少个 critic step 更新一次 student | `1` 表示每步都更新 student；增大可降 student 更新频率，但会改变训练动力学 |
| `real_score_guidance_scale` | teacher real score 的 guidance 权重 | 默认 `1.0`；改变它会调整 DMD 梯度目标 |
| `min_timestep_ratio` | teacher/critic 随机加噪时间范围下界 | 默认 `0.02`，避免极端低噪声端点 |
| `max_timestep_ratio` | teacher/critic 随机加噪时间范围上界 | 默认 `0.98`，避免极端高噪声端点 |
| `fake_score_learning_rate` | critic/fake-score 模型学习率 | 默认 `1e-6`；critic 过慢可小幅提高，震荡时降低 |
| `fake_score_betas` | critic Adam 一阶/二阶动量系数 | 默认 `[0.0, 0.999]`，先保持不变 |
| `fake_score_lr_scheduler` | critic 学习率调度器 | smoke 为 `constant`；长训在有实验依据后再改变 |

`dmd_denoising_steps` 控制 student rollout；`min_timestep_ratio` 和
`max_timestep_ratio` 控制 teacher/critic score 的随机采样区间，两者不是同一组时间步。

### 11.3 `training.distributed`

| 参数 | 含义 | 默认/建议 |
| --- | --- | --- |
| `num_gpus` | 当前节点训练进程数 | 单机八卡设为 `8` |
| `sp_size` | sequence parallel 组大小 | H3 当前八卡配置使用 `8` |
| `tp_size` | tensor parallel 组大小 | 当前为 `1` |
| `hsdp_replicate_dim` | HSDP 复制维度 | 单机配置为 `1` |
| `hsdp_shard_dim` | HSDP 分片维度 | 单机八卡配置为 `8` |
| `fsdp_cpu_offload` | 将 FSDP 参数/梯度状态卸载到 CPU | DMD2 三模型场景保持 `true` |
| `pin_cpu_memory` | CPU offload 是否使用锁页内存 | 默认 `false`；主机内存和 `ulimit -l` 充足时可设 `true` |

只在主机允许足够锁页内存时尝试：

```bash
ulimit -l

NUM_NPUS=8 bash examples/train/run_ascend.sh \
  examples/train/configs/ascend/minimax_h3_t2va_dmd2_4step_train.yaml \
  --training.distributed.pin_cpu_memory true
```

### 11.4 分布式启动环境变量

| 参数 | 含义 | 单机八卡示例 |
| --- | --- | --- |
| `NUM_NPUS` | 每节点进程数 | `8` |
| `NNODES` | 节点数量 | `1` |
| `NODE_RANK` | 当前节点编号 | `0` |
| `MASTER_ADDR` | rank 0 节点地址 | 单机可用 `127.0.0.1` |
| `MASTER_PORT` | torchrun rendezvous 端口 | 选择未占用端口，例如 `29500` |

单机完整写法：

```bash
NUM_NPUS=8 \
NNODES=1 \
NODE_RANK=0 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29500 \
bash examples/train/run_ascend.sh \
  examples/train/configs/ascend/minimax_h3_t2va_dmd2_4step_train.yaml
```

### 11.5 `training.data`

| 参数 | 含义 | 调整建议 |
| --- | --- | --- |
| `data_path` | 预处理 Parquet 数据目录 | 指向第 8 节产物或正式数据集 |
| `preprocessed_data_type` | 预处理数据任务类型 | MiniMax-H3 文生视频音频使用 `t2va` |
| `train_batch_size` | 每个 data-parallel rank 的 batch size | MiniMax-H3 当前要求 `1` |
| `training_cfg_rate` | 训练期丢弃文本条件的概率 | H3 当前要求 `0.0` |
| `dataloader_num_workers` | 每个 rank 的数据读取进程数 | 从较小值开始，观察 CPU、内存和存储吞吐 |
| `seed` | 数据与训练随机种子 | 对比实验保持相同；smoke 为 `42` |
| `num_latent_t` | 训练 latent 时间长度 | smoke 为 `37`，与 124 帧预处理设置匹配 |
| `num_height` / `num_width` | 训练空间尺寸 | smoke 为 `768 × 1344`；降低会影响训练目标和质量 |
| `num_frames` | 原视频采样帧数 | smoke 为 `124`；必须与预处理和 latent 长度一致 |

在本路线中，降低帧数、分辨率或修改采样规则都属于改变训练任务，不是纯性能优化。

### 11.6 `training.optimizer`

| 参数 | 含义 | 调整建议 |
| --- | --- | --- |
| `learning_rate` | student 学习率 | DMD2 默认 `1e-6`；SFT smoke 默认 `5e-5` |
| `betas` | Adam 的动量系数 | 首轮保持默认，调整时与学习率联合观察 |
| `weight_decay` | 权重衰减 | 保持 YAML 默认，除非有明确正则化实验 |
| `lr_scheduler` | student 学习率计划 | smoke 为 constant；长训可添加 warmup 和衰减 |
| `lr_warmup_steps` | 学习率 warmup 步数 | smoke 为 `0`；长训可按总步数设置小比例 warmup |
| `lr_num_cycles` | 周期型 scheduler 的周期数 | 只在所选 scheduler 使用该字段时生效 |
| `lr_power` | polynomial scheduler 的幂 | 只在 polynomial 调度时生效 |
| `min_lr_ratio` | 最低学习率相对初始学习率的比例 | 默认 `0.5`，仅由支持该字段的 scheduler 使用 |

student optimizer 与由 `method.fake_score_*` 配置的 critic optimizer 是两套独立
optimizer。调参时应分别记录 student loss、critic loss、梯度范数和学习率，不能只观察
总 loss。

### 11.7 `training.loop`

| 参数 | 含义 | 调整建议 |
| --- | --- | --- |
| `max_train_steps` | optimizer 总迭代数 | smoke 为 `1`；正式训练按数据规模和验证结果增加 |
| `gradient_accumulation_steps` | 累积多少个 micro-batch 再更新 | 增大会改变有效 batch 和更新频率，不能视为完全等价加速 |

### 11.8 `training.checkpoint`

| 参数 | 含义 | 调整建议 |
| --- | --- | --- |
| `output_dir` | DCP checkpoint 输出目录 | 放在持久化挂载目录 |
| `training_state_checkpointing_steps` | 每隔多少步保存训练状态 | DMD2 smoke 为 `1`；长训可设 `25`、`50` 或更大 |
| `checkpoints_total_limit` | 最多保留多少个 checkpoint | 按磁盘容量和恢复需求设置 |
| `resume_from_checkpoint` | 恢复来源 | 使用 `latest` 或具体 `checkpoint-N` 路径 |

降低保存频率不会改变参数更新，但会增加意外中断时需要重跑的步数。DCP checkpoint 是训练
状态，不能直接当成推理模型；推理前必须执行第 12 节导出。

### 11.9 tracker、模型与回调配置

| 参数 | 含义 | 建议 |
| --- | --- | --- |
| `training.tracker.trackers` | 实验记录后端列表 | 离线 smoke 使用 `[none]` |
| `training.tracker.entity` | tracker 账户或组织 | 仅在所选 tracker 需要时填写 |
| `training.tracker.project_name` | 实验项目名 | 建议固定为同一研究主题 |
| `training.tracker.run_name` | 单次实验名 | 建议包含数据版本、学习率和时间表 |
| `training.model.precondition_outputs` | 是否对模型输出做训练预条件处理 | 当前 H3 DMD2 为 `false` |
| `training.model.enable_gradient_checkpointing_type` | 训练层面的 activation checkpoint 类型 | 当前为 `full` |
| `training.dit_precision` | DiT 训练精度 | 当前为 `bf16` |
| `callbacks.grad_clip._target_` | 梯度裁剪回调类 | 保持补丁配置的 `GradNormClipCallback` |
| `callbacks.grad_clip.max_grad_norm` | 梯度范数裁剪阈值 | 默认 `1.0`，梯度尖峰时可适当降低 |
| `pipeline` | 可选 pipeline 配置 | 当前为空字典 `{}`，使用模型方法自身的训练流程 |

### 11.10 推荐调参顺序

1. 固定模型、数据 schema、四步时间表、分辨率、帧数和随机种子，确认数百步训练稳定。
2. 分别扫描 student 与 critic 学习率，记录 loss、梯度范数和固定 prompt 输出。
3. 调整 student/critic 更新比例，即 `generator_update_interval`。
4. 再评估 timestep 采样范围和 `real_score_guidance_scale`。
5. 最后才尝试四步时间表、数据配比和更长训练。

判断质量不能只看训练 loss。至少要固定一组未参与训练的 prompt 和 seed，对比视频主体一致性、
时间稳定性、运动幅度、镜头运动、音画同步、语音/音乐可懂度和整体审美。

## 12. 导出四步 student

训练输出是分布式 DCP 状态。使用下面的脚本导出可独立加载的 Diffusers 模型目录：

```bash
cd /workspace/FastVideo

bash examples/train/export_minimax_h3_dmd2_ascend.sh \
  runs/minimax_h3_dense_dmd2_4step/checkpoint-1000 \
  runs/minimax_h3_dense_dmd2_4step_export \
  2>&1 | tee /tmp/minimax_h3_dmd2_export.log
```

两个位置参数分别是：

1. 输入 DCP checkpoint 目录，例如 `checkpoint-1000`。
2. 输出的完整四步 student 模型目录。

脚本以基础模型目录为模板，恢复并写入 student Transformer，同时保留推理所需的 tokenizer、
text encoder、VAE、audio VAE 和 scheduler 等组件。支持硬链接的文件系统会尽量复用未修改
组件；否则导出可能复制大量文件，需要预留足够磁盘空间。

预期：

```text
Strict reload verification passed.
Exported four-step student: ...
```

## 13. 严格四步推理

```bash
cd /workspace/FastVideo

python examples/inference/basic/basic_minimax_h3_dense_4step_ascend.py \
  --model-path runs/minimax_h3_dense_dmd2_4step_export \
  --prompt 'A cinematic ocean wave crashes against dark rocks, with synchronized roaring water and wind.' \
  --output outputs/minimax_h3_dense_dmd2_4step \
  2>&1 | tee /tmp/minimax_h3_dense_4step_inference.log
```

该入口固定以下条件：

- Dense `TORCH_SDPA`；
- `guidance_scale=1.0`；
- 5 个 sigma 网格点，对应 4 次 DiT forward；
- 8 卡 sequence parallel；
- strict eager 执行；
- 不启用 VSA、FA4、CUDA 专用 kernel、compile 或数值顺序可能变化的融合。

主要参数：

| 参数 | 含义 |
| --- | --- |
| `--model-path` | 第 12 节导出的 student 目录 |
| `--prompt` | 同时描述画面、动作、镜头与声音的文本条件 |
| `--output` | 输出目录或输出前缀 |
| `--seed` | 随机种子；同条件对比时必须固定 |
| `--height` / `--width` | 输出分辨率，应与训练分布相符 |
| `--num-frames` | 输出帧数，应与训练时长分布相符 |

输出目录中应生成带音频的 MP4。可以检查媒体流：

```bash
ffprobe -v error \
  -show_entries stream=index,codec_type,codec_name,width,height,r_frame_rate,duration \
  -of default=noprint_wrappers=1 \
  outputs/minimax_h3_dense_dmd2_4step/*.mp4
```

## 14. 原始 H3 与四步 student 对比

使用同一 prompt、seed、尺寸和 Dense strict eager 路径依次运行两个模型：

```bash
cd /workspace/FastVideo

python examples/inference/basic/compare_minimax_h3_dense_ascend.py \
  --base-model-path /models/MiniMax-H3 \
  --student-model-path runs/minimax_h3_dense_dmd2_4step_export \
  --prompt 'A cinematic ocean wave crashes against dark rocks, with synchronized roaring water and wind.' \
  --seed 12345 \
  --output outputs/minimax_h3_dense_comparison
```

默认设置：

| 模型 | sigma 网格点 | DiT forward 次数 |
| --- | ---: | ---: |
| 原始 MiniMax-H3 | 50 | 49 |
| 四步 student | 5 | 4 |

主要参数：

| 参数 | 含义 |
| --- | --- |
| `--base-model-path` | 原始 MiniMax-H3 Diffusers 目录 |
| `--student-model-path` | 导出的四步 student 目录 |
| `--prompt` | 两个模型共用的提示词 |
| `--seed` | 两个模型共用的随机种子 |
| `--output` | 对比结果根目录 |
| `--height` / `--width` | 两次运行共用的输出尺寸 |
| `--num-frames` | 两次运行共用的帧数 |

两个模型在独立子进程中运行，前一个模型的 worker 和显存释放后才启动后一个模型。输出结构
如下：

```text
outputs/minimax_h3_dense_comparison/
├── base/
│   └── *.mp4
├── student/
│   └── *.mp4
├── base.log
├── student.log
├── base_run_config.json
├── student_run_config.json
├── comparison.json
└── comparison.md
```

报告会区分：

- 端到端耗时：包含模型加载、conditioning、denoising、VAE 解码和保存；
- denoising 耗时：主要反映 49 次与 4 次 DiT forward 的差异；
- 平均每次 DiT forward 耗时：用于观察单步计算成本；
- 端到端与 denoising 加速比。

性能比较应至少运行多次，并把首次模型加载和缓存影响与稳定运行分开统计。质量比较应使用
固定的多组 prompt/seed，由人工检查画面与声音，也可以在独立评测环境增加视频质量、文本
一致性和音画同步指标。单步 smoke 的目标是验证模型可训练、可导出和可四步生成，不代表已
达到最终质量。

## 15. 补丁包结构

```text
patches/
├── fastvideo-ascend-910b-patch-20260904/
│   ├── payload/
│   │   ├── replacements/
│   │   └── additions/
│   ├── install.sh
│   ├── verify.sh
│   ├── modified.tsv
│   └── added.tsv
├── fastvideo-ascend-910b-patch-20260904.tar.gz
└── fastvideo-ascend-910b-patch-20260904.tar.gz.sha256
```

## License

补丁代码延续 FastVideo 的 Apache-2.0 许可证。FastVideo、MiniMax-H3、训练数据和第三方
组件的许可及权利归各自作者所有。
