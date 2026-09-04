# FastVideo-Ascend

在华为昇腾 910B 上训练 MiniMax-H3 的 FastVideo 轻量补丁包。本仓库不复制完整
FastVideo 源码，也不包含模型权重和训练数据。

当前覆盖两层训练验收：

- MiniMax-H3 Dense T2VA 单步 SFT：验证 BF16 Dense SDPA、HCCL、FSDP/HSDP、
  sequence parallel、视频/音频联合前后向和优化器更新。
- MiniMax-H3 Dense DMD2 四步 bring-up：验证 student、teacher、critic 三模型联合训练、
  `[999, 749, 500, 250]` student ladder、DCP checkpoint、student 导出和四次 DiT
  forward 推理。

Dense DMD2 配置用于跑通工程闭环，不代表已经复现尚未公开的 FastH3 最终效果 recipe；
当前也不包含 VSA 训练。

## 1. 固定版本

| 组件 | 版本 |
| --- | --- |
| 设备 | 8 × Ascend 910B（64 GB） |
| 基础镜像 | `quay.io/ascend/triton:3.2.1-cann9.0.0-torch_npu2.7.1.post4-910b-ubuntu22.04-py3.11` |
| CANN | 9.0.0 |
| PyTorch | 2.7.1 |
| torch_npu | 2.7.1.post4 |
| FastVideo | `7bb76b5ec99807a66aa3047b901f15019abe0f00` |

建议准备充足的主机内存和共享存储。完整 MiniMax-H3 权重约占数百 GiB；DMD2 同时构建
三份 33B Transformer，并使用 FSDP CPU offload 换取 NPU 容量。

## 2. 在宿主机准备目录

下面的宿主机目录完全由使用者自行选择：

```bash
export FASTVIDEO_HOST=/path/to/FastVideo
export PATCH_REPO_HOST=/path/to/FastVideo-Ascend
export MINIMAX_H3_HOST=/path/to/MiniMax-H3
export TRAINING_MEDIA_HOST=/path/to/training-media
export OUTPUT_HOST=/path/to/fastvideo-output

mkdir -p "${TRAINING_MEDIA_HOST}" "${OUTPUT_HOST}/runs" "${OUTPUT_HOST}/outputs"
```

获取固定版本 FastVideo：

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git "${FASTVIDEO_HOST}"
git -C "${FASTVIDEO_HOST}" checkout 7bb76b5ec99807a66aa3047b901f15019abe0f00

git clone https://github.com/BowenXiong1215/FastVideo-Ascend.git \
  "${PATCH_REPO_HOST}"
```

无法从服务器访问 GitHub 时，可在联网机器上下载两个仓库并完整上传。模型同样可以手动
下载后上传，目录结构不得被压平或重新命名。

## 3. 启动容器

以下命令将宿主机目录映射到固定的容器内路径。`--privileged` 适合首次工程验收；生产环境
可按集群安全策略改成精确的设备和驱动挂载。

```bash
docker pull quay.io/ascend/triton:3.2.1-cann9.0.0-torch_npu2.7.1.post4-910b-ubuntu22.04-py3.11

docker run --rm -it \
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

如果集群要求显式挂载驱动目录，请按本机 Ascend 驱动安装方式补充挂载；不要用容器内的
CANN userspace 覆盖宿主机内核驱动。

进入容器后设置环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=1800
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

确认 NPU 和版本匹配：

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

## 4. 应用补丁

补丁固定对应上述 FastVideo commit。安装器使用 `sed -i` 更新官方文件，并复制新增文件；
可以重复执行，也能从本补丁的旧版原地升级。

```bash
cd /workspace/FastVideo-Ascend

bash patches/fastvideo-ascend-910b-patch-20260904/install.sh \
  /workspace/FastVideo

bash patches/fastvideo-ascend-910b-patch-20260904/verify.sh \
  /workspace/FastVideo
```

成功标志：

```text
FastVideo Ascend 910B Dense DMD2 bring-up patch: PASS
FastVideo Ascend 910B Dense DMD2 bring-up tree: PASS
```

## 5. 安装精简训练依赖

```bash
cd /workspace/FastVideo
bash scripts/install_ascend_dependencies.sh
```

不要直接执行无约束的 `pip install -r requirements/ascend-training.txt`。安全安装器会在
pip 前后锁定并检查：

- `torch==2.7.1`
- `torch_npu==2.7.1.post4`
- `torchvision==0.22.1`
- `torchaudio==2.7.1`

它还会安装并导入 H3 路径真正需要的 `ftfy`、`remote-pdb`、PyAV、Diffusers 和
PyArrow。上游包元数据声明的 Web UI、推理服务、CUDA kernel、Ray、NVIDIA 监控等可选
依赖不属于本训练路径，因此不要以完整项目的 `pip check` 作为验收标准。

安装后检查：

```bash
python - <<'PY'
import av, diffusers, ftfy, pyarrow, remote_pdb
import torch, torch_npu, torchaudio, torchvision
print("Ascend training imports: PASS")
print(torch.__version__, torch_npu.__version__)
PY
```

## 6. 检查 MiniMax-H3 权重

模型目录必须保持原始 Diffusers 组件布局，包括 `transformer/`、`vae/`、
`audio_vae/`、`text_encoder/`、`tokenizer/`、`processor/`、`scheduler/` 和
`audio_scheduler/`。ComfyUI 的扁平化重打包不能直接替代。

```bash
cd /workspace/FastVideo
python scripts/verify_minimax_h3_checkpoint.py /models/MiniMax-H3 \
  2>&1 | tee /tmp/minimax_h3_verify.log
```

检查器不会把全部权重加载进内存，而是验证目录、JSON、索引分片、空文件、Git LFS 指针和
safetensors header。继续之前应看到：

```text
MiniMax-H3 checkpoint verification: PASS
```

如果可以访问 ModelScope，并且其官方仓库仍提供相同的 Diffusers 布局，可以使用：

```bash
pip install modelscope
modelscope download --model MiniMax/MiniMax-H3 \
  --local_dir /models/MiniMax-H3
```

下载或补齐后必须重新执行完整性检查。

## 7. 准备一条有声视频数据

选择一条包含音轨、按 24 FPS 重采样后不少于 124 帧的 MP4，放入宿主机的训练媒体目录。
容器内执行：

```bash
cd /workspace/FastVideo

export MINIMAX_H3_MODEL_PATH=/models/MiniMax-H3
export TRAINING_VIDEO_PATH=/data/media/sample-with-audio.mp4
export TRAINING_CAPTION='Describe the visible action, scene, camera motion, speech, music, and environmental sounds precisely.'

bash examples/train/prepare_minimax_h3_ascend.sh \
  2>&1 | tee /tmp/minimax_h3_preprocess.log
```

成功标志：

```text
Validated one MiniMax H3 training row in ...
```

默认输出为：

```text
/workspace/FastVideo/data/crush-smol_h3_t2va_single_sample_preprocessed
```

## 8. Dense SFT 底座验收

先只构建配置、模型和数据加载器：

```bash
cd /workspace/FastVideo

NUM_NPUS=8 bash examples/train/run_ascend.sh \
  examples/train/configs/ascend/minimax_h3_t2va_sft_smoke.yaml \
  --dry-run
```

dry-run 的 `Training completed` 不代表发生了训练，只代表运行时构建成功。正式执行一个
optimizer step：

```bash
NUM_NPUS=8 bash examples/train/run_ascend.sh \
  examples/train/configs/ascend/minimax_h3_t2va_sft_smoke.yaml
```

成功标准是所有 rank 完成前向、反向和 optimizer step，且 loss 有限。HCCL 可用时，某些
可选 PyHCCL 动态库缺失的 warning 不等于 torch.distributed 的 HCCL 不可用。

## 9. Dense DMD2 四步训练

先检查四步配置及所有本地路径：

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

构建 student、teacher、critic，但不训练：

```bash
NUM_NPUS=8 bash examples/train/run_ascend.sh \
  examples/train/configs/ascend/minimax_h3_t2va_dmd2_4step_smoke.yaml \
  --dry-run
```

日志中应对三个角色分别出现一次：

```text
Loading transformer weights with CPU staging=True, FSDP CPU offload=True
```

正式执行一个 DMD2 optimizer step：

```bash
NUM_NPUS=8 bash examples/train/run_ascend.sh \
  examples/train/configs/ascend/minimax_h3_t2va_dmd2_4step_smoke.yaml
```

这一轮会执行 student rollout、teacher/critic score、student backward、critic backward、
两个 optimizer 更新和 DCP 保存。CPU offload 会显著降低速度并消耗大量主机内存，但不会
降低 BF16 计算精度。中断未完成的一步不能续训，应重新执行这一整步。

完成后检查：

```bash
test -f runs/ascend_minimax_h3_dense_dmd2_4step_smoke/checkpoint-1/metadata.json \
  && echo 'metadata: PASS'
test -d runs/ascend_minimax_h3_dense_dmd2_4step_smoke/checkpoint-1/dcp \
  && echo 'DCP checkpoint: PASS'
```

## 10. 导出 student

```bash
cd /workspace/FastVideo

bash examples/train/export_minimax_h3_dmd2_ascend.sh \
  runs/ascend_minimax_h3_dense_dmd2_4step_smoke/checkpoint-1 \
  runs/ascend_minimax_h3_dense_dmd2_4step_export \
  2>&1 | tee /tmp/minimax_h3_dmd2_export.log
```

成功标志：

```text
Strict reload verification passed.
Exported four-step student: ...
```

导出会以基础模型目录为模板。底层文件系统支持硬链接时，大部分未修改组件不会重复占用
空间；否则可能复制完整模型，导出前应检查磁盘余量。

## 11. 四次 DiT forward 推理

```bash
cd /workspace/FastVideo

python examples/inference/basic/basic_minimax_h3_dense_4step_ascend.py \
  --model-path runs/ascend_minimax_h3_dense_dmd2_4step_export \
  --prompt 'A cinematic ocean wave crashes against dark rocks, with synchronized roaring water and wind.' \
  --output outputs/minimax_h3_dense_dmd2_4step \
  2>&1 | tee /tmp/minimax_h3_dense_4step_inference.log
```

该入口强制使用：

- Dense `TORCH_SDPA`
- `guidance_scale=1.0`
- 5 个 sigma 网格点，即恰好 4 次 DiT forward
- 8 卡 sequence parallel
- 关闭 VSA、FA4、CUDA 专用优化和数值顺序可能变化的融合

单步 bring-up 的验收目标是成功导出、严格重载并生成有声 MP4，不是视频质量。要获得可用
质量仍需要公开且可验证的训练 recipe、数据规模和充分训练步数。

## 12. 停止和重新运行

训练可以用一次 `Ctrl+C` 中断；这不会修改基础模型、数据或源码。若中断发生在 checkpoint
写入阶段，可能留下不完整的 `checkpoint-1`，重新运行前应将其改名保存或删除。确认所有
rank 已退出：

```bash
pgrep -af 'fastvideo.train.entrypoint.train|torchrun'
```

## 13. 补丁结构

```text
patches/
├── fastvideo-ascend-910b-patch-20260904/
│   ├── payload/
│   │   ├── replacements/
│   │   └── additions/
│   ├── README.md
│   ├── install.sh
│   ├── verify.sh
│   ├── modified.tsv
│   └── added.tsv
├── fastvideo-ascend-910b-patch-20260904.tar.gz
└── fastvideo-ascend-910b-patch-20260904.tar.gz.sha256
```

更短的补丁使用说明见
[`patches/fastvideo-ascend-910b-patch-20260904/README.md`](patches/fastvideo-ascend-910b-patch-20260904/README.md)。

## License

补丁代码延续 FastVideo 的 Apache-2.0 许可证。FastVideo、MiniMax-H3、训练数据和第三方
组件的许可及权利归各自作者所有。
