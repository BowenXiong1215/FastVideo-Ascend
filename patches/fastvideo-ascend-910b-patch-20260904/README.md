# FastVideo MiniMax-H3 昇腾 910B 训练补丁包（阶段 1）

此补丁包将指定版本的官方 FastVideo 源码转换为昇腾 910B MiniMax-H3
Dense T2VA 训练底座。当前阶段覆盖 HCCL、Dense SDPA、FSDP/HSDP、序列并行、
视频/音频/文本预处理、前后向、优化器单步和 NPU RNG 断点恢复。

本阶段不宣称已经复现 FastH3 四步 DMD2 或 VSA；它是进入四步蒸馏前的训练底座验收。

## 基线

```text
仓库：https://github.com/hao-ai-lab/FastVideo
Commit：7bb76b5ec99807a66aa3047b901f15019abe0f00
目标设备：Ascend 910B
参考镜像：quay.io/ascend/triton:3.2.1-cann9.0.0-torch_npu2.7.1.post4-910b-ubuntu22.04-py3.11
```

安装器处理 8 个官方源码文件，并加入 10 个昇腾环境、容器、配置、训练、权重检查和说明文件。
官方源码文件通过 `sed -i` 更新；新增文件从补丁载荷复制。

## 使用

下载固定版本官方源码：

```bash
wget -O FastVideo.tar.gz \
  https://github.com/hao-ai-lab/FastVideo/archive/7bb76b5ec99807a66aa3047b901f15019abe0f00.tar.gz

tar -xzf FastVideo.tar.gz
mv FastVideo-7bb76b5ec99807a66aa3047b901f15019abe0f00 FastVideo
```

解压补丁包并执行：

```bash
bash fastvideo-ascend-910b-patch-20260904/install.sh \
  /absolute/path/to/FastVideo
```

成功标志：

```text
FastVideo Ascend 910B stage-1 patch: PASS
```

独立校验：

```bash
bash fastvideo-ascend-910b-patch-20260904/verify.sh \
  /absolute/path/to/FastVideo
```

安装脚本可以重复执行；已经应用的文件会显示 `present`。如果目标目录安装过本补丁的
上一版，安装器会识别旧版文件哈希并原地升级，不需要重新下载完整 FastVideo。

## 容器与依赖

可直接使用补丁加入的 Dockerfile：

```bash
docker build -f docker/Dockerfile.ascend -t fastvideo-h3-ascend:stage1 .
```

如直接在现有容器内安装，只运行补丁提供的安全安装器：

```bash
bash scripts/install_ascend_dependencies.sh
```

该脚本在 pip 前后校验 `torch==2.7.1` 与 `torch_npu==2.7.1.post4`，并约束
`torchvision==0.22.1`、`torchaudio==2.7.1`，避免 pip 升级到 PyTorch 2.14。
安装完成后还会执行 FastVideo、PyAV、Diffusers、PyArrow、remote-pdb 等预处理关键模块
的导入检查。不要再单独执行无约束的 `pip install -r requirements/ascend-training.txt`。

## 本地权重检查

默认检查用户现有目录：

```bash
cd /hpc-to-ds-0115/x00876811/fast/FastVideo
python scripts/verify_minimax_h3_checkpoint.py \
  /hpc-to-ds-0115/x00876811/models/MiniMax-H3
```

该检查不会加载 498GB 权重到内存；它检查 FastVideo 所需的 Diffusers 目录、JSON、
safetensors 索引引用、空文件、Git LFS 指针和每个 safetensors header。只有在官方
ModelScope 仓库提供同样 Diffusers 布局时，才建议用以下命令补齐：

```bash
pip install modelscope
modelscope download --model MiniMax/MiniMax-H3 \
  --local_dir /hpc-to-ds-0115/x00876811/models/MiniMax-H3
```

不要下载 `Comfy-Org/MiniMax-H3` 来代替：它是面向 ComfyUI 的扁平化重打包，不是
FastVideo 当前 loader 使用的目录结构。

## 数据与训练验收

不需要下载特定 Hugging Face 数据集。先选择任意一条包含音轨、按 24 FPS 重采样后
不少于 124 帧的本地 MP4：

```bash
cd /hpc-to-ds-0115/x00876811/fast/FastVideo
export TRAINING_VIDEO_PATH=/absolute/path/to/sample-with-audio.mp4
export TRAINING_CAPTION='准确描述画面、动作、镜头运动、对白、音乐和环境声。'
bash examples/train/prepare_minimax_h3_ascend.sh

NUM_NPUS=8 bash examples/train/run_ascend.sh \
  examples/train/configs/ascend/minimax_h3_t2va_sft_smoke.yaml --dry-run

NUM_NPUS=8 bash examples/train/run_ascend.sh \
  examples/train/configs/ascend/minimax_h3_t2va_sft_smoke.yaml
```

完整说明见：

```text
docs/getting_started/ascend_910b_h3_training.md
```

当前代码已通过补丁安装/重复安装、Shell 语法、Python 编译和配置约束检查。
真实 HCCL/FSDP 前后向仍需在 910B 实机上完成首轮验收。
