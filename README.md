# FastVideo-Ascend

FastVideo MiniMax-H3 training adaptation for Huawei Ascend 910B.

本仓库采用轻量补丁包形式发布，不镜像完整 FastVideo 源码，也不包含模型权重或训练数据。
补丁结构与 `DiffSynth-Studio-Ascend` 的发布方式保持一致：

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

## 当前状态

阶段 1 建立 MiniMax-H3 Dense T2VA 训练底座，包括：

- Ascend NPU 与 HCCL 启动；
- BF16 Dense SDPA；
- FSDP/HSDP 与 sequence parallel 通信；
- H3 权重 CPU 逐张量暂存和 FSDP 流式分片，避免加载阶段 NPU 峰值 OOM；
- H3 视频、双声道音频和文本联合数据预处理；
- 本地 Diffusers 权重目录、索引分片、JSON、LFS 指针和 safetensors header 完整性检查；
- 不访问 Hugging Face，直接使用本地有声 MP4 和 caption；
- 带 PyTorch ABI 约束和安装前后版本保护的依赖安装器；
- 前向、反向与 optimizer step；
- NPU RNG checkpoint 保存和恢复；
- 8×Ascend 910B 单步训练验收配置。

当前阶段不宣称已经复现 FastH3 四步 DMD2 或 VSA。Dense 单步训练通过后，下一阶段将接入
H3 联合视频/音频 DMD2，并固定 student rollout 为 `[999, 749, 500, 250]`。

## 上游基线

```text
Repository: https://github.com/hao-ai-lab/FastVideo
Commit:     7bb76b5ec99807a66aa3047b901f15019abe0f00
```

## 推荐环境

```text
Image:     quay.io/ascend/triton:3.2.1-cann9.0.0-torch_npu2.7.1.post4-910b-ubuntu22.04-py3.11
CANN:      9.0.0
PyTorch:   2.7.1
torch_npu: 2.7.1.post4
Device:    Ascend 910B
```

## 快速开始

下载固定版本 FastVideo：

```bash
wget -O FastVideo.tar.gz \
  https://github.com/hao-ai-lab/FastVideo/archive/7bb76b5ec99807a66aa3047b901f15019abe0f00.tar.gz
tar -xzf FastVideo.tar.gz
mv FastVideo-7bb76b5ec99807a66aa3047b901f15019abe0f00 FastVideo
```

下载并应用补丁：

```bash
tar -xzf fastvideo-ascend-910b-patch-20260904.tar.gz
bash fastvideo-ascend-910b-patch-20260904/install.sh /absolute/path/to/FastVideo
```

成功输出：

```text
FastVideo Ascend 910B stage-1 patch: PASS
```

详细环境、数据和训练命令见补丁目录内的 [README](patches/fastvideo-ascend-910b-patch-20260904/README.md)。

## 验证说明

补丁已完成：

- 固定上游源码 SHA256 校验；
- 补丁 payload SHA256 校验；
- 首次安装与重复安装；
- 独立 `verify.sh` 校验；
- Shell 语法检查；
- Python 编译检查；
- H3 训练配置约束检查。

真实 HCCL/FSDP 前后向仍需在 Ascend 910B 实机上完成首轮验证。

## License

补丁代码延续 FastVideo 的 Apache-2.0 许可证。FastVideo 及其模型、数据和第三方组件的权利
归各自作者所有。
