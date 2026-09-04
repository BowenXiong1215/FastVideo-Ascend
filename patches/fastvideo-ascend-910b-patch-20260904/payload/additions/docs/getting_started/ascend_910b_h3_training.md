# MiniMax-H3 training on Ascend 910B

This port starts with the public, reproducible boundary: exact H3 joint
video/audio packing, dense BF16 SDPA, HCCL sequence parallelism, FSDP/HSDP,
forward/backward, optimizer step, and checkpoint RNG state. It does not claim
that the unpublished FastH3 H3-DMD2 recipe has been reproduced.

## 1. Environment

Use a CANN release and matching `torch`/`torch_npu` pair supplied for the 910B.
Do not let FastVideo's normal Linux dependencies replace that PyTorch build:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/install_ascend_dependencies.sh
python -c 'import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())'
```

Do not install `requirements/ascend-training.txt` without the bundled
constraints. The installer verifies `torch==2.7.1` and
`torch_npu==2.7.1.post4` before and after pip, constrains torchvision to the
matching `0.22.1`, installs FastVideo with `--no-deps`, and imports the actual
H3 preprocessing and training entrypoints as its acceptance test.

Do not use the full-project `pip check` result as the acceptance test for this
focused environment. Upstream package metadata also describes optional Web UI,
serving, CUDA kernel, NVIDIA monitoring, Ray/torchcodec preprocessing, and
experiment-tracking paths. Packages such as `aiofiles`, `fastapi`, `gradio`,
`ray`, `torchcodec`, and `fastvideo-kernel` are intentionally absent here. The
checked-in smoke configuration selects the `none` tracker and does not require
W&B. In contrast, `remote-pdb` and `ftfy` are included because FastVideo imports
them eagerly, while torchvision and torchaudio are both required by the H3
media path.

## 2. Checkpoint integrity

The Ascend preparation path does not require Hugging Face access. By default it
uses `/models/MiniMax-H3`. Verify the checkpoint before
spending NPU time:

```bash
python scripts/verify_minimax_h3_checkpoint.py \
  /models/MiniMax-H3
```

The verifier checks the Diffusers component layout, all JSON files, all shards
named by safetensors indexes, empty files, unresolved Git LFS pointers, and
safetensors headers. FastVideo requires `transformer/`, `vae/`, `audio_vae/`,
`text_encoder/`, `tokenizer/`, `processor/`, `scheduler/`, and
`audio_scheduler/`. A flattened ComfyUI checkpoint is not interchangeable.

If the official ModelScope repository exposes that same layout, missing files
can be resumed directly into the final directory:

```bash
pip install modelscope
modelscope download --model MiniMax/MiniMax-H3 \
  --local_dir /models/MiniMax-H3
```

Run the verifier again after a resumed download. If that repository is not yet
populated or differs in layout, manually copy the complete original
`MiniMaxAI/MiniMax-H3` snapshot from a connected machine without renaming or
flattening its files.

## 3. One-sample data

The public Crush-Smol sample is optional for the first optimizer-step run. Any
local MP4 containing a real audio track and at least 124 frames after 24 FPS
resampling is sufficient. The preprocessor retains the exact H3 resize/crop,
32 kHz audio, VAE, and text-encoder path:

```bash
export TRAINING_VIDEO_PATH=/absolute/path/to/sample-with-audio.mp4
export TRAINING_CAPTION='Describe the visible action, scene, camera motion, speech, music, and other sounds precisely.'
bash examples/train/prepare_minimax_h3_ascend.sh
```

To use the pinned Crush-Smol fixture instead, manually place these two files
and run the same script without those environment variables:

```text
data/crush-smol/videos2caption.json
data/crush-smol/videos/1gGQy4nxyUo-Scene-016.mp4
```

The script loads the video VAE, audio VAE, and text encoder one at a time on one
NPU, then writes a synchronized T2VA Parquet row under
`data/crush-smol_h3_t2va_single_sample_preprocessed`.

For non-default paths, set `MINIMAX_H3_MODEL_PATH`, `MINIMAX_H3_DATA_DIR`, or
`MINIMAX_H3_OUTPUT_DIR` before running the preparation script.

## 4. Acceptance ladder

First validate configuration and model construction:

```bash
NUM_NPUS=8 bash examples/train/run_ascend.sh \
  examples/train/configs/ascend/minimax_h3_t2va_sft_smoke.yaml --dry-run
```

Then run one real optimizer step:

```bash
NUM_NPUS=8 bash examples/train/run_ascend.sh \
  examples/train/configs/ascend/minimax_h3_t2va_sft_smoke.yaml
```

The checked-in shape is the real 124-frame 768x1344 H3 contract. Eight 64-GB
910B devices are the initial target; this is an engineering starting point,
not a proven minimum. If the job is multi-node, set `NUM_NPUS`, `NNODES`,
`NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` on every node.

Success means all ranks pass model load and HCCL warmup, the logged loss is
finite, backward completes, and one optimizer step completes. Keep validation
off until this passes, because it loads additional inference components.

## 5. Dense DMD2 four-step bring-up

After dense SFT is numerically stable, run the engineering bring-up recipe. It
implements the public DMD2 student/teacher/critic structure over H3's joint
video/audio latent pair, uses each modality's scheduler shift, and locks the
student ladder to `[999, 749, 500, 250]`:

```bash
NUM_NPUS=8 bash examples/train/run_ascend.sh \
  examples/train/configs/ascend/minimax_h3_t2va_dmd2_4step_smoke.yaml
```

All three 33B roles use FSDP CPU offload. This trades training speed and host
memory for NPU capacity without changing BF16 computation. The one-step smoke
is intended to validate finite student and critic losses, both backwards,
optimizer updates, and DCP output—not model quality.

Export the resulting student checkpoint:

```bash
bash examples/train/export_minimax_h3_dmd2_ascend.sh
```

Run the exported model with exactly five sigma-grid points, hence four DiT
forwards, using dense SDPA on 8 NPUs:

```bash
python examples/inference/basic/basic_minimax_h3_dense_4step_ascend.py \
  --model-path runs/ascend_minimax_h3_dense_dmd2_4step_export \
  --prompt 'A cinematic scene with synchronized environmental sound.'
```

This is explicitly a mechanics recipe. Recipe-specific loss weights, prompt
distribution, score sampling, and convergence length remain gated on the
official FastH3 training release. VSA comes only after dense four-step parity.
