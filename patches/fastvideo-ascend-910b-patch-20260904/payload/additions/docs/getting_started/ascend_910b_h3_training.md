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
pip install -r requirements/ascend-training.txt
pip install -e . --no-deps
python -c 'import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())'
```

## 2. Checkpoint integrity

The Ascend preparation path does not require Hugging Face access. By default it
uses `/hpc-to-ds-0115/x00876811/models/MiniMax-H3`. Verify the checkpoint before
spending NPU time:

```bash
python scripts/verify_minimax_h3_checkpoint.py \
  /hpc-to-ds-0115/x00876811/models/MiniMax-H3
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
  --local_dir /hpc-to-ds-0115/x00876811/models/MiniMax-H3
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

## 5. What follows this smoke test

After dense SFT is numerically stable, extend the H3 model plugin to the public
DMD2 role contract (student, frozen teacher, trainable critic), then lock the
student rollout to `[999, 749, 500, 250]`. The exact H3 VSA backend comes after
dense four-step parity: its CUDA/Triton training kernel cannot be reused on NPU,
so Ascend needs a reference forward/backward implementation before any kernel
optimization. Recipe-specific loss weights and sampling policy remain gated on
the official H3 recipe.
