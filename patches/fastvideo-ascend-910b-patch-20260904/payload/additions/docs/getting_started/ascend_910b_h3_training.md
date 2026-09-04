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

## 2. Checkpoint and one-sample data

Log in to Hugging Face if the model requires acceptance, then run:

```bash
hf auth login
bash examples/train/prepare_minimax_h3_ascend.sh
```

The preprocessing process loads the video VAE, audio VAE, and text encoder one
at a time on one NPU. It writes one synchronized T2VA parquet row under
`data/crush-smol_h3_t2va_single_sample_preprocessed`.

## 3. Acceptance ladder

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

## 4. What follows this smoke test

After dense SFT is numerically stable, extend the H3 model plugin to the public
DMD2 role contract (student, frozen teacher, trainable critic), then lock the
student rollout to `[999, 749, 500, 250]`. The exact H3 VSA backend comes after
dense four-step parity: its CUDA/Triton training kernel cannot be reused on NPU,
so Ascend needs a reference forward/backward implementation before any kernel
optimization. Recipe-specific loss weights and sampling policy remain gated on
the official H3 recipe.
