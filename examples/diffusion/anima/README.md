# Anima Aesthetic 1.1 SVDQuant

This workflow creates a Nunchaku-compatible INT4 checkpoint from the native
ComfyUI Anima Aesthetic 1.1 checkpoint. It is post-training quantization: it
does not train a LoRA, run backpropagation, or create optimizer state.

## What the pipeline does

1. Native ComfyUI samples each prompt and captures every denoiser invocation.
   With classifier-free guidance and 30 steps this is 60 records per prompt.
   The final latent is also retained.
2. One timestep/guidance record per prompt is selected in a deterministic,
   stratified way for PTQ. All trajectory records remain available for audit
   and future recipes.
3. DeepCompressor applies the published GELU shift and smoothing transforms.
   The production recipe searches 20 smoothing candidates using real module
   output error.
4. For each supported projection group, DeepCompressor computes a truncated
   BF16 low-rank branch with SVD, quantizes the remaining weight to signed
   group-64 INT4, quantizes inputs dynamically to INT4, and chooses iterative
   residual refinements by module-output error. The rank is 32 or 128.
5. The result is packed into Nunchaku's fused W4A4 + W16A16 layout. Nunchaku is
   the deployment backend; the PTQ search itself uses mutable PyTorch modules
   because every candidate has different scales and residual weights.
6. Validation runs identical seeds through BF16 and Nunchaku INT4 models,
   decodes both with the same VAE, and measures raw RGB pixel error. Acceptance
   requires the minimum `1 - RGB_RMSE` to be at least 0.99 by default.

## Install

```bash
cd ~/Documents/forks-deepcompressor-productized
source .venv/bin/activate
uv sync --dev
```

ComfyUI and Nunchaku are resolved from `https://nodes.appmana.com/simple` by
the checked-in `pyproject.toml` and `uv.lock`.

## Parallel calibration collection

The prompt/index table is a Hugging Face `Dataset`. Accelerate divides it
without padding, one disjoint slice per process. Tensor payloads remain
individual `.pt` files so a multi-gigabyte trajectory is never copied into an
Arrow cell. Rank 0 writes a portable `hf_dataset` Arrow manifest containing
paths, seeds, timesteps, guidance indexes, and the PTQ-selection flag.

Two local GPUs:

```bash
accelerate launch --multi_gpu --num_processes 2 \
  --module deepcompressor.app.diffusion.anima.cli collect \
  --num-prompts 100 \
  --output runs/anima-aesthetic-v1.1/dataset
```

The same command works under a multi-node Accelerate launch in a Kueue
JobSet. Every process must see the same output filesystem. Separate rank-32,
rank-128, or recipe experiments are independent jobs and can occupy separate
GPUs or JobSets.

## PTQ

First prove the complete path using one calibration record, one manual
smoothing candidate, and one residual SVD:

```bash
deepcompressor-svdquant quantize \
  --gpu 0 \
  --dataset runs/anima-aesthetic-v1.1/dataset/hf_dataset \
  --num-samples 1 \
  --rank 32 \
  --fast \
  --resume \
  --output runs/anima-aesthetic-v1.1/smoke-rank32
```

Then run the production 100-prompt, 100-iteration recipe:

```bash
deepcompressor-svdquant quantize \
  --gpu 0 \
  --dataset runs/anima-aesthetic-v1.1/dataset/hf_dataset \
  --num-samples 100 \
  --rank 32 \
  --num-iters 100 \
  --output runs/anima-aesthetic-v1.1/rank32
```

The PTQ of one checkpoint is intentionally single-process today. Blocks are
consumed in order because DeepCompressor propagates the effective output of a
processed block into the next block's cache. Ordinary Accelerate data
parallelism would independently choose candidates on partial calibration
sets, which changes the objective and is therefore incorrect. Collection,
validation prompts, ranks, and recipe sweeps are the safe coarse-grained
parallel axes.

The exact production SVD uses an economy decomposition
(`full_matrices=False`). This preserves the leading singular vectors used by
SVDQuant while avoiding unused square matrices for tall and wide projections.
The smoke recipe uses deterministic randomized truncated SVD with eight
oversampling vectors and two power iterations; it proves integration quickly
but is not the production checkpoint recipe. `--resume` reuses a completed
smoothing cache after an interrupted run.

## Pixel validation

```bash
deepcompressor-svdquant validate \
  --gpu 0 \
  --manifest runs/anima-aesthetic-v1.1/rank32/nunchaku/manifest.json \
  --num-prompts 100 \
  --threshold 0.99 \
  --output runs/anima-aesthetic-v1.1/rank32/validation
```

The command exits with status 2 if any prompt misses the threshold.
