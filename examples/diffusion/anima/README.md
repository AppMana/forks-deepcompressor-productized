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

If the GPUs have different sustained throughput, give faster ranks more
prompts so the final barrier does not wait on an equal but slower shard. For
example, a 42/58 split keeps a desktop-contended GPU 0 and an idle GPU 1 busy
for approximately the same wall time:

```bash
accelerate launch --multi_gpu --num_processes 2 --gpu_ids 0,1 \
  --module deepcompressor.app.diffusion.anima.cli collect \
  --num-prompts 10000 --rank-weights 0.42,0.58 --resume \
  --output runs/anima-aesthetic-v1.1/calibration-10000prompts
```

`--resume` verifies the latent and every expected timestep/guidance cache for
each completed prompt, reconstructs its manifest rows without loading tensor
payloads, and regenerates only incomplete prompts. This makes multi-hour local
and Kueue collections restartable without changing seeds or calibration
selection.

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
  --num-iters 1 \
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

For a controlled randomized-SVD ladder, hold rank and validation prompts
fixed while changing calibration coverage and residual passes:

```bash
# Calibration coverage control
deepcompressor-svdquant quantize \
  --dataset runs/anima-aesthetic-v1.1/calibration-100prompts/hf_dataset \
  --num-samples 100 --rank 128 --num-iters 1 --fast \
  --run-name rank128-randomized-1iter-100prompts \
  --output runs/anima-aesthetic-v1.1/rank128-randomized-1iter-100prompts

# Residual-refinement experiment
deepcompressor-svdquant quantize \
  --dataset runs/anima-aesthetic-v1.1/calibration-100prompts/hf_dataset \
  --num-samples 100 --rank 128 --num-iters 4 --fast \
  --run-name rank128-randomized-4iter-100prompts \
  --output runs/anima-aesthetic-v1.1/rank128-randomized-4iter-100prompts
```

`--fast` selects manual one-candidate smoothing and randomized truncated SVD;
it does not override `--num-iters`. Early stopping can still accept fewer
residual passes when another pass does not reduce the module-output objective.

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

## AppMana MLflow tracking

Quantization and validation track to the deployed
`https://mlflow.appmana.com` server by default, in the
`anima-aesthetic-v1.1-svdquant` experiment. The quantization run records the
complete recipe, GPU/software versions, per-phase wall times, and packed
checkpoint size. The large checkpoint stays on the shared filesystem; MLflow
receives its manifest, recipe, and log. Validation automatically discovers
`mlflow-run.json` beside the checkpoint and resumes the same run, adding raw
pixel metrics and the paired BF16/INT4 PNG artifacts.

The cluster's existing `mlflow-auth-admin-secret` supplies HTTP Basic auth.
Use the same secret references in a Kueue JobSet container:

```yaml
env:
  - name: MLFLOW_TRACKING_URI
    value: https://mlflow.appmana.com
  - name: MLFLOW_TRACKING_USERNAME
    valueFrom:
      secretKeyRef:
        name: mlflow-auth-admin-secret
        key: username
  - name: MLFLOW_TRACKING_PASSWORD
    valueFrom:
      secretKeyRef:
        name: mlflow-auth-admin-secret
        key: password
  - name: AWS_ACCESS_KEY_ID
    valueFrom:
      secretKeyRef:
        name: mlflow-s3-user
        key: ACCESS_KEY_ID
  - name: AWS_SECRET_ACCESS_KEY
    valueFrom:
      secretKeyRef:
        name: mlflow-s3-user
        key: ACCESS_SECRET_KEY
  - name: MLFLOW_S3_ENDPOINT_URL
    value: http://seaweedfs-s3.seaweedfs.svc.cluster.local:8333
```

For an authenticated invocation from a workstation with cluster access:

```bash
export MLFLOW_TRACKING_USERNAME="$(kubectl get secret mlflow-auth-admin-secret \
  -n appmana -o jsonpath='{.data.username}' | base64 --decode)"
export MLFLOW_TRACKING_PASSWORD="$(kubectl get secret mlflow-auth-admin-secret \
  -n appmana -o jsonpath='{.data.password}' | base64 --decode)"
export AWS_ACCESS_KEY_ID="$(kubectl get secret mlflow-s3-user \
  -n appmana -o jsonpath='{.data.ACCESS_KEY_ID}' | base64 --decode)"
export AWS_SECRET_ACCESS_KEY="$(kubectl get secret mlflow-s3-user \
  -n appmana -o jsonpath='{.data.ACCESS_SECRET_KEY}' | base64 --decode)"
export MLFLOW_S3_ENDPOINT_URL=https://s3-lfs.appmana.com
```

List tracked recipes in descending minimum-pixel-similarity order:

```bash
deepcompressor-svdquant compare
```

Use `--no-track` for an intentionally offline smoke test. No credential is
ever written to a run directory or MLflow artifact.

## Pixel validation

```bash
deepcompressor-svdquant validate \
  --gpu 0 \
  --manifest runs/anima-aesthetic-v1.1/rank32/nunchaku/anima-aesthetic-v1.1-svdquant-int4.json \
  --num-prompts 100 \
  --threshold 0.99 \
  --output runs/anima-aesthetic-v1.1/rank32/validation
```

The command exits with status 2 if any prompt misses the threshold.

### Published Qwen-Image baseline

Use the staged Qwen command to measure what raw-pixel agreement the published
Nunchaku rank-32 checkpoint actually achieves. Prompt conditioning and initial
noise are saved once, both variants return latents, and a separately loaded
shared BF16 VAE decodes both sets. This prevents text-encoder, random-number,
or VAE differences from being attributed to INT4.

```bash
deepcompressor-qwen-baseline encode \
  --prompt-index 100 --prompt-index 2500 \
  --prompt-index 5000 --prompt-index 7500 \
  --gpu 1 --output runs/qwen-image-r32-baseline

deepcompressor-qwen-baseline render-bf16 \
  --output runs/qwen-image-r32-baseline \
  --width 512 --height 512 --steps 30 --cfg 4 \
  --gpu0-memory 13GiB --gpu1-memory 21GiB --cpu-memory 1GiB

deepcompressor-qwen-baseline render-int4 \
  --output runs/qwen-image-r32-baseline --gpu 1 \
  --width 512 --height 512 --steps 30 --cfg 4

deepcompressor-qwen-baseline decode \
  --output runs/qwen-image-r32-baseline --gpu 1

deepcompressor-qwen-baseline report \
  --output runs/qwen-image-r32-baseline --threshold 0.99
```

The BF16 memory limits are machine-specific. Modules that exceed both GPUs and
the CPU allowance are disk-offloaded under the run directory and loaded onto a
GPU for computation. The report records the complete device map, so this
constrained-system timing is never confused with a single-GPU kernel benchmark.

Use prompts outside the calibration range for acceptance. For example, if
prompts 0--99 supplied PTQ, validate on 100--199:

```bash
deepcompressor-svdquant validate \
  --manifest runs/anima-aesthetic-v1.1/rank128/nunchaku/anima-aesthetic-v1.1-svdquant-int4.json \
  --prompt-offset 100 --num-prompts 100 --steps 30 --threshold 0.99 \
  --output runs/anima-aesthetic-v1.1/rank128/validation-held-out
```

## Performance validation

Image-level timing includes text encoding, sampler orchestration, VAE decode,
and file I/O shared by BF16 and INT4. Measure the denoiser separately with
cached timestep records to determine whether the fused projection path is
actually fast:

```bash
deepcompressor-svdquant benchmark \
  --manifest runs/anima-aesthetic-v1.1/rank128/nunchaku/anima-aesthetic-v1.1-svdquant-int4.json \
  --dataset runs/anima-aesthetic-v1.1/calibration-100prompts/hf_dataset \
  --num-samples 16 --warmup 2 --iterations 10 \
  --output runs/anima-aesthetic-v1.1/rank128/benchmark
```

MLflow records `performance.denoiser.int4_vs_bf16_speedup` independently from
`performance.image.int4_vs_bf16_speedup`. A lower result is a performance
failure to profile; it is not hidden by reporting only linear-kernel timings.
