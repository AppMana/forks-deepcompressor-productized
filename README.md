# AppMana Anima SVDQuant

This repository productizes DeepCompressor for the native ComfyUI implementation of
[Anima Aesthetic 1.1](https://huggingface.co/circlestone-labs/Anima). It can collect deterministic
calibration trajectories, run the released SVDQuant post-training quantization (PTQ) algorithm, pack a
Nunchaku-compatible fused INT4 W4A4 + BF16 W16A16 checkpoint, and compare its decoded pixels and speed with the
original BF16 model.

The current implementation is a PTQ pipeline. It does **not** train its low-rank branch with backpropagation. Its default
activation-aware solver replaces weight-only residual SVD with reduced-rank regression over real W4A4 calibration
activations, while preserving DeepCompressor's block order, `OutputsError` selection, Nunchaku checkpoint layout, and
inference kernels. The released weight-only SVD remains available as a control.

## Repository map

- [10,000 Anima calibration prompts](examples/diffusion/prompts/anima-aesthetic-v1.1-calibration-10000.yaml) — the
  immutable prompt corpus used by the current collection run.
- [100 prompt categories](examples/diffusion/prompts/anima-aesthetic-v1.1-calibration-categories.yaml) — 100 prompts
  belong to each category.
- [Inspiration ledger](examples/diffusion/prompts/anima-aesthetic-v1.1-inspiration-ledger.md) and
  [visual-artist ledger](examples/diffusion/prompts/anima-aesthetic-v1.1-visual-artist-ledger.md) — diversity references
  used while authoring the corpus.
- [Prompt audit](deepcompressor/app/diffusion/anima/prompt_audit.py) — validates IDs, category balance, duplicates,
  repeated openings, forbidden score tags, and word-trigram overlap without rewriting prompts.
- [Typer workflow](deepcompressor/app/diffusion/anima/cli.py) — `collect`, `quantize`, `export`, `validate`, `benchmark`,
  and `compare`.
- [Canonical Anima recipe](deepcompressor/app/diffusion/anima/config.py) — the exact W4A4, smoothing, rank, SVD, and
  calibration settings.
- [Native ComfyUI sampler and cache collection](deepcompressor/app/diffusion/anima/pipeline.py).
- [Anima model structure](deepcompressor/app/diffusion/anima/struct.py) — maps ComfyUI Anima blocks into
  DeepCompressor structures.
- [Nunchaku packer and runtime patch](deepcompressor/app/diffusion/anima/nunchaku.py) — converts DeepCompressor weights,
  scales, smoothing factors, and low-rank factors to `SVDQW4A4Linear` groups.
- [MLflow integration](deepcompressor/app/diffusion/anima/tracking.py).
- [Published Qwen-Image baseline](deepcompressor/app/diffusion/anima/qwen_baseline.py) — controls prompt embeddings,
  initial noise, and VAE decoding while comparing official BF16 and published Nunchaku rank-32 outputs.
- [Detailed Anima operator notes](examples/diffusion/anima/README.md).
- [Console entrypoints](pyproject.toml#L53-L56).

## What has been implemented

The AppMana fork currently provides:

1. A `uv` project for Python 3.10–3.12. ComfyUI and Nunchaku resolve from
   `https://nodes.appmana.com/simple`; the lock file pins the complete environment.
2. Native ComfyUI Anima Aesthetic 1.1 model loading, text conditioning, sampling, and structured denoiser-input
   collection. This does not depend on an old Diffusers Anima implementation.
3. A manually authored corpus of 10,000 distinct prompts: 100 categories with 100 prompts each. The audit passes with
   no duplicates or threshold violations; its highest normalized word-trigram Jaccard similarity is
   `0.095238 < 0.10`.
4. Accelerate collection across local GPUs, optional throughput-weighted partitions, deterministic prompt seeds, and
   strict restart validation with `--resume`.
5. A DeepCompressor model structure and paper-faithful signed INT4 group-64 recipe with rank-32 or rank-128 BF16
   low-rank branches.
6. Economy exact SVD for production, deterministic randomized truncated SVD for smoke experiments, and activation-aware
   reduced-rank regression that includes both W4 weight error and A4 activation error.
7. Packing into the installed Nunchaku 1.3 checkpoint field names and replacement of native Anima attention/MLP
   projections with fused `SVDQW4A4Linear` operations.
8. Raw RGB acceptance testing using `1 - RGB_RMSE`, with a required minimum of `0.99`, rather than DINO or another
   feature-space proxy.
9. Separate steady-state denoiser benchmarking and end-to-end image timing.
10. MLflow experiment tracking for recipes, commits, environment details, timings, checkpoint manifests, raw-pixel
    metrics, comparison images, and speedups.

The following is **not implemented yet**:

- An optional gradient-based `QuantLowRankCalibrator` that differentiates only the W16A16 branch after activation-aware
  initialization.
- Gradient synchronization for low-rank factors across Accelerate/Kueue workers.
- Progressive candidate re-ranking over a large PTQ calibration set.
- A distributed PTQ error reduction inside DeepCompressor.
- An independent Kueue-shard manifest merger; the current multi-process collector expects one shared output filesystem.

## Install

Work in the checked-out repository and activate its existing environment:

```bash
cd ~/Documents/forks-deepcompressor-productized
source .venv/bin/activate
uv sync --dev
```

The CLI automatically discovers these files when their Hugging Face snapshots are already cached:

- `circlestone-labs/Anima`: `anima-aesthetic-v1.1.safetensors`
- `circlestone-labs/Anima`: `qwen_3_06b_base.safetensors`
- `Comfy-Org/Qwen-Image_ComfyUI`: `qwen_image_vae.safetensors` for validation

Pass `--model`, `--text-encoder`, or `--vae` explicitly when they are stored elsewhere. Inspect every available command
with:

```bash
deepcompressor-svdquant --help
deepcompressor-svdquant collect --help
deepcompressor-svdquant quantize --help
deepcompressor-svdquant validate --help
```

## Audit the prompts

```bash
deepcompressor-audit-prompts \
  examples/diffusion/prompts/anima-aesthetic-v1.1-calibration-10000.yaml \
  --expected-count 10000 \
  --prompts-per-category 100 \
  --max-similarity 0.10
```

This command is read-only. It never generates, combines, or edits prompt text.

## Collect calibration trajectories

Each prompt is seeded from its stable ID. A 30-step classifier-free-guidance run produces:

- one final latent;
- 60 denoiser input records: 30 timesteps × two guidance branches;
- one deterministic, stratified timestep/guidance record selected for PTQ or QAT;
- an Arrow manifest containing paths and metadata, while tensor payloads remain separate `.pt` files.

The full corpus therefore produces 10,000 latents, 600,000 cache records, and 10,000 selected hard links. Current disk
growth projects to approximately 1.4 TB.

The active two-GPU recipe assigns fewer prompts to the desktop-contended first GPU:

```bash
accelerate launch --multi_gpu --num_processes 2 --gpu_ids 0,1 \
  --mixed_precision bf16 --num_cpu_threads_per_process 8 \
  --module deepcompressor.app.diffusion.anima.cli collect \
  --prompts examples/diffusion/prompts/anima-aesthetic-v1.1-calibration-10000.yaml \
  --num-prompts 10000 --prompt-offset 0 \
  --rank-weights 0.42,0.58 --resume \
  --output runs/anima-aesthetic-v1.1/aesthetic-v1.1-calibration-10000prompts \
  --width 512 --height 512 --steps 30 --cfg 4.0 \
  --sampler er_sde --scheduler simple
```

Collection is data-parallel across prompts. Denoising steps inside one prompt are sequential. `--resume` accepts a
prompt only when its latent and every expected timestep/guidance cache exist, then reconstructs its manifest entries
without loading tensor payloads.

For Kueue, the efficient collection topology is one single-GPU process per prompt range. Shared-output Accelerate works
today. Independent JobSet output shards are preferable for fault isolation, but require the not-yet-implemented manifest
merge command. TP or PP does not help Anima collection while one replica fits on one GPU; use independent data replicas.

## What Muyang's released SVDQuant actually does

[SVDQuant](https://arxiv.org/abs/2411.05007) is post-training quantization, not QAT. For a weight matrix `W`, it builds a
high-precision low-rank branch and a quantized main branch:

```text
y = W4A4(x) + B(A(x))
```

`A` has shape `[rank, in_features]`, `B` has shape `[out_features, rank]`, and Nunchaku stores them as
`proj_down`/`proj_up`. The released algorithm initializes these factors directly with SVD of a weight/quantization
residual. The SVD implementation is under `torch.no_grad()`; it does not train a LoRA or create optimizer state.

DeepCompressor then scores a bounded set of local candidates by module-output error:

```text
E(candidate) = sum_i ||module_candidate(x_i) - module_BF16(x_i)||²
```

The calibration examples choose between candidates. They do not update `A` and `B`.

### Why the released search is tractable

The upstream recipe does **not** evaluate 139 candidates on 10,000 complete image generations:

- It collects 128 COCO-caption calibration prompts; the upstream fast recipe uses 64.
- It caches denoiser inputs once and calibrates local modules/blocks rather than rerunning text encoding, the sampler,
  and VAE decoding for every candidate.
- It computes each BF16 reference output once, stores it on CPU, and reuses it while candidates are scored.
- It obtains low-rank factors by direct SVD. It is not searching an arbitrary rank-32 parameter space.
- Its 100 residual-SVD iterations are a maximum and can stop early when the local output error stops improving.
- The paper's 5,000-image MJHQ measurements are evaluation, not calibration.
- The upstream evaluation path distributes image generation across eight GPUs; that is separate from PTQ.

The current exact Anima recipe has these candidate bounds per calibrated projection:

| Stage | Candidates | Meaning |
|---|---:|---|
| Smoothing | 39 | With 20 grid divisions and `beta=-2`: baseline + 19 activation-only pairs + 19 complementary pairs |
| Iterative residual SVD | At most 100 | One deterministic candidate per iteration; early stopping is enabled |
| Weight range | 1 | Manual `ratio=1.0` |
| Rank | 32 or 128 | Separate checkpoint experiments, not candidates inside one run |

The fast Anima recipe uses one manual smoothing candidate and randomized SVD. `--num-iters` still controls residual
passes.

The product CLI defaults to `--activation-aware`. Each residual iteration solves one activation-aware candidate,
requantizes `W - BA`, and lets the unchanged DeepCompressor `OutputsError` accept or reject it. Use `--weight-svd` to run
the released candidate generator unchanged.

Ten thousand examples do not imply ten thousand candidates. More examples reduce sampling error in candidate selection;
they do not expand the candidate family or guarantee lower calibration error. The dataset loader now opens selected
cache records lazily, while DeepCompressor retains only the activation cache required for the block currently being
calibrated.

## Run the implemented PTQ baseline

First prove the complete pack/load path:

```bash
deepcompressor-svdquant quantize \
  --gpu 0 \
  --dataset runs/anima-aesthetic-v1.1/aesthetic-v1.1-calibration-10000prompts/hf_dataset \
  --num-samples 1 --rank 32 --num-iters 1 --fast --resume \
  --run-name anima-r32-smoke \
  --output runs/anima-aesthetic-v1.1/anima-r32-smoke
```

Use the released calibration scale for the PTQ control, rather than trying to load all 10,000 records:

```bash
deepcompressor-svdquant quantize \
  --gpu 0 \
  --dataset runs/anima-aesthetic-v1.1/aesthetic-v1.1-calibration-10000prompts/hf_dataset \
  --num-samples 128 --rank 32 --num-iters 100 --weight-svd --resume \
  --run-name anima-r32-exact-128samples \
  --output runs/anima-aesthetic-v1.1/anima-r32-exact-128samples
```

`--resume` reuses completed DeepCompressor caches in the output directory. It is not a general mid-candidate distributed
checkpoint.

## Activation-aware W16A16 solver

The implemented alternative changes only how DeepCompressor determines `LowRankBranch.a.weight` and
`LowRankBranch.b.weight`. For original activation rows `X`, dynamically quantized rows `Q(X)`, original weight `W`, and
the current quantized residual weight `Q(W)`, it accumulates:

```text
C = E[X.T @ X]
K = E[Q(X).T @ X]
H = W @ C - Q(W) @ K
```

It then solves the regularized reduced-rank regression:

```text
minimize over rank(L) <= r:
  E[||W X - Q(W) Q(X) - L X||²] + damping * ||L||²
```

If `C + damping*I = chol @ chol.T`, the whitened unconstrained correction is `H @ chol^-T`. A rank-32 or rank-128 SVD of
that matrix, transformed back by `chol^-1`, gives `L = B @ A`. The factors have the same shapes and meaning as released
SVDQuant and are exported through the unchanged Nunchaku `proj_down`/`proj_up` fields.

The covariance and original/quantized cross-covariance are computed once per projection group. By default, 64 uniformly
spaced activation rows are taken from each cached tensor, so every calibration batch contributes without materializing
all spatial tokens. `--activation-num-tokens -1` uses every row. The dataset itself now loads selected records lazily;
DeepCompressor still materializes the current block's activation cache because its blockwise calibration requires it.

After each solve, DeepCompressor requantizes `W - L`, evaluates its unchanged full `OutputsError`, and keeps the best
iteration. Thus the local closed-form regression proposes candidates and the existing module-level objective selects
them. Blocks remain ordered, and the checkpoint/runtime path is identical to the weight-SVD control.

Run a bounded rank-32 pilot first:

```bash
deepcompressor-svdquant quantize \
  --gpu 0 \
  --dataset runs/anima-aesthetic-v1.1/aesthetic-v1.1-calibration-10000prompts/hf_dataset \
  --num-samples 100 --rank 32 --num-iters 4 --fast --activation-aware \
  --activation-damping 1e-4 --activation-num-tokens 64 --resume \
  --run-name anima-r32-aware-100samples-4iter \
  --output runs/anima-aesthetic-v1.1/anima-r32-aware-100samples-4iter
```

MLflow records `low_rank_solver=activation-aware-rrr`, damping, token sampling, sample count, rank, and iteration count.
If the pilot improves held-out raw pixels, increase the sample count while keeping those controls explicit. A later
gradient-only polish of `A/B` can use the same DeepCompressor `OutputsError`, but it is not required by this solver and is
not implemented yet.

## Validate raw pixels

```bash
deepcompressor-svdquant validate \
  --gpu 0 \
  --manifest runs/anima-aesthetic-v1.1/anima-r32-exact-128samples/nunchaku/anima-aesthetic-v1.1-svdquant-int4.json \
  --prompts examples/diffusion/prompts/anima-aesthetic-v1.1-calibration-10000.yaml \
  --prompt-offset 1000 --num-prompts 100 \
  --steps 30 --threshold 0.99 \
  --output runs/anima-aesthetic-v1.1/anima-r32-exact-128samples/validation-held-out
```

Validation regenerates BF16 and INT4 samples with identical prompt IDs and seeds, decodes both through the same VAE, and
reports RGB RMSE, MAE, maximum absolute error, and `pixel_similarity = 1 - RMSE`. It exits with status 2 when any sample
falls below the threshold.

## Benchmark the fused denoiser

```bash
deepcompressor-svdquant benchmark \
  --gpu 0 \
  --manifest runs/anima-aesthetic-v1.1/anima-r32-exact-128samples/nunchaku/anima-aesthetic-v1.1-svdquant-int4.json \
  --dataset runs/anima-aesthetic-v1.1/aesthetic-v1.1-calibration-10000prompts/hf_dataset \
  --num-samples 16 --warmup 2 --iterations 10 \
  --output runs/anima-aesthetic-v1.1/anima-r32-exact-128samples/benchmark
```

This separates denoiser speed from text encoding, sampler orchestration, VAE decoding, and image I/O.

## MLflow experiment tracking

Tracking defaults to `https://mlflow.appmana.com`, experiment `anima-aesthetic-v1.1-svdquant`. Quantization creates
`mlflow-run.json` beside the checkpoint. Validation and benchmarking discover that file and resume the same run, so one
MLflow run contains the recipe and its quality/performance results.

The integration records:

- git commit, rank, SVD mode, sample count, iterations, quantization formats, group size, and software/GPU versions;
- model-load, smoothing, low-rank calibration, packing, validation, and total wall times;
- the recipe, PTQ log, Nunchaku manifest, raw-pixel metrics, paired images, and benchmark results;
- minimum/mean pixel similarity, target gap to 0.99, acceptance rate, and denoiser/image speedups.

Credentials remain in environment variables or Kubernetes secrets and are never written to run artifacts. Cluster jobs
use `mlflow-auth-admin-secret`, `mlflow-s3-user`, and the SeaweedFS S3 endpoint described in the
[operator notes](examples/diffusion/anima/README.md#appmana-mlflow-tracking).

List experiments in descending minimum-pixel-similarity order:

```bash
deepcompressor-svdquant compare
```

Use `--no-track` only for an intentionally offline smoke test.

### Results recorded so far

These are randomized-SVD integration pilots, not successful production checkpoints:

| Anima recipe | Validation prompts | Mean pixel similarity | Minimum pixel similarity | Result |
|---|---:|---:|---:|---|
| Rank 32, 1 residual iteration | 100 | 0.7493 | 0.5507 | Fails 0.99 |
| Rank 128, 1 residual iteration | 100 | 0.7750 | 0.5281 | Fails 0.99 |
| Rank 128, 4 residual iterations | 100 | 0.7834 | 0.5371 | Fails 0.99 |

The four-iteration rank-128 run improved mean similarity over one iteration, but remains far from the raw-pixel target.
That is evidence for optimizing the low-rank branch, not evidence that simply increasing the PTQ calibration set will
solve the gap.

The controlled published Qwen-Image rank-32 baseline is MLflow run `804e2aba1b7b45ab969ed0d3493ebf11`:

- mean pixel similarity: `0.844220`;
- minimum pixel similarity: `0.750185`;
- BF16 wall time: `557.703 s` with two-GPU sharding and disk overflow;
- INT4 wall time: `46.720 s` on one GPU.

The resulting `11.94×` wall-time ratio is not a kernel speedup claim because the BF16 baseline used disk-offloaded
overflow. Use `benchmark` for a congruent fused-denoiser comparison.

## Upstream references

- [SVDQuant paper](https://arxiv.org/abs/2411.05007)
- [DeepCompressor](https://github.com/nunchaku-ai/deepcompressor)
- [Nunchaku](https://github.com/nunchaku-ai/nunchaku)
- [Original DeepCompressor diffusion guide](examples/diffusion/README.md)

This fork retains the upstream Apache-2.0 license in [LICENSE](LICENSE).
