"""Command-line workflow for the 100-prompt Anima SVDQuant pilot."""

from __future__ import annotations

import contextlib
import gc
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import typer


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _first_existing(patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(Path.home().glob(pattern))
        if matches:
            return matches[-1].resolve()
    return None


def _default_model() -> Path | None:
    return _first_existing(
        [
            ".cache/huggingface/hub/models--circlestone-labs--Anima/snapshots/*/split_files/"
            "diffusion_models/anima-aesthetic-v1.1.safetensors"
        ]
    )


def _default_text_encoder() -> Path | None:
    return _first_existing(
        [
            ".cache/huggingface/hub/models--circlestone-labs--Anima/snapshots/*/split_files/"
            "text_encoders/qwen_3_06b_base.safetensors"
        ]
    )


def _default_vae() -> Path | None:
    return _first_existing(
        [
            ".cache/huggingface/hub/models--Comfy-Org--Qwen-Image_ComfyUI/snapshots/*/split_files/"
            "vae/qwen_image_vae.safetensors",
            ".cache/huggingface/hub/models--Comfy-Org--Qwen-Image_ComfyUI/snapshots/*/"
            "split_files/vae/qwen_image_vae.safetensors",
        ]
    )


def _require_path(value: str | Path | None, label: str) -> Path:
    if value is None:
        raise FileNotFoundError(f"No {label} was supplied and no cached default was found")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


PROMPTS_DEFAULT = str(_repo_root() / "examples/diffusion/prompts/qdiff.yaml")
OUTPUT_DEFAULT = _repo_root() / "runs/anima-aesthetic-v1.1"
MODEL_DEFAULT = str(_default_model() or "")
TEXT_ENCODER_DEFAULT = str(_default_text_encoder() or "")
VAE_DEFAULT = str(_default_vae() or "")
MLFLOW_URI_DEFAULT = os.environ.get("MLFLOW_TRACKING_URI", "https://mlflow.appmana.com")
MLFLOW_EXPERIMENT_DEFAULT = "anima-aesthetic-v1.1-svdquant"

app = typer.Typer(
    name="deepcompressor-svdquant",
    help="Native ComfyUI Anima Aesthetic 1.1 SVDQuant W4A4 pilot.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


def _select_gpu(gpu: int) -> None:
    # Accelerate assigns each process before ComfyUI is imported. Keep the
    # explicit physical-GPU convenience for ordinary one-process invocations.
    if "LOCAL_RANK" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def _weighted_partition(total: int, weights: list[float], rank: int) -> tuple[int, int]:
    if not weights or any(weight <= 0 for weight in weights):
        raise ValueError("Accelerate rank weights must all be positive")
    if rank < 0 or rank >= len(weights):
        raise IndexError(f"Rank {rank} is outside {len(weights)} partition weights")
    weight_sum = sum(weights)
    boundaries = [0]
    cumulative = 0.0
    for weight in weights[:-1]:
        cumulative += weight
        boundaries.append(round(total * cumulative / weight_sum))
    boundaries.append(total)
    return boundaries[rank], boundaries[rank + 1]


def _wait_for_collection_shards(
    shards: Path,
    num_processes: int,
    *,
    timeout_seconds: float = 24 * 60 * 60,
) -> list[dict]:
    """Wait for atomically published collection metadata without a GPU collective.

    Weighted prompt partitions intentionally take different amounts of time.
    A final NCCL barrier uses the process-group timeout (ten minutes by default),
    so a healthy slower rank can otherwise be killed after a faster rank finishes.
    """

    expected = [shards / f"rank-{rank:05d}.json" for rank in range(num_processes)]
    deadline = time.monotonic() + timeout_seconds
    last_missing: tuple[int, ...] | None = None
    while True:
        missing = tuple(rank for rank, path in enumerate(expected) if not path.is_file())
        if not missing:
            return [json.loads(path.read_text()) for path in expected]
        if missing != last_missing:
            print(json.dumps({"waiting_for_accelerate_ranks": list(missing)}), flush=True)
            last_missing = missing
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for collection metadata from Accelerate ranks {missing}")
        time.sleep(1.0)


def command_collect(args: SimpleNamespace) -> int:
    import torch
    from accelerate import PartialState

    from .pipeline import (
        collect_calibration_dataset,
        load_components,
        load_prompt_dataset,
        save_calibration_manifest,
    )

    state = PartialState()
    output = Path(args.output).expanduser().resolve()
    shards = output / ".accelerate-shards"
    if (output / "hf_dataset").exists():
        raise FileExistsError(f"Collection output already contains a dataset: {output / 'hf_dataset'}")
    if state.is_main_process:
        shards.mkdir(parents=True, exist_ok=True)
        # Shards describe one invocation. Remove incomplete/stale metadata
        # before any rank begins so resume reconstructs it from validated files.
        for path in (*shards.glob("rank-*.json"), *shards.glob(".rank-*.json.tmp")):
            path.unlink()
    state.wait_for_everyone()
    prompts = load_prompt_dataset(args.prompts, args.num_prompts, args.prompt_offset)
    partition = None
    if args.rank_weights:
        rank_weights = [float(value.strip()) for value in args.rank_weights.split(",") if value.strip()]
        if len(rank_weights) != state.num_processes:
            raise ValueError(
                f"Expected {state.num_processes} comma-separated rank weights, received {len(rank_weights)}"
            )
        start, end = _weighted_partition(len(prompts), rank_weights, state.process_index)
        partition = {"start": start, "end": end, "weight": rank_weights[state.process_index]}
        prompt_context = contextlib.nullcontext(prompts.select(range(start, end)))
    else:
        prompt_context = state.split_between_processes(prompts, apply_padding=False)
    with prompt_context as local_prompts:
        components = load_components(
            _require_path(args.model, "Anima Aesthetic 1.1 model"),
            _require_path(args.text_encoder, "Anima text encoder"),
            dtype=torch.bfloat16,
        )
        try:
            metadata = collect_calibration_dataset(
                components,
                local_prompts,
                output,
                width=args.width,
                height=args.height,
                steps=args.steps,
                cfg=args.cfg,
                sampler=args.sampler,
                scheduler=args.scheduler,
                resume=args.resume,
            )
            metadata["accelerate_rank"] = state.process_index
            metadata["accelerate_device"] = str(state.device)
            metadata["accelerate_partition"] = partition
            shard_path = shards / f"rank-{state.process_index:05d}.json"
            temporary_shard_path = shards / f".rank-{state.process_index:05d}.json.tmp"
            temporary_shard_path.write_text(json.dumps(metadata, indent=2) + "\n")
            temporary_shard_path.replace(shard_path)
        finally:
            components.close()
    if state.is_main_process:
        shard_metadata = _wait_for_collection_shards(shards, state.num_processes)
        manifest_path, metadata = save_calibration_manifest(output, shard_metadata)
        print(json.dumps({**metadata, "hf_dataset": str(manifest_path)}, indent=2))
    state.destroy_process_group()
    return 0


def _load_ptq_model(model_path: Path):
    import torch
    from comfy import model_management

    from .pipeline import load_diffusion_model
    from .struct import AnimaModelStruct

    components = load_diffusion_model(model_path, dtype=torch.bfloat16)
    model_management.load_models_gpu([components.model], force_full_load=True)
    diffusion = components.diffusion_model
    if diffusion.__class__.__name__ != "Anima":
        components.close()
        raise TypeError(f"Expected native ComfyUI Anima, got {type(diffusion)}")
    return components, diffusion, AnimaModelStruct


def command_quantize(args: SimpleNamespace) -> int:
    import torch

    from deepcompressor.app.diffusion.nn.patch import shift_input_activations
    from deepcompressor.app.diffusion.ptq import ptq
    from deepcompressor.utils import tools

    from .config import build_anima_svdquant_config
    from .nunchaku import export_nunchaku_checkpoint
    from .tracking import REFERENCE_FILENAME, ExperimentTracker

    output = Path(args.output).expanduser().resolve()
    quant_dir = output / "deepcompressor"
    packed_dir = output / "nunchaku"
    quant_dir.mkdir(parents=True, exist_ok=True)
    model_path = _require_path(args.model, "Anima Aesthetic 1.1 model")
    dataset_path = Path(args.dataset).expanduser().resolve()
    recipe = {
        "base_model": "anima-aesthetic-v1.1",
        "model_path": str(model_path),
        "calibration_dataset": str(dataset_path),
        "num_samples": args.num_samples,
        "svd_rank": args.rank,
        "low_rank_iterations": args.num_iters,
        "svd_mode": "randomized" if args.fast else "exact",
        "svd_oversample": 8,
        "svd_power_iterations": 2,
        "low_rank_solver": "activation-aware-rrr" if args.activation_aware else "weight-svd",
        "activation_damping": args.activation_damping if args.activation_aware else "",
        "activation_tokens_per_cache": args.activation_num_tokens if args.activation_aware else "",
        "weight_dtype": "sint4",
        "activation_dtype": "sint4",
        "group_size": 64,
        "low_rank_dtype": "bfloat16",
        "fast": args.fast,
        "resume": args.resume,
        "runtime": "nunchaku",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda": torch.version.cuda or "",
        "torch": torch.__version__,
    }
    recipe_path = output / "recipe.json"
    recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")
    timings: dict[str, float] = {}
    components = None
    weights = manifest = None
    tracker = ExperimentTracker(
        enabled=args.track,
        tracking_uri=args.mlflow_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        run_id=args.mlflow_run_id,
        tags={
            "stage": "quantize",
            "recipe.fast": args.fast,
            "recipe.rank": args.rank,
            "recipe.low_rank_solver": recipe["low_rank_solver"],
        },
    )
    with tracker:
        # MLflow initializes Python logging on import. Install DeepCompressor's
        # file handler afterward so the tracking client cannot replace it.
        tools.logging.setup(path=str(output / "ptq.log"), level=tools.logging.INFO)
        tracker.save_reference(output / REFERENCE_FILENAME)
        # MLflow parameters are immutable. Cache reuse describes this
        # execution attempt rather than the quantization recipe, so keep it as
        # a mutable tag and allow a failed run to resume its completed stages.
        tracker.log_params({key: value for key, value in recipe.items() if key != "resume"})
        tracker.set_tags({"execution.resume": args.resume})
        tracker.log_artifact(recipe_path, artifact_path="recipe")
        total_started = time.perf_counter()
        try:
            phase_started = time.perf_counter()
            components, diffusion, struct_cls = _load_ptq_model(model_path)
            timings["time.model_load_seconds"] = time.perf_counter() - phase_started
            # This is the published INT4 GELU shift. It is later consumed by
            # Nunchaku's fused GELU/MLP kernel, not left as a Python-side op.
            shift_input_activations(diffusion)
            model = struct_cls.construct(diffusion)
            config = build_anima_svdquant_config(
                args.dataset,
                rank=args.rank,
                num_samples=args.num_samples,
                num_iters=args.num_iters,
                fast=args.fast,
                activation_aware=args.activation_aware,
                activation_damping=args.activation_damping,
                activation_num_tokens=args.activation_num_tokens,
            )
            phase_started = time.perf_counter()
            ptq(
                model,
                config,
                cache=None,
                load_dirpath=str(quant_dir) if args.resume else "",
                save_dirpath=str(quant_dir),
                copy_on_save=True,
                save_model=True,
                timings=timings,
            )
            timings["time.ptq.total_seconds"] = time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            weights, manifest = export_nunchaku_checkpoint(
                model,
                quant_dir,
                packed_dir,
                rank=args.rank,
            )
            timings["time.pack_seconds"] = time.perf_counter() - phase_started
            timings["artifact.checkpoint_bytes"] = weights.stat().st_size
            timings["artifact.checkpoint_gib"] = weights.stat().st_size / 1024**3
            tracker.log_artifact(manifest, artifact_path="checkpoint")
        finally:
            timings["time.total_seconds"] = time.perf_counter() - total_started
            if components is not None:
                components.close()
            tools.logging.shutdown()
            tracker.log_metrics(timings)
            tracker.log_artifact(output / "ptq.log", artifact_path="logs")
            tracker.log_artifact(output / REFERENCE_FILENAME, artifact_path="recipe")
    result = {
        "weights": str(weights),
        "manifest": str(manifest),
        "mlflow_run_id": tracker.run_id,
        "timings": timings,
    }
    print(json.dumps(result, indent=2))
    return 0


def command_export(args: SimpleNamespace) -> int:
    import torch

    from deepcompressor.app.diffusion.nn.patch import shift_input_activations

    from .nunchaku import export_nunchaku_checkpoint

    components, diffusion, struct_cls = _load_ptq_model(_require_path(args.model, "Anima Aesthetic 1.1 model"))
    try:
        shift_input_activations(diffusion)
        model_state = torch.load(
            Path(args.quant_dir).expanduser().resolve() / "model.pt",
            map_location=next(diffusion.parameters()).device,
            weights_only=True,
        )
        diffusion.load_state_dict(model_state, strict=True)
        model = struct_cls.construct(diffusion)
        weights, manifest = export_nunchaku_checkpoint(
            model,
            args.quant_dir,
            args.output,
            rank=args.rank,
        )
        print(json.dumps({"weights": str(weights), "manifest": str(manifest)}, indent=2))
    finally:
        components.close()
    return 0


def command_validate(args: SimpleNamespace) -> int:
    import torch
    from comfy import model_management

    from deepcompressor.utils.common import hash_str_to_int

    from .nunchaku import apply_nunchaku_checkpoint
    from .pipeline import (
        decode_latent,
        load_components,
        load_prompts,
        raw_pixel_similarity,
        sample_latent,
        save_image,
    )
    from .tracking import REFERENCE_FILENAME, ExperimentTracker, RunReference, find_run_reference

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.prompts, args.num_prompts, args.prompt_offset)
    checkpoint_manifest_path = Path(args.manifest).expanduser().resolve()
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text())
    reference_path = find_run_reference(args.manifest, args.mlflow_reference)
    run_reference = RunReference.load(reference_path) if reference_path is not None else None
    run_id = args.mlflow_run_id or (run_reference.run_id if run_reference is not None else "")
    tracking_uri = args.mlflow_uri or (run_reference.tracking_uri if run_reference is not None else "")
    tracker = ExperimentTracker(
        enabled=args.track,
        tracking_uri=tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        run_id=run_id,
        tags={
            "stage": "validated",
            "validation.steps": args.steps,
            "validation.resolution": f"{args.width}x{args.height}",
            "validation.prompt_count": len(prompts),
        },
    )
    validation_recipe = {
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "prompts": str(Path(args.prompts).expanduser().resolve()),
        "num_prompts": len(prompts),
        "prompt_offset": args.prompt_offset,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "cfg": args.cfg,
        "sampler": args.sampler,
        "scheduler": args.scheduler,
        "acceptance_threshold": args.threshold,
        "metric": "1 - raw_rgb_rmse",
    }
    validation_recipe_path = output / "validation-recipe.json"
    validation_recipe_path.write_text(json.dumps(validation_recipe, indent=2) + "\n")
    components = None
    references: dict[str, torch.Tensor] = {}
    results: list[dict] = []
    validation_started = time.perf_counter()
    with tracker:
        tracker.save_reference(output / REFERENCE_FILENAME)
        if not run_id:
            tracker.log_params(
                {
                    "base_model": checkpoint_manifest.get("base_model", "anima-aesthetic-v1.1"),
                    "svd_rank": checkpoint_manifest.get("rank", ""),
                    "weight_dtype": "sint4",
                    "activation_dtype": "sint4",
                    "group_size": checkpoint_manifest.get("group_size", ""),
                    "low_rank_dtype": checkpoint_manifest.get("low_rank_dtype", ""),
                    "runtime": "nunchaku",
                    "checkpoint_manifest": str(checkpoint_manifest_path),
                    "validation_only_import": True,
                }
            )
        tracker.log_artifact(validation_recipe_path, artifact_path="validation")
        try:
            components = load_components(
                _require_path(args.model, "Anima Aesthetic 1.1 model"),
                _require_path(args.text_encoder, "Anima text encoder"),
                _require_path(args.vae, "Qwen Image VAE"),
                dtype=torch.bfloat16,
            )
            phase_started = time.perf_counter()
            for name, prompt in prompts:
                seed = hash_str_to_int(name)
                latent = sample_latent(
                    components,
                    prompt,
                    seed=seed,
                    width=args.width,
                    height=args.height,
                    steps=args.steps,
                    cfg=args.cfg,
                    sampler=args.sampler,
                    scheduler=args.scheduler,
                )
                pixels = decode_latent(components.vae, latent)
                references[name] = pixels
                save_image(pixels, output / "bf16" / f"{name}.png")
            bf16_seconds = time.perf_counter() - phase_started
            apply_nunchaku_checkpoint(components.diffusion_model, args.manifest)
            model_management.load_models_gpu([components.model], force_full_load=True)
            phase_started = time.perf_counter()
            for sample_index, (name, prompt) in enumerate(prompts):
                seed = hash_str_to_int(name)
                latent = sample_latent(
                    components,
                    prompt,
                    seed=seed,
                    width=args.width,
                    height=args.height,
                    steps=args.steps,
                    cfg=args.cfg,
                    sampler=args.sampler,
                    scheduler=args.scheduler,
                )
                pixels = decode_latent(components.vae, latent)
                save_image(pixels, output / "int4" / f"{name}.png")
                metrics = raw_pixel_similarity(references.pop(name), pixels)
                results.append({"name": name, "prompt": prompt, **metrics})
                tracker.log_metrics(
                    {f"quality.sample.{key}": value for key, value in metrics.items()},
                    step=sample_index,
                )
            int4_seconds = time.perf_counter() - phase_started
            similarities = [result["pixel_similarity"] for result in results]
            rmses = [result["rmse"] for result in results]
            maes = [result["mae"] for result in results]
            max_errors = [result["max_abs_error"] for result in results]
            accepted_samples = sum(similarity >= args.threshold for similarity in similarities)
            report = {
                "acceptance_threshold": args.threshold,
                "accepted": bool(similarities) and min(similarities) >= args.threshold,
                "acceptance_rate": accepted_samples / len(similarities),
                "mean_pixel_similarity": sum(similarities) / len(similarities),
                "min_pixel_similarity": min(similarities),
                "mean_rmse": sum(rmses) / len(rmses),
                "mean_mae": sum(maes) / len(maes),
                "max_abs_error": max(max_errors),
                "target_gap": max(0.0, args.threshold - min(similarities)),
                "bf16_seconds": bf16_seconds,
                "int4_seconds": int4_seconds,
                "samples": results,
            }
            metrics_path = output / "metrics.json"
            metrics_path.write_text(json.dumps(report, indent=2) + "\n")
            tracker.log_metrics(
                {
                    "quality.pixel_similarity.mean": report["mean_pixel_similarity"],
                    "quality.pixel_similarity.min": report["min_pixel_similarity"],
                    "quality.rmse.mean": report["mean_rmse"],
                    "quality.mae.mean": report["mean_mae"],
                    "quality.max_abs_error": report["max_abs_error"],
                    "quality.acceptance_threshold": args.threshold,
                    "quality.acceptance_rate": report["acceptance_rate"],
                    "quality.accepted": int(report["accepted"]),
                    "quality.target_gap": report["target_gap"],
                    "time.validation.bf16_seconds": bf16_seconds,
                    "time.validation.int4_seconds": int4_seconds,
                    "time.validation.total_seconds": time.perf_counter() - validation_started,
                    "performance.image.int4_vs_bf16_speedup": bf16_seconds / int4_seconds,
                }
            )
            tracker.log_artifact(metrics_path, artifact_path="validation")
            tracker.log_artifacts(output / "bf16", artifact_path="validation/bf16")
            tracker.log_artifacts(output / "int4", artifact_path="validation/int4")
            tracker.log_artifact(output / REFERENCE_FILENAME, artifact_path="validation")
            print(json.dumps({**report, "mlflow_run_id": tracker.run_id}, indent=2))
            return 0 if report["accepted"] else 2
        finally:
            references.clear()
            if components is not None:
                components.close()
            gc.collect()


def command_benchmark(args: SimpleNamespace) -> int:
    import torch
    from comfy import model_management
    from datasets import load_from_disk

    from .nunchaku import apply_nunchaku_checkpoint
    from .tracking import REFERENCE_FILENAME, ExperimentTracker, RunReference, find_run_reference

    def move_to_device(value, device: torch.device):
        if isinstance(value, torch.Tensor):
            return value.to(device=device)
        if isinstance(value, list):
            return [move_to_device(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(move_to_device(item, device) for item in value)
        if isinstance(value, dict):
            return {key: move_to_device(item, device) for key, item in value.items()}
        return value

    dataset_path = Path(args.dataset).expanduser().resolve()
    dataset = load_from_disk(str(dataset_path))
    rows = [row for row in dataset if row.get("selected_for_calibration")][: args.num_samples]
    if not rows:
        raise ValueError(f"No selected calibration records in {dataset_path}")
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference_path = find_run_reference(args.manifest, args.mlflow_reference)
    run_reference = RunReference.load(reference_path) if reference_path is not None else None
    run_id = args.mlflow_run_id or (run_reference.run_id if run_reference is not None else "")
    tracking_uri = args.mlflow_uri or (run_reference.tracking_uri if run_reference is not None else "")
    tracker = ExperimentTracker(
        enabled=args.track,
        tracking_uri=tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        run_id=run_id,
        tags={"stage": "benchmarked", "benchmark": "denoiser-forward"},
    )
    components = None
    with tracker:
        tracker.save_reference(output / REFERENCE_FILENAME)
        components, diffusion, _ = _load_ptq_model(_require_path(args.model, "Anima Aesthetic 1.1 model"))
        device = next(diffusion.parameters()).device
        caches = []
        for row in rows:
            cache_path = Path(row["cache_path"])
            if not cache_path.is_absolute():
                cache_path = dataset_path.parent / cache_path
            caches.append(move_to_device(torch.load(cache_path, weights_only=True), device))

        def measure() -> float:
            with torch.inference_mode():
                for _ in range(args.warmup):
                    for cache in caches:
                        result = diffusion(*cache["input_args"], **cache["input_kwargs"])
                        del result
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                for _ in range(args.iterations):
                    for cache in caches:
                        result = diffusion(*cache["input_args"], **cache["input_kwargs"])
                        del result
                torch.cuda.synchronize(device)
            return (time.perf_counter() - started) / (args.iterations * len(caches))

        try:
            bf16_seconds = measure()
            apply_nunchaku_checkpoint(diffusion, args.manifest)
            model_management.load_models_gpu([components.model], force_full_load=True)
            int4_seconds = measure()
            report = {
                "num_timestep_records": len(caches),
                "warmup_rounds": args.warmup,
                "measured_rounds": args.iterations,
                "bf16_milliseconds_per_denoiser": bf16_seconds * 1000,
                "int4_milliseconds_per_denoiser": int4_seconds * 1000,
                "int4_vs_bf16_speedup": bf16_seconds / int4_seconds,
            }
            report_path = output / "denoiser-benchmark.json"
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            tracker.log_metrics(
                {
                    "performance.denoiser.bf16_milliseconds": bf16_seconds * 1000,
                    "performance.denoiser.int4_milliseconds": int4_seconds * 1000,
                    "performance.denoiser.int4_vs_bf16_speedup": bf16_seconds / int4_seconds,
                    "performance.denoiser.timestep_records": len(caches),
                    "performance.denoiser.iterations": args.iterations,
                }
            )
            tracker.log_artifact(report_path, artifact_path="benchmark")
            print(json.dumps({**report, "mlflow_run_id": tracker.run_id}, indent=2))
        finally:
            caches.clear()
            if components is not None:
                components.close()
    return 0


def _invoke(command, gpu: int, **kwargs) -> None:
    _select_gpu(gpu)
    status = command(SimpleNamespace(gpu=gpu, **kwargs))
    if status:
        raise typer.Exit(status)


@app.command("collect", help="Generate trajectories and a Hugging Face calibration dataset.")
def collect_cli(
    model: str = typer.Option(MODEL_DEFAULT, help="Anima Aesthetic 1.1 safetensors."),
    text_encoder: str = typer.Option(TEXT_ENCODER_DEFAULT, help="Native ComfyUI Anima text encoder."),
    gpu: int = typer.Option(0, help="Physical GPU for a direct one-process invocation."),
    prompts: str = typer.Option(PROMPTS_DEFAULT, help="YAML, JSON, or text prompt dataset."),
    num_prompts: int = typer.Option(100, min=1),
    prompt_offset: int = typer.Option(0, min=0),
    output: str = typer.Option(str(OUTPUT_DEFAULT / "dataset")),
    width: int = typer.Option(512, min=16),
    height: int = typer.Option(512, min=16),
    steps: int = typer.Option(30, min=1),
    cfg: float = typer.Option(4.0, min=0.0),
    sampler: str = typer.Option("er_sde"),
    scheduler: str = typer.Option("simple"),
    rank_weights: str = typer.Option(
        "",
        help="Optional comma-separated relative prompt counts for Accelerate ranks, e.g. 0.42,0.58.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Reuse prompts whose latent and complete trajectory cache already exist.",
    ),
) -> None:
    _invoke(
        command_collect,
        gpu,
        model=model,
        text_encoder=text_encoder,
        prompts=prompts,
        num_prompts=num_prompts,
        prompt_offset=prompt_offset,
        output=output,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        sampler=sampler,
        scheduler=scheduler,
        rank_weights=rank_weights,
        resume=resume,
    )


@app.command("quantize", help="Run SVDQuant PTQ and pack Nunchaku weights.")
def quantize_cli(
    model: str = typer.Option(MODEL_DEFAULT, help="Anima Aesthetic 1.1 safetensors."),
    gpu: int = typer.Option(0, help="Physical GPU for this PTQ process."),
    dataset: str = typer.Option(str(OUTPUT_DEFAULT / "dataset/hf_dataset")),
    num_samples: int = typer.Option(100, min=1),
    rank: int = typer.Option(32, help="BF16 SVD branch rank; supported values are 32 and 128."),
    num_iters: int = typer.Option(1, min=1, help="Iterative residual-SVD passes."),
    output: str = typer.Option(str(OUTPUT_DEFAULT / "rank32")),
    fast: bool = typer.Option(
        False,
        "--fast",
        help="Use one smoothing candidate and randomized truncated SVD passes.",
    ),
    activation_aware: bool = typer.Option(
        True,
        "--activation-aware/--weight-svd",
        help="Fit the W16A16 branch to W4A4 activation error; use --weight-svd for the released control.",
    ),
    activation_damping: float = typer.Option(
        1e-4,
        min=0.0,
        help="Relative diagonal damping for activation-aware covariance factorization.",
    ),
    activation_num_tokens: int = typer.Option(
        64,
        min=-1,
        help="Activation rows sampled per cached tensor; -1 uses every row.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Reuse completed caches already present in the output directory.",
    ),
    track: bool = typer.Option(True, "--track/--no-track", help="Record the recipe and timings in MLflow."),
    mlflow_uri: str = typer.Option(MLFLOW_URI_DEFAULT, help="AppMana MLflow tracking server."),
    experiment_name: str = typer.Option(MLFLOW_EXPERIMENT_DEFAULT, help="MLflow experiment name."),
    run_name: str = typer.Option("", help="Optional human-readable MLflow run name."),
    mlflow_run_id: str = typer.Option("", help="Explicit MLflow run to resume."),
) -> None:
    _invoke(
        command_quantize,
        gpu,
        model=model,
        dataset=dataset,
        num_samples=num_samples,
        rank=rank,
        num_iters=num_iters,
        output=output,
        fast=fast,
        activation_aware=activation_aware,
        activation_damping=activation_damping,
        activation_num_tokens=activation_num_tokens,
        resume=resume,
        track=track,
        mlflow_uri=mlflow_uri,
        experiment_name=experiment_name,
        run_name=run_name,
        mlflow_run_id=mlflow_run_id,
    )


@app.command("export", help="Repack an existing DeepCompressor PTQ directory for Nunchaku.")
def export_cli(
    quant_dir: str = typer.Option(..., help="Directory containing model.pt, scale.pt, and branch.pt."),
    output: str = typer.Option(..., help="Destination for packed safetensors and its manifest."),
    model: str = typer.Option(MODEL_DEFAULT, help="Anima Aesthetic 1.1 safetensors."),
    gpu: int = typer.Option(0),
    rank: int = typer.Option(32, help="BF16 SVD branch rank; supported values are 32 and 128."),
) -> None:
    _invoke(
        command_export,
        gpu,
        quant_dir=quant_dir,
        output=output,
        model=model,
        rank=rank,
    )


@app.command("validate", help="Compare raw BF16 and Nunchaku INT4 RGB pixels.")
def validate_cli(
    manifest: str = typer.Option(..., help="Nunchaku checkpoint JSON manifest."),
    model: str = typer.Option(MODEL_DEFAULT, help="Anima Aesthetic 1.1 safetensors."),
    text_encoder: str = typer.Option(TEXT_ENCODER_DEFAULT),
    vae: str = typer.Option(VAE_DEFAULT),
    gpu: int = typer.Option(0),
    prompts: str = typer.Option(PROMPTS_DEFAULT),
    num_prompts: int = typer.Option(4, min=1),
    prompt_offset: int = typer.Option(100, min=0),
    output: str = typer.Option(str(OUTPUT_DEFAULT / "validation")),
    width: int = typer.Option(512, min=16),
    height: int = typer.Option(512, min=16),
    steps: int = typer.Option(30, min=1),
    cfg: float = typer.Option(4.0, min=0.0),
    sampler: str = typer.Option("er_sde"),
    scheduler: str = typer.Option("simple"),
    threshold: float = typer.Option(0.99, min=0.0, max=1.0),
    track: bool = typer.Option(True, "--track/--no-track", help="Record raw-pixel results and images in MLflow."),
    mlflow_uri: str = typer.Option(MLFLOW_URI_DEFAULT, help="AppMana MLflow tracking server."),
    experiment_name: str = typer.Option(MLFLOW_EXPERIMENT_DEFAULT, help="MLflow experiment name."),
    run_name: str = typer.Option("", help="Run name when validation creates a new MLflow run."),
    mlflow_run_id: str = typer.Option("", help="Explicit MLflow run to resume."),
    mlflow_reference: str = typer.Option("", help="Explicit path to mlflow-run.json."),
) -> None:
    _invoke(
        command_validate,
        gpu,
        manifest=manifest,
        model=model,
        text_encoder=text_encoder,
        vae=vae,
        prompts=prompts,
        num_prompts=num_prompts,
        prompt_offset=prompt_offset,
        output=output,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        sampler=sampler,
        scheduler=scheduler,
        threshold=threshold,
        track=track,
        mlflow_uri=mlflow_uri,
        experiment_name=experiment_name,
        run_name=run_name,
        mlflow_run_id=mlflow_run_id,
        mlflow_reference=mlflow_reference,
    )


@app.command("benchmark", help="Measure steady-state BF16 versus Nunchaku INT4 denoiser forwards.")
def benchmark_cli(
    manifest: str = typer.Option(..., help="Nunchaku checkpoint JSON manifest."),
    dataset: str = typer.Option(str(OUTPUT_DEFAULT / "dataset/hf_dataset")),
    model: str = typer.Option(MODEL_DEFAULT, help="Anima Aesthetic 1.1 safetensors."),
    gpu: int = typer.Option(0),
    num_samples: int = typer.Option(8, min=1, help="Distinct cached timestep records."),
    warmup: int = typer.Option(2, min=0),
    iterations: int = typer.Option(10, min=1),
    output: str = typer.Option(str(OUTPUT_DEFAULT / "benchmark")),
    track: bool = typer.Option(True, "--track/--no-track"),
    mlflow_uri: str = typer.Option(MLFLOW_URI_DEFAULT),
    experiment_name: str = typer.Option(MLFLOW_EXPERIMENT_DEFAULT),
    run_name: str = typer.Option(""),
    mlflow_run_id: str = typer.Option(""),
    mlflow_reference: str = typer.Option(""),
) -> None:
    _invoke(
        command_benchmark,
        gpu,
        manifest=manifest,
        dataset=dataset,
        model=model,
        num_samples=num_samples,
        warmup=warmup,
        iterations=iterations,
        output=output,
        track=track,
        mlflow_uri=mlflow_uri,
        experiment_name=experiment_name,
        run_name=run_name,
        mlflow_run_id=mlflow_run_id,
        mlflow_reference=mlflow_reference,
    )


@app.command("compare", help="List SVDQuant recipes by raw-pixel fidelity from AppMana MLflow.")
def compare_cli(
    mlflow_uri: str = typer.Option(MLFLOW_URI_DEFAULT, help="AppMana MLflow tracking server."),
    experiment_name: str = typer.Option(MLFLOW_EXPERIMENT_DEFAULT, help="MLflow experiment name."),
    limit: int = typer.Option(25, min=1, max=1000),
) -> None:
    import mlflow

    mlflow.set_tracking_uri(mlflow_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise typer.BadParameter(f"MLflow experiment does not exist: {experiment_name}")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=limit,
        output_format="list",
    )
    rows = []
    for run in runs:
        rows.append(
            {
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName", ""),
                "status": run.info.status,
                "rank": run.data.params.get("svd_rank", ""),
                "svd_mode": run.data.params.get("svd_mode", ""),
                "num_samples": run.data.params.get("num_samples", ""),
                "min_pixel_similarity": run.data.metrics.get("quality.pixel_similarity.min"),
                "mean_pixel_similarity": run.data.metrics.get("quality.pixel_similarity.mean"),
                "target_gap": run.data.metrics.get("quality.target_gap"),
                "total_seconds": run.data.metrics.get("time.total_seconds"),
                "low_rank_seconds": run.data.metrics.get("time.ptq.low_rank_seconds"),
                "denoiser_speedup": run.data.metrics.get("performance.denoiser.int4_vs_bf16_speedup"),
                "image_speedup": run.data.metrics.get("performance.image.int4_vs_bf16_speedup"),
            }
        )
    rows.sort(
        key=lambda row: row["min_pixel_similarity"] if row["min_pixel_similarity"] is not None else -1.0,
        reverse=True,
    )
    typer.echo(json.dumps({"experiment": experiment_name, "runs": rows}, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
