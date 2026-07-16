"""Command-line workflow for the 100-prompt Anima SVDQuant pilot."""

from __future__ import annotations

import gc
import json
import os
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
    state.wait_for_everyone()
    prompts = load_prompt_dataset(args.prompts, args.num_prompts, args.prompt_offset)
    with state.split_between_processes(prompts, apply_padding=False) as local_prompts:
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
            )
            metadata["accelerate_rank"] = state.process_index
            metadata["accelerate_device"] = str(state.device)
            shard_path = shards / f"rank-{state.process_index:05d}.json"
            shard_path.write_text(json.dumps(metadata, indent=2) + "\n")
        finally:
            components.close()
    state.wait_for_everyone()
    if state.is_main_process:
        shard_metadata = [json.loads(path.read_text()) for path in sorted(shards.glob("rank-*.json"))]
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

    from deepcompressor.app.diffusion.nn.patch import shift_input_activations
    from deepcompressor.app.diffusion.ptq import ptq
    from deepcompressor.utils import tools

    from .config import build_anima_svdquant_config
    from .nunchaku import export_nunchaku_checkpoint

    output = Path(args.output).expanduser().resolve()
    quant_dir = output / "deepcompressor"
    packed_dir = output / "nunchaku"
    quant_dir.mkdir(parents=True, exist_ok=True)
    tools.logging.setup(path=str(output / "ptq.log"), level=tools.logging.INFO)
    components, diffusion, struct_cls = _load_ptq_model(_require_path(args.model, "Anima Aesthetic 1.1 model"))
    try:
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
        )
        ptq(
            model,
            config,
            cache=None,
            load_dirpath=str(quant_dir) if args.resume else "",
            save_dirpath=str(quant_dir),
            copy_on_save=True,
            save_model=True,
        )
        weights, manifest = export_nunchaku_checkpoint(
            model,
            quant_dir,
            packed_dir,
            rank=args.rank,
        )
        print(json.dumps({"weights": str(weights), "manifest": str(manifest)}, indent=2))
    finally:
        components.close()
        tools.logging.shutdown()
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

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    components = load_components(
        _require_path(args.model, "Anima Aesthetic 1.1 model"),
        _require_path(args.text_encoder, "Anima text encoder"),
        _require_path(args.vae, "Qwen Image VAE"),
        dtype=torch.bfloat16,
    )
    prompts = load_prompts(args.prompts, args.num_prompts, args.prompt_offset)
    references: dict[str, torch.Tensor] = {}
    results: list[dict] = []
    try:
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
        apply_nunchaku_checkpoint(components.diffusion_model, args.manifest)
        model_management.load_models_gpu([components.model], force_full_load=True)
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
            save_image(pixels, output / "int4" / f"{name}.png")
            metrics = raw_pixel_similarity(references.pop(name), pixels)
            results.append({"name": name, "prompt": prompt, **metrics})
        similarities = [result["pixel_similarity"] for result in results]
        report = {
            "acceptance_threshold": args.threshold,
            "accepted": bool(similarities) and min(similarities) >= args.threshold,
            "mean_pixel_similarity": sum(similarities) / len(similarities),
            "min_pixel_similarity": min(similarities),
            "samples": results,
        }
        (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0 if report["accepted"] else 2
    finally:
        references.clear()
        components.close()
        gc.collect()


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
    )


@app.command("quantize", help="Run SVDQuant PTQ and pack Nunchaku weights.")
def quantize_cli(
    model: str = typer.Option(MODEL_DEFAULT, help="Anima Aesthetic 1.1 safetensors."),
    gpu: int = typer.Option(0, help="Physical GPU for this PTQ process."),
    dataset: str = typer.Option(str(OUTPUT_DEFAULT / "dataset/hf_dataset")),
    num_samples: int = typer.Option(100, min=1),
    rank: int = typer.Option(32, help="BF16 SVD branch rank; supported values are 32 and 128."),
    num_iters: int = typer.Option(100, min=1),
    output: str = typer.Option(str(OUTPUT_DEFAULT / "rank32")),
    fast: bool = typer.Option(
        False,
        "--fast",
        help="Use one smoothing candidate and one randomized residual SVD integration pass.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Reuse completed caches already present in the output directory.",
    ),
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
        resume=resume,
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
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
