"""Reproducible BF16 versus published Nunchaku Qwen-Image baseline."""

from __future__ import annotations

import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Annotated

import typer
import yaml

from deepcompressor.utils.common import hash_str_to_int

from .tracking import REFERENCE_FILENAME, ExperimentTracker


def _first_existing(pattern: str) -> Path | None:
    matches = sorted(Path.home().glob(pattern))
    return matches[-1].resolve() if matches else None


BASE_DEFAULT = str(_first_existing(".cache/huggingface/hub/models--wavespeed--Qwen-Image-bf16/snapshots/*") or "")
NUNCHAKU_DEFAULT = str(
    _first_existing(
        ".cache/huggingface/hub/models--nunchaku-tech--nunchaku-qwen-image/snapshots/*/"
        "svdq-int4_r32-qwen-image.safetensors"
    )
    or _first_existing(
        ".cache/huggingface/hub/models--nunchaku-ai--nunchaku-qwen-image/snapshots/*/"
        "svdq-int4_r32-qwen-image.safetensors"
    )
    or ""
)
PROMPTS_DEFAULT = str(
    Path(__file__).resolve().parents[4] / "examples/diffusion/prompts/anima-aesthetic-v1.1-calibration-10000.yaml"
)
OUTPUT_DEFAULT = Path(__file__).resolve().parents[4] / "runs/qwen-image-r32-baseline"
POSITIVE_MAGIC = " Ultra HD, 4K, cinematic composition."
NEGATIVE_PROMPT = " "

app = typer.Typer(
    name="deepcompressor-qwen-baseline",
    help="Compare official Qwen-Image BF16 with the published Nunchaku rank-32 W4A4 checkpoint.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


def _require_file(path: str | Path, label: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise FileNotFoundError(f"{label} does not exist: {value}")
    return value


def _require_dir(path: str | Path, label: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {value}")
    return value


def _select_gpu(gpu: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def _load_prompt_rows(path: str | Path, indices: list[int]) -> list[dict[str, str | int]]:
    prompt_path = _require_file(path, "prompt corpus")
    data = yaml.safe_load(prompt_path.read_text())
    if isinstance(data, dict):
        items = list(data.items())
    elif isinstance(data, list):
        items = [(f"{index:04d}", value) for index, value in enumerate(data)]
    else:
        raise TypeError(f"Unsupported prompt corpus type: {type(data).__name__}")
    rows = []
    for index in indices:
        if index < 0 or index >= len(items):
            raise IndexError(f"Prompt index {index} is outside the {len(items)}-prompt corpus")
        name, prompt = items[index]
        rows.append({"index": index, "name": str(name), "prompt": str(prompt)})
    return rows


def _load_bundle(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _synchronize() -> None:
    import torch

    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


def _patch_nunchaku_qwen_rope(transformer, text_sequence_length: int) -> None:
    """Bridge Nunchaku 1.3's pre-Diffusers-0.39 positional call signature."""

    import torch.nn as nn

    original = transformer.pos_embed

    class QwenEmbedRopeCompat(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.original = original
            self.text_sequence_length = text_sequence_length

        def forward(self, video_fhw, txt_seq_lens=None, *, device=None, max_txt_seq_len=None):
            if max_txt_seq_len is None:
                if isinstance(txt_seq_lens, (list, tuple)):
                    max_txt_seq_len = max(txt_seq_lens)
                else:
                    max_txt_seq_len = txt_seq_lens or self.text_sequence_length
            return self.original(video_fhw, device=device, max_txt_seq_len=max_txt_seq_len)

    compat = QwenEmbedRopeCompat()
    transformer.pos_embed = compat

    def update_text_sequence_length(_module, args) -> None:
        compat.text_sequence_length = args[0].shape[1]

    transformer.txt_norm.register_forward_pre_hook(update_text_sequence_length)


@app.command("encode", help="Encode selected prompts once with the official BF16 Qwen text encoder.")
def encode(
    prompt_index: Annotated[list[int] | None, typer.Option("--prompt-index", min=0)] = None,
    prompts: str = typer.Option(PROMPTS_DEFAULT),
    base_model: str = typer.Option(BASE_DEFAULT),
    output: str = typer.Option(str(OUTPUT_DEFAULT)),
    gpu: int = typer.Option(1, min=0),
    max_sequence_length: int = typer.Option(512, min=1, max=1024),
) -> None:
    _select_gpu(gpu)
    import torch
    from diffusers import QwenImagePipeline

    base = _require_dir(base_model, "Qwen-Image Diffusers model")
    output_dir = Path(output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_prompt_rows(prompts, prompt_index or [100])
    started = time.perf_counter()
    pipe = QwenImagePipeline.from_pretrained(
        str(base),
        transformer=None,
        vae=None,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    pipe.text_encoder.to("cuda:0")
    positives = [str(row["prompt"]) + POSITIVE_MAGIC for row in rows]
    negatives = [NEGATIVE_PROMPT] * len(rows)
    with torch.inference_mode():
        positive_embeds, positive_mask = pipe.encode_prompt(
            positives,
            device=torch.device("cuda:0"),
            max_sequence_length=max_sequence_length,
        )
        negative_embeds, negative_mask = pipe.encode_prompt(
            negatives,
            device=torch.device("cuda:0"),
            max_sequence_length=max_sequence_length,
        )
    bundle = {
        "base_model": str(base),
        "prompt_corpus": str(Path(prompts).expanduser().resolve()),
        "positive_magic": POSITIVE_MAGIC,
        "negative_prompt": NEGATIVE_PROMPT,
        "max_sequence_length": max_sequence_length,
        "rows": rows,
        "positive_embeds": positive_embeds.detach().cpu(),
        "positive_mask": None if positive_mask is None else positive_mask.detach().cpu(),
        "negative_embeds": negative_embeds.detach().cpu(),
        "negative_mask": None if negative_mask is None else negative_mask.detach().cpu(),
        "encode_seconds": time.perf_counter() - started,
    }
    bundle_path = output_dir / "prompt-embeddings.pt"
    torch.save(bundle, bundle_path)
    (output_dir / "prompts.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"embeddings": str(bundle_path), "rows": rows, "seconds": bundle["encode_seconds"]}, indent=2))
    del pipe, bundle, positive_embeds, negative_embeds
    gc.collect()
    torch.cuda.empty_cache()


def _render_bundle(pipe, bundle: dict, output_dir: Path, *, width: int, height: int, steps: int, cfg: float) -> dict:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    times = []
    execution_device = pipe._execution_device
    for index, row in enumerate(bundle["rows"]):
        positive_mask = bundle["positive_mask"]
        negative_mask = bundle["negative_mask"]
        generator = torch.Generator(device="cpu").manual_seed(hash_str_to_int(str(row["name"])))
        _synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            latent = pipe(
                prompt_embeds=bundle["positive_embeds"][index : index + 1].to(execution_device),
                prompt_embeds_mask=(
                    None if positive_mask is None else positive_mask[index : index + 1].to(execution_device)
                ),
                negative_prompt_embeds=bundle["negative_embeds"][index : index + 1].to(execution_device),
                negative_prompt_embeds_mask=(
                    None if negative_mask is None else negative_mask[index : index + 1].to(execution_device)
                ),
                generator=generator,
                width=width,
                height=height,
                num_inference_steps=steps,
                true_cfg_scale=cfg,
                output_type="latent",
            ).images
        _synchronize()
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        torch.save(latent.detach().cpu(), output_dir / f"{row['name']}.pt")
        print(json.dumps({"name": row["name"], "seconds": elapsed}), flush=True)
    return {
        "samples": len(times),
        "seconds": sum(times),
        "mean_seconds": sum(times) / len(times),
        "per_sample_seconds": times,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": cfg,
    }


@app.command("render-bf16", help="Render BF16 latents with two-GPU sharding and disk-offloaded overflow blocks.")
def render_bf16(
    output: str = typer.Option(str(OUTPUT_DEFAULT)),
    base_model: str = typer.Option(BASE_DEFAULT),
    width: int = typer.Option(512, min=64),
    height: int = typer.Option(512, min=64),
    steps: int = typer.Option(30, min=1),
    cfg: float = typer.Option(4.0, min=0.0),
    gpu0_memory: str = typer.Option("13GiB"),
    gpu1_memory: str = typer.Option("21GiB"),
    cpu_memory: str = typer.Option("1GiB"),
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    import torch
    from diffusers import QwenImagePipeline, QwenImageTransformer2DModel

    base = _require_dir(base_model, "Qwen-Image Diffusers model")
    output_dir = Path(output).expanduser().resolve()
    bundle = _load_bundle(output_dir / "prompt-embeddings.pt")
    offload_dir = output_dir / "bf16-offload"
    offload_dir.mkdir(parents=True, exist_ok=True)
    load_started = time.perf_counter()
    transformer = QwenImageTransformer2DModel.from_pretrained(
        str(base / "transformer"),
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        max_memory={0: gpu0_memory, 1: gpu1_memory, "cpu": cpu_memory},
        offload_folder=str(offload_dir),
        offload_state_dict=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    pipe = QwenImagePipeline.from_pretrained(
        str(base),
        transformer=transformer,
        text_encoder=None,
        tokenizer=None,
        vae=None,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    report = _render_bundle(pipe, bundle, output_dir / "latents/bf16", width=width, height=height, steps=steps, cfg=cfg)
    report.update(
        load_seconds=time.perf_counter() - load_started - report["seconds"],
        device_map={key: str(value) for key, value in transformer.hf_device_map.items()},
        gpu0_memory=gpu0_memory,
        gpu1_memory=gpu1_memory,
        cpu_memory=cpu_memory,
    )
    (output_dir / "bf16-timings.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


@app.command("render-int4", help="Render matching latents with the published rank-32 Nunchaku checkpoint.")
def render_int4(
    output: str = typer.Option(str(OUTPUT_DEFAULT)),
    base_model: str = typer.Option(BASE_DEFAULT),
    checkpoint: str = typer.Option(NUNCHAKU_DEFAULT),
    gpu: int = typer.Option(1, min=0),
    width: int = typer.Option(512, min=64),
    height: int = typer.Option(512, min=64),
    steps: int = typer.Option(30, min=1),
    cfg: float = typer.Option(4.0, min=0.0),
) -> None:
    _select_gpu(gpu)
    import torch
    from diffusers import QwenImagePipeline
    from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel

    base = _require_dir(base_model, "Qwen-Image Diffusers model")
    checkpoint_path = _require_file(checkpoint, "Nunchaku Qwen-Image checkpoint")
    output_dir = Path(output).expanduser().resolve()
    bundle = _load_bundle(output_dir / "prompt-embeddings.pt")
    load_started = time.perf_counter()
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
        checkpoint_path,
        device="cuda:0",
        torch_dtype=torch.bfloat16,
        offload=False,
    )
    _patch_nunchaku_qwen_rope(transformer, bundle["positive_embeds"].shape[1])
    pipe = QwenImagePipeline.from_pretrained(
        str(base),
        transformer=transformer,
        text_encoder=None,
        tokenizer=None,
        vae=None,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    report = _render_bundle(pipe, bundle, output_dir / "latents/int4", width=width, height=height, steps=steps, cfg=cfg)
    report.update(load_seconds=time.perf_counter() - load_started - report["seconds"], checkpoint=str(checkpoint_path))
    (output_dir / "int4-timings.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


@app.command("decode", help="Decode both latent sets through one shared official Qwen VAE.")
def decode(
    output: str = typer.Option(str(OUTPUT_DEFAULT)),
    base_model: str = typer.Option(BASE_DEFAULT),
    gpu: int = typer.Option(1, min=0),
) -> None:
    _select_gpu(gpu)
    import torch
    from diffusers import AutoencoderKLQwenImage, QwenImagePipeline

    base = _require_dir(base_model, "Qwen-Image Diffusers model")
    output_dir = Path(output).expanduser().resolve()
    bundle = _load_bundle(output_dir / "prompt-embeddings.pt")
    vae = AutoencoderKLQwenImage.from_pretrained(
        str(base / "vae"), torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True
    ).to("cuda:0")
    pipe = QwenImagePipeline.from_pretrained(
        str(base),
        transformer=None,
        text_encoder=None,
        tokenizer=None,
        vae=vae,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    width = json.loads((output_dir / "bf16-timings.json").read_text())["width"]
    height = json.loads((output_dir / "bf16-timings.json").read_text())["height"]
    for variant in ("bf16", "int4"):
        image_dir = output_dir / "images" / variant
        pixel_dir = output_dir / "pixels" / variant
        image_dir.mkdir(parents=True, exist_ok=True)
        pixel_dir.mkdir(parents=True, exist_ok=True)
        for row in bundle["rows"]:
            latent = torch.load(output_dir / "latents" / variant / f"{row['name']}.pt", weights_only=True).to(
                "cuda:0", torch.bfloat16
            )
            latent = pipe._unpack_latents(latent, height, width, pipe.vae_scale_factor)
            latents_mean = torch.tensor(vae.config.latents_mean, device="cuda:0", dtype=torch.bfloat16).view(
                1, vae.config.z_dim, 1, 1, 1
            )
            latents_std = 1.0 / torch.tensor(vae.config.latents_std, device="cuda:0", dtype=torch.bfloat16).view(
                1, vae.config.z_dim, 1, 1, 1
            )
            with torch.inference_mode():
                image = vae.decode(latent / latents_std + latents_mean, return_dict=False)[0][:, :, 0]
            pixels = pipe.image_processor.postprocess(image, output_type="pt").float().cpu()
            pil = pipe.image_processor.postprocess(image, output_type="pil")[0]
            torch.save(pixels, pixel_dir / f"{row['name']}.pt")
            pil.save(image_dir / f"{row['name']}.png")
    print(json.dumps({"images": str(output_dir / "images"), "samples": len(bundle["rows"])}, indent=2))


def _pixel_metrics(reference, candidate) -> dict[str, float]:
    delta = reference.float() - candidate.float()
    rmse = math.sqrt(delta.square().mean().item())
    return {
        "pixel_similarity": 1.0 - rmse,
        "rmse": rmse,
        "mae": delta.abs().mean().item(),
        "max_abs_error": delta.abs().max().item(),
    }


@app.command("report", help="Calculate raw RGB metrics and log the baseline to MLflow.")
def report(
    output: str = typer.Option(str(OUTPUT_DEFAULT)),
    threshold: float = typer.Option(0.99, min=0.0, max=1.0),
    track: bool = typer.Option(True, "--track/--no-track"),
    mlflow_uri: str = typer.Option("https://mlflow.appmana.com"),
    experiment_name: str = typer.Option("anima-aesthetic-v1.1-svdquant"),
    run_name: str = typer.Option("qwen-image-r32-pixel-baseline"),
    mlflow_run_id: str = typer.Option("", help="Explicit MLflow run to resume."),
) -> None:
    import torch

    output_dir = Path(output).expanduser().resolve()
    bundle = _load_bundle(output_dir / "prompt-embeddings.pt")
    bf16_timing = json.loads((output_dir / "bf16-timings.json").read_text())
    int4_timing = json.loads((output_dir / "int4-timings.json").read_text())
    samples = []
    for row in bundle["rows"]:
        reference = torch.load(output_dir / "pixels/bf16" / f"{row['name']}.pt", weights_only=True)
        candidate = torch.load(output_dir / "pixels/int4" / f"{row['name']}.pt", weights_only=True)
        samples.append({**row, **_pixel_metrics(reference, candidate)})
    similarities = [sample["pixel_similarity"] for sample in samples]
    report_data = {
        "model": "Qwen/Qwen-Image",
        "quantization": "published-nunchaku-svdquant-w4a4-r32",
        "acceptance_threshold": threshold,
        "accepted": min(similarities) >= threshold,
        "acceptance_rate": sum(value >= threshold for value in similarities) / len(similarities),
        "mean_pixel_similarity": sum(similarities) / len(similarities),
        "min_pixel_similarity": min(similarities),
        "mean_rmse": sum(sample["rmse"] for sample in samples) / len(samples),
        "mean_mae": sum(sample["mae"] for sample in samples) / len(samples),
        "max_abs_error": max(sample["max_abs_error"] for sample in samples),
        "bf16_seconds": bf16_timing["seconds"],
        "int4_seconds": int4_timing["seconds"],
        "int4_vs_bf16_speedup": bf16_timing["seconds"] / int4_timing["seconds"],
        "samples": samples,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report_data, indent=2) + "\n")
    tracker = ExperimentTracker(
        enabled=track,
        tracking_uri=mlflow_uri,
        experiment_name=experiment_name,
        run_name=run_name,
        run_id=mlflow_run_id,
        tags={"stage": "qwen-baseline", "model": "qwen-image", "recipe.rank": 32},
    )
    with tracker:
        tracker.save_reference(output_dir / REFERENCE_FILENAME)
        report_data["mlflow_run_id"] = tracker.run_id
        metrics_path.write_text(json.dumps(report_data, indent=2) + "\n")
        tracker.log_params(
            {
                "base_model": "Qwen/Qwen-Image",
                "svd_rank": 32,
                "weight_dtype": "sint4",
                "activation_dtype": "sint4",
                "runtime": "nunchaku-published",
                "num_samples": len(samples),
                "steps": bf16_timing["steps"],
                "resolution": f"{bf16_timing['width']}x{bf16_timing['height']}",
                "cfg": bf16_timing["cfg"],
                "bf16_topology": "2gpu-balanced-with-overflow-offload",
            }
        )
        tracker.log_metrics(
            {
                "quality.pixel_similarity.mean": report_data["mean_pixel_similarity"],
                "quality.pixel_similarity.min": report_data["min_pixel_similarity"],
                "quality.rmse.mean": report_data["mean_rmse"],
                "quality.mae.mean": report_data["mean_mae"],
                "quality.max_abs_error": report_data["max_abs_error"],
                "quality.acceptance_threshold": threshold,
                "quality.acceptance_rate": report_data["acceptance_rate"],
                "performance.image.int4_vs_bf16_speedup": report_data["int4_vs_bf16_speedup"],
                "time.validation.bf16_seconds": report_data["bf16_seconds"],
                "time.validation.int4_seconds": report_data["int4_seconds"],
            }
        )
        tracker.log_artifact(metrics_path, artifact_path="qwen-baseline")
        tracker.log_artifacts(output_dir / "images/bf16", artifact_path="qwen-baseline/bf16")
        tracker.log_artifacts(output_dir / "images/int4", artifact_path="qwen-baseline/int4")
    print(json.dumps(report_data, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
