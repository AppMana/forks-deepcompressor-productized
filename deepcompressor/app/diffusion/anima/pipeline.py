"""Native ComfyUI loading, sampling, and calibration collection for Anima."""

from __future__ import annotations

import gc
import inspect
import json
import math
import os
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from comfy import model_management, sd, utils
from comfy.nodes import base_nodes
from datasets import Dataset
from PIL import Image

from deepcompressor.utils.common import hash_str_to_int, tree_map, tree_split

DEFAULT_NEGATIVE_PROMPT = "worst quality, low quality, artist name, blurry, jpeg artifacts, chromatic aberration"


def load_prompt_dataset(path: str | Path, limit: int = -1, offset: int = 0) -> Dataset:
    """Load prompts as a Hugging Face dataset with stable global indexes."""

    path = Path(path).expanduser().resolve()
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text())
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
    else:
        data = {f"{idx:04d}": line for idx, line in enumerate(path.read_text().splitlines()) if line.strip()}
    if isinstance(data, list):
        items = [(f"{idx:04d}", str(prompt)) for idx, prompt in enumerate(data)]
    elif isinstance(data, dict):
        items = [(str(name), str(prompt)) for name, prompt in data.items()]
    else:
        raise TypeError(f"Unsupported prompt dataset in {path}: {type(data).__name__}")
    items = items[offset:]
    items = items if limit < 0 else items[:limit]
    return Dataset.from_dict(
        {
            "name": [name for name, _ in items],
            "prompt": [prompt for _, prompt in items],
            "prompt_index": list(range(offset, offset + len(items))),
        }
    )


def load_prompts(path: str | Path, limit: int = -1, offset: int = 0) -> list[tuple[str, str]]:
    """Compatibility view used by single-process validation."""

    dataset = load_prompt_dataset(path, limit=limit, offset=offset)
    return list(zip(dataset["name"], dataset["prompt"], strict=True))


@dataclass
class AnimaComponents:
    model: tp.Any
    clip: tp.Any
    vae: tp.Any | None = None

    @property
    def diffusion_model(self) -> torch.nn.Module:
        return self.model.get_model_object("diffusion_model")

    def close(self) -> None:
        """Release ComfyUI patchers while module globals are still alive."""

        model_management.unload_all_models()
        for component in (self.vae, self.clip, self.model):
            patcher = getattr(component, "patcher", component)
            if patcher is not None and hasattr(patcher, "detach"):
                try:
                    patcher.detach(unpatch_all=False)
                except TypeError:
                    patcher.detach()
        self.vae = self.clip = self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_components(
    model_path: str | Path,
    text_encoder_path: str | Path,
    vae_path: str | Path | None = None,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> AnimaComponents:
    """Load the exact native ComfyUI model used by AppMana inference."""

    model_path = Path(model_path).expanduser().resolve()
    text_encoder_path = Path(text_encoder_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not text_encoder_path.is_file():
        raise FileNotFoundError(text_encoder_path)
    model = sd.load_diffusion_model(str(model_path), model_options={"dtype": dtype})
    if model is None:
        raise RuntimeError(f"ComfyUI could not recognize diffusion model {model_path}")
    clip = sd.load_clip(
        ckpt_paths=[str(text_encoder_path)],
        embedding_directory=[],
        clip_type=sd.CLIPType.STABLE_DIFFUSION,
    )
    vae = None
    if vae_path is not None:
        vae_path = Path(vae_path).expanduser().resolve()
        if not vae_path.is_file():
            raise FileNotFoundError(vae_path)
        vae_state, metadata = utils.load_torch_file(str(vae_path), return_metadata=True)
        vae = sd.VAE(sd=vae_state, metadata=metadata, ckpt_name=vae_path.name)
        vae.throw_exception_if_invalid()
    return AnimaComponents(model=model, clip=clip, vae=vae)


def load_diffusion_model(
    model_path: str | Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> AnimaComponents:
    """Load only the denoiser for cache-driven PTQ/export."""

    model_path = Path(model_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model = sd.load_diffusion_model(str(model_path), model_options={"dtype": dtype})
    if model is None:
        raise RuntimeError(f"ComfyUI could not recognize diffusion model {model_path}")
    return AnimaComponents(model=model, clip=None, vae=None)


def encode_conditioning(clip: tp.Any, prompt: str) -> tp.Any:
    tokens = clip.tokenize(prompt)
    return clip.encode_from_tokens_scheduled(tokens)


@torch.inference_mode()
def sample_latent(
    components: AnimaComponents,
    prompt: str,
    *,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    seed: int,
    width: int = 512,
    height: int = 512,
    steps: int = 30,
    cfg: float = 4.0,
    sampler: str = "er_sde",
    scheduler: str = "simple",
) -> torch.Tensor:
    """Run the official Anima Aesthetic recipe through native ComfyUI."""

    positive = encode_conditioning(components.clip, prompt)
    negative = encode_conditioning(components.clip, negative_prompt)
    latent = base_nodes.EmptyLatentImage().generate(width, height, 1)[0]
    sampled = base_nodes.common_ksampler(
        components.model,
        seed,
        steps,
        cfg,
        sampler,
        scheduler,
        positive,
        negative,
        latent,
        denoise=1.0,
    )[0]
    return sampled["samples"].detach().cpu()


@torch.inference_mode()
def decode_latent(vae: tp.Any, latent: torch.Tensor) -> torch.Tensor:
    if vae is None:
        raise RuntimeError("A VAE is required to decode pixel-space validation images")
    return vae.decode(latent).detach().cpu().clamp_(0, 1)


def save_image(image: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = image.detach().cpu()
    while image.ndim > 3:
        if image.shape[0] != 1:
            raise ValueError(f"Expected singleton batch/time dimensions for an image, got {tuple(image.shape)}")
        image = image[0]
    if image.ndim != 3 or image.shape[-1] not in {1, 3, 4}:
        raise ValueError(f"Expected HWC image pixels, got {tuple(image.shape)}")
    array = image.mul(255).round().to(torch.uint8).numpy()
    Image.fromarray(array).save(path)


class AnimaCollectHook:
    """Split every denoiser invocation into per-guidance calibration records."""

    def __init__(self) -> None:
        self.caches: list[dict[str, tp.Any]] = []

    def __call__(
        self,
        module: torch.nn.Module,
        input_args: tuple[tp.Any, ...],
        input_kwargs: dict[str, tp.Any],
    ) -> None:
        arguments = inspect.signature(module.forward).bind_partial(*input_args, **input_kwargs).arguments
        extra_kwargs = dict(arguments.pop("kwargs", {}))
        for optional in ("fps", "padding_mask"):
            value = arguments.get(optional)
            if value is not None:
                extra_kwargs[optional] = value
        # Comfy's runtime-only patch UUIDs/callbacks are neither serializable
        # as a weights-only cache nor needed when replaying the plain denoiser.
        extra_kwargs["transformer_options"] = {}
        try:
            x = arguments["x"]
            timesteps = arguments["timesteps"]
            context = arguments["context"]
        except KeyError as error:
            raise RuntimeError(f"Unexpected Anima forward arguments: {tuple(arguments)}") from error
        cache = {
            "input_args": [x],
            "input_kwargs": {
                "timesteps": timesteps,
                "context": context,
                **extra_kwargs,
            },
        }
        cache = tree_map(lambda tensor: tensor.detach().cpu(), cache)
        self.caches.extend(tree_split(cache))


@torch.inference_mode()
def collect_calibration_dataset(
    components: AnimaComponents,
    prompts: Dataset | list[tuple[str, str]],
    output_dir: str | Path,
    *,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    width: int = 512,
    height: int = 512,
    steps: int = 30,
    cfg: float = 4.0,
    sampler: str = "er_sde",
    scheduler: str = "simple",
    resume: bool = False,
) -> dict[str, tp.Any]:
    """Collect full denoising trajectories and one final latent per prompt."""

    output_dir = Path(output_dir).expanduser().resolve()
    cache_dir = output_dir / "caches"
    calibration_dir = output_dir / "calibration"
    latent_dir = output_dir / "latents"
    cache_dir.mkdir(parents=True, exist_ok=True)
    calibration_dir.mkdir(parents=True, exist_ok=True)
    latent_dir.mkdir(parents=True, exist_ok=True)
    collector = AnimaCollectHook()
    handle = components.diffusion_model.register_forward_pre_hook(collector, with_kwargs=True)
    records: list[dict[str, tp.Any]] = []
    prompt_rows: list[dict[str, tp.Any]] = []
    try:
        for local_prompt_idx, item in enumerate(prompts):
            if isinstance(item, dict):
                name = str(item["name"])
                prompt = str(item["prompt"])
                prompt_idx = int(item["prompt_index"])
            else:
                name, prompt = item
                prompt_idx = local_prompt_idx
            prompt_rows.append({"name": name, "prompt": prompt, "prompt_index": prompt_idx})
            seed = hash_str_to_int(name)
            selected_step = prompt_idx % steps
            if resume and (latent_dir / f"{name}.pt").is_file():
                num_guidances = 0
                while (cache_dir / f"{name}-00000-{num_guidances}.pt").is_file():
                    num_guidances += 1
                complete = num_guidances > 0 and all(
                    (cache_dir / f"{name}-{step:05d}-{guidance}.pt").is_file()
                    for step in range(steps)
                    for guidance in range(num_guidances)
                )
                if complete:
                    selected_guidance = (prompt_idx // steps) % num_guidances
                    selected_name = f"{name}-{selected_step:05d}-{selected_guidance}.pt"
                    destination = calibration_dir / selected_name
                    if not destination.exists():
                        os.link(cache_dir / selected_name, destination)
                    for step in range(steps):
                        for guidance in range(num_guidances):
                            records.append(
                                {
                                    "name": name,
                                    "prompt": prompt,
                                    "prompt_index": prompt_idx,
                                    "seed": seed,
                                    "step": step,
                                    "guidance": guidance,
                                    "cache_path": str(Path("caches") / f"{name}-{step:05d}-{guidance}.pt"),
                                    "latent_path": str(Path("latents") / f"{name}.pt"),
                                    "selected_for_calibration": step == selected_step and guidance == selected_guidance,
                                }
                            )
                    continue
            latent = sample_latent(
                components,
                prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
                sampler=sampler,
                scheduler=scheduler,
            )
            torch.save(latent, latent_dir / f"{name}.pt")
            if not collector.caches or len(collector.caches) % steps:
                raise RuntimeError(
                    f"Prompt {name} produced {len(collector.caches)} calibration records for {steps} steps"
                )
            num_guidances = len(collector.caches) // steps
            for step in range(steps):
                for guidance in range(num_guidances):
                    cache = collector.caches[step * num_guidances + guidance]
                    cache.update(
                        filename=name,
                        prompt=prompt,
                        prompt_index=prompt_idx,
                        seed=seed,
                        step=step,
                        guidance=guidance,
                    )
                    torch.save(cache, cache_dir / f"{name}-{step:05d}-{guidance}.pt")
                    records.append(
                        {
                            "name": name,
                            "prompt": prompt,
                            "prompt_index": prompt_idx,
                            "seed": seed,
                            "step": step,
                            "guidance": guidance,
                            "cache_path": str(Path("caches") / f"{name}-{step:05d}-{guidance}.pt"),
                            "latent_path": str(Path("latents") / f"{name}.pt"),
                            "selected_for_calibration": False,
                        }
                    )
            # DeepCompressor's released recipes calibrate on a bounded set of
            # denoiser records, not every step of every generated image. Keep
            # every trajectory for auditability, while selecting one
            # stratified timestamp/guidance record from each of the 100
            # prompts for the memory-bounded PTQ objective.
            selected_guidance = (prompt_idx // steps) % num_guidances
            selected_record = next(
                record
                for record in reversed(records)
                if record["name"] == name
                and record["step"] == selected_step
                and record["guidance"] == selected_guidance
            )
            selected_record["selected_for_calibration"] = True
            source = cache_dir / f"{name}-{selected_step:05d}-{selected_guidance}.pt"
            destination = calibration_dir / source.name
            if not destination.exists():
                os.link(source, destination)
            collector.caches.clear()
    finally:
        handle.remove()
    metadata = {
        "model": "anima-aesthetic-v1.1",
        "num_prompts": len(prompts),
        "num_records": len(records),
        "num_calibration_records": len(prompts),
        "records_per_prompt": len(records) // max(len(prompts), 1),
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": cfg,
        "sampler": sampler,
        "scheduler": scheduler,
        "negative_prompt": negative_prompt,
        "prompts": prompt_rows,
        "records": records,
    }
    return metadata


def save_calibration_manifest(
    output_dir: str | Path,
    shard_metadata: list[dict[str, tp.Any]],
) -> tuple[Path, dict[str, tp.Any]]:
    """Merge Accelerate rank metadata into one portable Arrow manifest."""

    output_dir = Path(output_dir).expanduser().resolve()
    records = sorted(
        (record for shard in shard_metadata for record in shard["records"]),
        key=lambda record: (record["prompt_index"], record["step"], record["guidance"]),
    )
    prompts = sorted(
        (prompt for shard in shard_metadata for prompt in shard["prompts"]),
        key=lambda prompt: prompt["prompt_index"],
    )
    manifest_path = output_dir / "hf_dataset"
    if manifest_path.exists():
        raise FileExistsError(f"Hugging Face dataset already exists: {manifest_path}")
    Dataset.from_list(records).save_to_disk(manifest_path)
    template = shard_metadata[0]
    metadata = {
        key: value
        for key, value in template.items()
        if key not in {"num_prompts", "num_records", "num_calibration_records", "prompts", "records"}
    }
    metadata.update(
        num_prompts=len(prompts),
        num_records=len(records),
        num_calibration_records=sum(record["selected_for_calibration"] for record in records),
        prompts=prompts,
        hf_dataset=str(manifest_path),
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return manifest_path, metadata


def raw_pixel_similarity(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    """Report raw RGB agreement; no perceptual or feature-space proxies."""

    if reference.shape != candidate.shape:
        raise ValueError(f"Pixel tensor shapes differ: {reference.shape} != {candidate.shape}")
    delta = reference.float() - candidate.float()
    mse = delta.square().mean().item()
    rmse = math.sqrt(mse)
    return {
        "pixel_similarity": 1.0 - rmse,
        "rmse": rmse,
        "mae": delta.abs().mean().item(),
        "max_abs_error": delta.abs().max().item(),
    }
