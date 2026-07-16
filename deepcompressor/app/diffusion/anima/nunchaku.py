"""Pack and run Anima SVDQuant weights with Nunchaku's fused INT4 kernels."""

from __future__ import annotations

import json
import typing as tp
from pathlib import Path

import safetensors.torch
import torch
import torch.nn as nn
from comfy import quant_ops
from einops import rearrange
from nunchaku.models.linear import SVDQW4A4Linear
from nunchaku.ops.fused import fused_gelu_mlp

from deepcompressor.backend.nunchaku.convert import convert_to_nunchaku_w4x4y16_linear_state_dict
from deepcompressor.nn.patch.linear import ShiftedLinear

from .struct import AnimaModelStruct

__all__ = ["apply_nunchaku_checkpoint", "export_nunchaku_checkpoint"]


def _concat_scales(scales: list[torch.Tensor], modules: list[nn.Linear]) -> torch.Tensor:
    if all(scale.numel() == 1 for scale in scales):
        return torch.cat(
            [
                scale.reshape(1).expand(module.out_features).reshape(module.out_features, 1, 1, 1)
                for scale, module in zip(scales, modules, strict=True)
            ],
            dim=0,
        )
    return torch.cat(scales, dim=0)


def _pack_group(
    modules: list[nn.Linear],
    names: list[str],
    *,
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, dict[str, torch.Tensor]],
    shift: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if not modules or len(modules) != len(names):
        raise ValueError("A packed group needs matching modules and names")
    device = modules[0].weight.device
    scales = [scale_dict[f"{name}.weight.scale.0"].to(device) for name in names]
    weight = torch.cat([module.weight.detach() for module in modules], dim=0)
    scale = _concat_scales(scales, modules)
    bias = None
    if any(module.bias is not None for module in modules):
        bias = torch.cat(
            [
                module.bias.detach()
                if module.bias is not None
                else torch.zeros(module.out_features, dtype=module.weight.dtype, device=module.weight.device)
                for module in modules
            ]
        )
    branch = branch_dict.get(names[0])
    if branch is None:
        raise KeyError(f"Missing calibrated low-rank branch for {names[0]}")
    lora = (branch["a.weight"].to(device), branch["b.weight"].to(device))
    smooth = smooth_dict.get(names[0])
    if smooth is not None:
        smooth = smooth.to(device)
    packed = convert_to_nunchaku_w4x4y16_linear_state_dict(
        weight=weight,
        scale=scale,
        bias=bias,
        smooth=smooth,
        lora=lora,
        shift=shift,
        smooth_fused=False,
        float_point=False,
    )
    renamed = {
        "qweight": packed["qweight"],
        "wscales": packed["wscales"],
        "bias": packed["bias"],
        "smooth_factor_orig": packed["smooth_orig"],
        "smooth_factor": packed["smooth"],
        "proj_down": packed["lora_down"],
        "proj_up": packed["lora_up"],
    }
    return {key: value.detach().contiguous().cpu() for key, value in renamed.items()}


def export_nunchaku_checkpoint(
    model: AnimaModelStruct,
    quant_dir: str | Path,
    output_dir: str | Path,
    *,
    rank: int,
) -> tuple[Path, Path]:
    """Export current PTQ weights with the installed Nunchaku 1.3 field names."""

    quant_dir = Path(quant_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scale_dict = torch.load(quant_dir / "scale.pt", map_location="cpu", weights_only=True)
    smooth_dict = torch.load(quant_dir / "smooth.pt", map_location="cpu", weights_only=True)
    branch_dict = torch.load(quant_dir / "branch.pt", map_location="cpu", weights_only=True)
    state: dict[str, torch.Tensor] = {}
    groups: list[dict[str, tp.Any]] = []

    def add_group(role: str, block_index: int, modules: list[nn.Linear], names: list[str], **kwargs) -> None:
        prefix = f"blocks.{block_index}.{role}"
        packed = _pack_group(
            modules,
            names,
            scale_dict=scale_dict,
            smooth_dict=smooth_dict,
            branch_dict=branch_dict,
            **kwargs,
        )
        for key, value in packed.items():
            state[f"{prefix}.{key}"] = value
        groups.append(
            {
                "prefix": prefix,
                "role": role,
                "block": block_index,
                "in_features": modules[0].in_features,
                "out_features": sum(module.out_features for module in modules),
                "rank": rank,
                "act_unsigned": role == "mlp_fc2",
                "source_names": names,
            }
        )

    for index, block in enumerate(model.block_structs):
        self_attn, cross_attn = block.attn_structs
        add_group("self_qkv", index, self_attn.qkv_proj, self_attn.qkv_proj_names)
        add_group("self_out", index, [self_attn.o_proj], [self_attn.o_proj_name])
        add_group("cross_q", index, [cross_attn.q_proj], [cross_attn.q_proj_name])
        add_group("cross_kv", index, cross_attn.add_qkv_proj, cross_attn.add_qkv_proj_names)
        add_group("cross_out", index, [cross_attn.o_proj], [cross_attn.o_proj_name])
        ffn = block.ffn_struct
        add_group("mlp_fc1", index, ffn.up_projs, ffn.up_proj_names)
        down_layer = block.module.mlp.layer2
        shift = down_layer.shift if isinstance(down_layer, ShiftedLinear) else None
        add_group("mlp_fc2", index, ffn.down_projs, ffn.down_proj_names, shift=shift)

    weights_path = output_dir / "anima-aesthetic-v1.1-svdquant-int4.safetensors"
    manifest_path = output_dir / "anima-aesthetic-v1.1-svdquant-int4.json"
    safetensors.torch.save_file(
        state,
        weights_path,
        metadata={
            "architecture": "comfyui-anima",
            "quantization": "svdquant-w4a4-int4",
            "low_rank_dtype": "bfloat16",
            "rank": str(rank),
            "runtime": "nunchaku",
        },
    )
    manifest = {
        "format_version": 1,
        "architecture": "comfyui-anima",
        "base_model": "anima-aesthetic-v1.1",
        "quantization": "svdquant-w4a4-int4",
        "rank": rank,
        "group_size": 64,
        "low_rank_dtype": "bfloat16",
        "weights": weights_path.name,
        "groups": groups,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return weights_path, manifest_path


def _load_linear(
    state: dict[str, torch.Tensor],
    group: dict[str, tp.Any],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> SVDQW4A4Linear:
    prefix = group["prefix"] + "."
    module = SVDQW4A4Linear(
        in_features=group["in_features"],
        out_features=group["out_features"],
        rank=group["rank"],
        bias=True,
        precision="int4",
        act_unsigned=group["act_unsigned"],
        torch_dtype=dtype,
        device=device,
    )
    module.load_state_dict({key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)})
    return module


class NunchakuAnimaAttention(nn.Module):
    """Cosmos attention with fused SVDQ QKV/KV projection groups."""

    def __init__(
        self,
        original: nn.Module,
        first: SVDQW4A4Linear,
        second: SVDQW4A4Linear,
    ) -> None:
        super().__init__()
        self.is_selfattn = original.is_selfattn
        self.n_heads = original.n_heads
        self.head_dim = original.head_dim
        self.query_dim = original.query_dim
        self.context_dim = original.context_dim
        self.q_norm = original.q_norm
        self.k_norm = original.k_norm
        self.output_dropout = original.output_dropout
        self.attn_op = original.attn_op
        if self.is_selfattn:
            self.qkv_proj = first
        else:
            self.q_proj = first
            self.kv_proj = second
        self.output_proj = second if self.is_selfattn else None

    def set_output(self, output: SVDQW4A4Linear) -> None:
        self.output_proj = output

    def compute_qkv(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        rope_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.is_selfattn:
            q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        else:
            context = x if context is None else context
            q = self.q_proj(x)
            k, v = self.kv_proj(context).chunk(2, dim=-1)
        q, k, v = (
            rearrange(tensor, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim) for tensor in (q, k, v)
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.is_selfattn and rope_emb is not None:
            q, k = quant_ops.ck.apply_rope_split_half(q, k, rope_emb)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        rope_emb: torch.Tensor | None = None,
        transformer_options: dict | None = None,
    ) -> torch.Tensor:
        q, k, v = self.compute_qkv(x, context, rope_emb)
        result = self.attn_op(q, k, v, transformer_options=transformer_options or {})
        return self.output_dropout(self.output_proj(result))


class NunchakuAnimaFeedForward(nn.Module):
    """Shape-preserving wrapper around Nunchaku's fused INT4 GELU MLP."""

    def __init__(self, fc1: SVDQW4A4Linear, fc2: SVDQW4A4Linear) -> None:
        super().__init__()
        self.layer1 = fc1
        self.layer2 = fc2
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(1, -1, shape[-1])
        x = fused_gelu_mlp(x, self.layer1, self.layer2)
        return x.reshape(*shape[:-1], -1)


def apply_nunchaku_checkpoint(
    model: nn.Module,
    manifest_path: str | Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Replace Anima block projections with the exported fused Nunchaku modules."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest["architecture"] != "comfyui-anima":
        raise ValueError(f"Unsupported checkpoint architecture: {manifest['architecture']}")
    device = next(model.parameters()).device
    state = safetensors.torch.load_file(str(manifest_path.with_name(manifest["weights"])), device=str(device))
    groups = {(group["block"], group["role"]): group for group in manifest["groups"]}
    for index, block in enumerate(model.blocks):

        def linear(role: str, block_index: int = index) -> SVDQW4A4Linear:
            return _load_linear(state, groups[(block_index, role)], dtype=dtype, device=device)

        self_attn = NunchakuAnimaAttention(block.self_attn, linear("self_qkv"), linear("self_out"))
        cross_attn = NunchakuAnimaAttention(block.cross_attn, linear("cross_q"), linear("cross_kv"))
        cross_attn.set_output(linear("cross_out"))
        block.self_attn = self_attn
        block.cross_attn = cross_attn
        block.mlp = NunchakuAnimaFeedForward(linear("mlp_fc1"), linear("mlp_fc2"))
    return model
