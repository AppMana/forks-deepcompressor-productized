"""DeepCompressor model structures for ComfyUI's native Anima DiT."""

from __future__ import annotations

import typing as tp
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field

import torch.nn as nn
from comfy.ldm.anima.model import Anima
from comfy.ldm.cosmos.predict2 import Attention, Block, GPT2FeedForward

from deepcompressor.nn.patch.linear import ShiftedLinear
from deepcompressor.nn.struct.attn import AttentionConfigStruct, FeedForwardConfigStruct
from deepcompressor.nn.struct.base import BaseModuleStruct
from deepcompressor.utils.common import join_name

from ..nn.struct import (
    DiffusionAttentionStruct,
    DiffusionBlockStruct,
    DiffusionFeedForwardStruct,
    DiffusionModelStruct,
    DiffusionModuleStruct,
    DiffusionTransformerBlockStruct,
)

__all__ = ["AnimaModelStruct", "register_anima_struct_factories"]


@dataclass(kw_only=True)
class AnimaAttentionStruct(DiffusionAttentionStruct):
    """Describe Cosmos/Anima attention using DeepCompressor's QKV abstraction."""

    module: Attention = field(repr=False, kw_only=False)

    def filter_kwargs(self, kwargs: dict) -> dict:
        return {
            "rope_emb": kwargs.get("rope_emb_L_1_1_D"),
            "transformer_options": kwargs.get("transformer_options", {}),
        }

    @staticmethod
    def _default_construct(
        module: Attention,
        /,
        parent: tp.Optional["AnimaTransformerBlockStruct"] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "AnimaAttentionStruct":
        if module.is_selfattn:
            q_proj, k_proj, v_proj = module.q_proj, module.k_proj, module.v_proj
            add_k_proj = add_v_proj = None
            k_proj_rname, v_proj_rname = "k_proj", "v_proj"
            add_k_proj_rname = add_v_proj_rname = ""
            add_hidden_size = 0
        else:
            q_proj, k_proj, v_proj = module.q_proj, None, None
            add_k_proj, add_v_proj = module.k_proj, module.v_proj
            k_proj_rname = v_proj_rname = ""
            add_k_proj_rname, add_v_proj_rname = "k_proj", "v_proj"
            add_hidden_size = module.context_dim
        config = AttentionConfigStruct(
            hidden_size=module.query_dim,
            add_hidden_size=add_hidden_size,
            inner_size=module.n_heads * module.head_dim,
            num_query_heads=module.n_heads,
            num_key_value_heads=module.n_heads,
            with_qk_norm=True,
            with_rope=module.is_selfattn,
        )
        return AnimaAttentionStruct(
            module=module,
            parent=parent,
            fname=fname,
            idx=idx,
            rname=rname,
            rkey=rkey,
            config=config,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            o_proj=module.output_proj,
            add_q_proj=None,
            add_k_proj=add_k_proj,
            add_v_proj=add_v_proj,
            add_o_proj=None,
            q=None,
            k=None,
            v=None,
            q_proj_rname="q_proj",
            k_proj_rname=k_proj_rname,
            v_proj_rname=v_proj_rname,
            o_proj_rname="output_proj",
            add_q_proj_rname="",
            add_k_proj_rname=add_k_proj_rname,
            add_v_proj_rname=add_v_proj_rname,
            add_o_proj_rname="",
            q_rname="",
            k_rname="",
            v_rname="",
        )


@dataclass(kw_only=True)
class AnimaFeedForwardStruct(DiffusionFeedForwardStruct):
    module: GPT2FeedForward = field(repr=False, kw_only=False)

    @staticmethod
    def _default_construct(
        module: GPT2FeedForward,
        /,
        parent: tp.Optional["AnimaTransformerBlockStruct"] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "AnimaFeedForwardStruct":
        up_proj = module.layer1
        down_layer = module.layer2
        if isinstance(down_layer, ShiftedLinear):
            down_proj = down_layer.linear
            down_proj_rname = "layer2.linear"
            act_type = "gelu_shifted"
        else:
            down_proj = down_layer
            down_proj_rname = "layer2"
            act_type = "gelu"
        assert isinstance(up_proj, nn.Linear)
        assert isinstance(down_proj, nn.Linear)
        config = FeedForwardConfigStruct(
            hidden_size=up_proj.weight.shape[1],
            intermediate_size=down_proj.weight.shape[1],
            intermediate_act_type=act_type,
        )
        return AnimaFeedForwardStruct(
            module=module,
            parent=parent,
            fname=fname,
            idx=idx,
            rname=rname,
            rkey=rkey,
            config=config,
            up_projs=[up_proj],
            down_projs=[down_proj],
            up_proj_rnames=["layer1"],
            down_proj_rnames=[down_proj_rname],
        )


@dataclass(kw_only=True)
class AnimaTransformerBlockStruct(DiffusionTransformerBlockStruct):
    module: Block = field(repr=False, kw_only=False)
    parent: tp.Optional["AnimaModelStruct"] = field(repr=False)
    attn_struct_cls: tp.ClassVar[type[AnimaAttentionStruct]] = AnimaAttentionStruct
    ffn_struct_cls: tp.ClassVar[type[AnimaFeedForwardStruct]] = AnimaFeedForwardStruct

    @staticmethod
    def _default_construct(
        module: Block,
        /,
        parent: tp.Optional["AnimaModelStruct"] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "AnimaTransformerBlockStruct":
        # These LayerNorms are followed by timestep-dependent affine modulation,
        # so they must be described as AdaNorm rather than fuseable LayerNorm.
        return AnimaTransformerBlockStruct(
            module=module,
            parent=parent,
            fname=fname,
            idx=idx,
            rname=rname,
            rkey=rkey,
            parallel=False,
            pre_attn_norms=[module.layer_norm_self_attn, module.layer_norm_cross_attn],
            pre_attn_add_norms=[None, None],
            attns=[module.self_attn, module.cross_attn],
            pre_ffn_norm=module.layer_norm_mlp,
            ffn=module.mlp,
            pre_add_ffn_norm=None,
            add_ffn=None,
            pre_attn_norm_rnames=["layer_norm_self_attn", "layer_norm_cross_attn"],
            # Non-empty synthetic names satisfy the generic cross-attention
            # structure without pretending the external context has a norm.
            pre_attn_add_norm_rnames=["self_attn.context_norm", "cross_attn.context_norm"],
            attn_rnames=["self_attn", "cross_attn"],
            pre_ffn_norm_rname="layer_norm_mlp",
            ffn_rname="mlp",
            pre_add_ffn_norm_rname="",
            add_ffn_rname="",
            norm_type="ada_norm",
            add_norm_type="ada_norm",
        )


@dataclass(kw_only=True)
class AnimaModelStruct(DiffusionModelStruct):
    """Expose only Anima's main 28 DiT blocks to SVDQuant.

    The input/output embeddings, AdaLN projections, final layer, and LLM adapter
    deliberately stay BF16. This mirrors Nunchaku's outlier-safe policy and
    prevents the text adapter from being altered.
    """

    module: Anima = field(repr=False, kw_only=False)
    blocks: nn.ModuleList = field(repr=False)
    blocks_rname: str = "blocks"
    block_names: list[str] = field(init=False, repr=False)
    _block_structs: list[AnimaTransformerBlockStruct] = field(init=False, repr=False)

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def block_structs(self) -> list[AnimaTransformerBlockStruct]:
        return self._block_structs

    def __post_init__(self) -> None:
        super().__post_init__()
        self.pre_module_structs = OrderedDict()
        self.post_module_structs = OrderedDict()
        self.block_names = [join_name(self.name, f"{self.blocks_rname}.{idx}") for idx in range(len(self.blocks))]
        self._block_structs = [
            AnimaTransformerBlockStruct.construct(
                block,
                parent=self,
                fname="block",
                rname=f"{self.blocks_rname}.{idx}",
                rkey="",
                idx=idx,
            )
            for idx, block in enumerate(self.blocks)
        ]

    def get_prev_module_keys(self) -> tuple[str, ...]:
        return ()

    def get_post_module_keys(self) -> tuple[str, ...]:
        return ()

    def _get_iter_block_activations_args(
        self, **input_kwargs
    ) -> tuple[list[nn.Module], list[DiffusionBlockStruct | DiffusionModuleStruct], list[bool], list[bool]]:
        return (
            list(self.blocks),
            list(self._block_structs),
            [False] * len(self.blocks),
            [False, *([True] * (len(self.blocks) - 1))],
        )

    @staticmethod
    def _default_construct(
        module: Anima,
        /,
        parent: tp.Optional[BaseModuleStruct] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "AnimaModelStruct":
        return AnimaModelStruct(
            module=module,
            parent=parent,
            fname=fname,
            idx=idx,
            rname=rname,
            rkey=rkey,
            blocks=module.blocks,
        )

    @classmethod
    def _get_default_key_map(cls) -> dict[str, set[str]]:
        block_map = AnimaTransformerBlockStruct._get_default_key_map()
        key_map: dict[str, set[str]] = defaultdict(set)
        for rkey, keys in block_map.items():
            key_map[rkey].update(keys)
        # Preserve the generic skip vocabulary used by the published configs.
        for skipped in (
            "embed",
            "resblock",
            "resblock_shortcut",
            "resblock_time_proj",
            "transformer_proj_in",
            "transformer_proj_out",
            "down_sample",
            "up_sample",
        ):
            key_map[skipped].add(skipped)
        return {key: values for key, values in key_map.items() if values}


def register_anima_struct_factories() -> None:
    """Register exact-type factories once for the installed ComfyUI classes."""

    registrations = (
        (DiffusionModelStruct, Anima, AnimaModelStruct._default_construct),
        (DiffusionTransformerBlockStruct, Block, AnimaTransformerBlockStruct._default_construct),
        (AnimaTransformerBlockStruct, Block, AnimaTransformerBlockStruct._default_construct),
        (DiffusionAttentionStruct, Attention, AnimaAttentionStruct._default_construct),
        (AnimaAttentionStruct, Attention, AnimaAttentionStruct._default_construct),
        (DiffusionFeedForwardStruct, GPT2FeedForward, AnimaFeedForwardStruct._default_construct),
        (AnimaFeedForwardStruct, GPT2FeedForward, AnimaFeedForwardStruct._default_construct),
    )
    for struct_cls, module_cls, factory in registrations:
        factories = getattr(struct_cls, "_factories", {})
        if module_cls not in factories:
            struct_cls.register_factory(module_cls, factory)


register_anima_struct_factories()
