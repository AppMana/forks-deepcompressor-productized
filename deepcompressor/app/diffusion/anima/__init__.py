"""Productized SVDQuant support for ComfyUI's native Anima implementation."""

__all__ = ["AnimaModelStruct", "register_anima_struct_factories"]


def __getattr__(name: str):
    # Keep ComfyUI/CUDA initialization lazy so the CLI can apply --gpu before
    # model_management chooses its device.
    if name in __all__:
        from .struct import AnimaModelStruct, register_anima_struct_factories

        return {
            "AnimaModelStruct": AnimaModelStruct,
            "register_anima_struct_factories": register_anima_struct_factories,
        }[name]
    raise AttributeError(name)
