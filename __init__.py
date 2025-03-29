from .LantentCompose import LatentInterpolate 
from .LantentComposeMask import LatentInterpolateMask
from .UnsamplerCustom import UnsamplerCustom
from .LantentComposeMuti import LatentInterpolateMuti

NODE_CLASS_MAPPINGS = {
    "LatentCompose": LatentInterpolate,
    "LatentComposeMask": LatentInterpolateMask,
    "UnsamplerCustom": UnsamplerCustom,
    "LatentComposeMuti": LatentInterpolateMuti
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LantentCompose": "Latent Compose Two",
    "LatentComposeMask": "Latent Compose Mask",
    "UnsamplerCustom": "Unsampler Custom",
    "LatentComposeMuti": "Lantent Compose Muti"
}
