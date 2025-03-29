from .LantentCompose import LatentInterpolate 
from .LantentComposeMask import LatentInterpolateMask

NODE_CLASS_MAPPINGS = {
    "LatentCompose": LatentInterpolate,
    "LatentComposeMask": LatentInterpolateMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LantentCompose": "Latent Compose Two",
    "LatentComposeMask": "Latent Compose Mask"
}
