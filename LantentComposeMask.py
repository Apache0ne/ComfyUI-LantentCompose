import torch

class LatentInterpolateMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent1": ("LATENT",),
                "latent2": ("LATENT",),
                "ratio": ("FLOAT", {"default": 0.55, "min": 0.00, "max": 1.00}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "interpolate_latents"
    CATEGORY = "Latent/Advanced"
    
    def slerp(self, latent1, latent2, ratio):
        """Perform spherical linear interpolation between two latent tensors."""
        latent1 = latent1.to(torch.float32)
        latent2 = latent2.to(torch.float32)
        v1 = latent1.view(-1)
        v2 = latent2.view(-1)
        norm1 = torch.norm(v1) + 1e-8  # Add small epsilon to avoid division by zero
        norm2 = torch.norm(v2) + 1e-8
        dot = torch.dot(v1, v2) / (norm1 * norm2)
        dot = torch.clamp(dot, -1.0, 1.0)  # Prevent numerical errors
        theta = torch.acos(dot)
        if torch.abs(theta) < 1e-6:
            # If angle is near zero, use linear interpolation
            return (1.0 - ratio) * latent1 + ratio * latent2
        else:
            sin_theta = torch.sin(theta)
            factor1 = torch.sin((1.0 - ratio) * theta) / sin_theta
            factor2 = torch.sin(ratio * theta) / sin_theta
            return factor1 * latent1 + factor2 * latent2
        
    def interpolate_latents(self, latent1, latent2, ratio, mask=None):
        """Interpolate between latents with an optional mask."""
        # Extract latent tensors from input dictionaries
        latent1_tensor = latent1["samples"]
        latent2_tensor = latent2["samples"]
        
        # Compute global slerp interpolation
        interpolated_global = self.slerp(latent1_tensor, latent2_tensor, ratio)
        
        if mask is not None:
            # Prepare the mask
            if mask.dim() == 3:  # [batch, height, width]
                mask = mask.unsqueeze(1)  # Add channel dimension: [batch, 1, height, width]
            elif mask.dim() == 4 and mask.shape[1] == 1:  # Already [batch, 1, height, width]
                pass
            else:
                raise ValueError("Mask must be [batch, height, width] or [batch, 1, height, width]")
            
            # Get latent spatial dimensions
            latent_height, latent_width = latent1_tensor.shape[2], latent1_tensor.shape[3]
            
            # Resize mask if its spatial dimensions don’t match the latent’s
            if mask.shape[2] != latent_height or mask.shape[3] != latent_width:
                mask = torch.nn.functional.interpolate(
                    mask, 
                    size=(latent_height, latent_width), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Ensure mask matches the latent’s device and dtype
            mask = mask.to(latent1_tensor.device, dtype=latent1_tensor.dtype)
            
            # Expand mask to match latent channels: [batch, channels, height, width]
            mask_expanded = mask.expand(-1, latent1_tensor.shape[1], -1, -1)
            
            # Blend: where mask=0 -> latent1, mask=1 -> interpolated_global
            interpolated_tensor = latent1_tensor * (1 - mask_expanded) + interpolated_global * mask_expanded
        else:
            # No mask: use global interpolation
            interpolated_tensor = interpolated_global
        
        # Return as a latent dictionary
        return ({"samples": interpolated_tensor},)