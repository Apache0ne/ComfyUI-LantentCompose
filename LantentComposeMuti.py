import torch
import math

class LatentInterpolateMuti:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("LATENT",),
                "ratio": ("FLOAT", {"default": 0.55, "min": 0.00, "max": 1.00})
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "interpolate_latents"
    CATEGORY = "Latent/Advanced"
    
    def interpolate_latents(self, latents, ratio):
        """
        Interpolates through a batch of latents using slerp along a sequence.
        
        Parameters:
            latents (dict): Latent dictionary with 'samples' tensor of shape [B, C, H, W].
            ratio (float): Interpolation ratio (0.0 to 1.0) across the batch.
        
        Returns:
            tuple: A tuple containing the interpolated latent dictionary.
        """
        samples = latents['samples'].to(torch.float32)  # Shape: [B, C, H, W]
        B = samples.shape[0]
        
        if B == 1:
            return (latents,)
        
        # Compute interpolation position
        s = (B - 1) * ratio
        i = min(math.floor(s), B - 2)  # Ensure i <= B-2
        t = s - i  # Local ratio between 0 and 1
        
        # Extract the two latents to interpolate between
        L1 = samples[i]    # Shape: [C, H, W]
        L2 = samples[i + 1]  # Shape: [C, H, W]
        
        # Flatten for slerp computation
        v1 = L1.view(-1)
        v2 = L2.view(-1)
        
        # Compute norms with epsilon to avoid division by zero
        norm1 = torch.norm(v1) + 1e-8
        norm2 = torch.norm(v2) + 1e-8
        
        # Compute dot product and clamp
        dot = torch.dot(v1, v2) / (norm1 * norm2)
        dot = torch.clamp(dot, -1.0, 1.0)
        
        # Compute angle
        theta = torch.acos(dot)
        
        # Perform slerp
        if torch.abs(theta) < 1e-6:
            interpolated_tensor = (1.0 - t) * L1 + t * L2
        else:
            sin_theta = torch.sin(theta)
            factor1 = torch.sin((1.0 - t) * theta) / sin_theta
            factor2 = torch.sin(t * theta) / sin_theta
            interpolated_tensor = factor1 * L1 + factor2 * L2
        
        # Reshape to [1, C, H, W] and wrap in dictionary
        interpolated = {'samples': interpolated_tensor.unsqueeze(0)}
        return (interpolated,)