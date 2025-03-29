import torch

class LatentInterpolate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent1": ("LATENT",),
                "latent2": ("LATENT",),
                "ratio": ("FLOAT", {"default": 0.55, "min": 0.00, "max": 1.00})
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "interpolate_latents"
    CATEGORY = "Latent/Advanced"
    
    def interpolate_latents(self, latent1, latent2, ratio):
        """
        Interpolates between two latent representations using spherical linear interpolation (slerp).
        
        Parameters:
            latent1 (dict): The first latent dictionary containing the tensor under 'samples'.
            latent2 (dict): The second latent dictionary containing the tensor under 'samples'.
            ratio (float): The interpolation ratio (0.0 gives latent1, 1.0 gives latent2).
        
        Returns:
            tuple: A tuple containing the interpolated latent dictionary.
        """
        # Extract tensors from the latent dictionaries and ensure they are float32
        latent1_tensor = latent1['samples'].to(torch.float32)
        latent2_tensor = latent2['samples'].to(torch.float32)
        
        # Flatten the tensors for dot product computation
        v1 = latent1_tensor.view(-1)
        v2 = latent2_tensor.view(-1)
        
        # Compute norms with a small epsilon to avoid division by zero
        norm1 = torch.norm(v1) + 1e-8
        norm2 = torch.norm(v2) + 1e-8
        
        # Compute the dot product and clamp to avoid numerical errors
        dot = torch.dot(v1, v2) / (norm1 * norm2)
        dot = torch.clamp(dot, -1.0, 1.0)
        
        # Compute the angle between the two vectors
        theta = torch.acos(dot)
        
        # If the angle is very small, use linear interpolation to avoid numerical instability
        if torch.abs(theta) < 1e-6:
            interpolated_tensor = (1.0 - ratio) * latent1_tensor + ratio * latent2_tensor
        else:
            # Perform spherical linear interpolation (slerp)
            sin_theta = torch.sin(theta)
            factor1 = torch.sin((1.0 - ratio) * theta) / sin_theta
            factor2 = torch.sin(ratio * theta) / sin_theta
            interpolated_tensor = factor1 * latent1_tensor + factor2 * latent2_tensor
        
        # Wrap the interpolated tensor in a dictionary and return as a tuple
        interpolated = {'samples': interpolated_tensor}
        return (interpolated,)