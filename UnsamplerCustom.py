import torch

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import comfy.model_management
import comfy.sample
import comfy.sampler_helpers

class UnsamplerCustom:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "end_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "normalize": (["disable", "enable"], ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    }}
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "unsampler"

    CATEGORY = "sampling"
        
    def unsampler(self, model, cfg, sampler_name, steps, end_at_step, scheduler, normalize, positive, negative, latent_image):
        normalize = normalize == "enable"
        device = comfy.model_management.get_torch_device()
        latent = latent_image
        latent_image = latent["samples"].to(device)

        end_at_step = min(end_at_step, steps-1)
        end_at_step = steps - end_at_step
        
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = comfy.sampler_helpers.prepare_mask(latent["noise_mask"], noise.shape, device)

        noise = noise.to(device)

        conds0 = {"positive": comfy.sampler_helpers.convert_cond(positive),
                  "negative": comfy.sampler_helpers.convert_cond(negative)}

        conds = {}
        for k in conds0:
            conds[k] = list(map(lambda a: a.copy(), conds0[k]))

        models, inference_memory = comfy.sampler_helpers.get_additional_models(conds, model.model_dtype())
        
        comfy.model_management.load_models_gpu([model] + models, model.memory_required(noise.shape) + inference_memory)

        sampler = comfy.samplers.KSampler(model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=1.0, model_options=model.model_options)

        sigmas = sampler.sigmas.flip(0) + 0.0001

        pbar = comfy.utils.ProgressBar(steps)
        
        latent_list = [latent_image.clone().cpu()]  # Start with initial latent on CPU
    
        def callback(step, x0, x, total_steps):
            latent_list.append(x.clone().cpu())  # Collect intermediate latent on CPU
            pbar.update_absolute(step + 1, total_steps)
        
        _ = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, force_full_denoise=False, denoise_mask=noise_mask, sigmas=sigmas, start_step=0, last_step=end_at_step, callback=callback)
        
        if normalize:
            for i in range(len(latent_list)):
                latent_list[i] -= latent_list[i].mean()
                latent_list[i] /= latent_list[i].std() + 1e-8
        
        # Replace torch.stack with torch.cat
        samples = torch.cat(latent_list, dim=0)  # [ (num_steps + 1) * batch_size, channels, height, width ]
        
        comfy.sampler_helpers.cleanup_additional_models(models)
        
        out = latent.copy()
        out["samples"] = samples
        return (out,)