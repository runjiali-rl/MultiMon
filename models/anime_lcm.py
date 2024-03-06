import torch
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_video, export_to_gif


adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM",
                                        torch_dtype=torch.float16,
                                        cache_dir="/homes/55/runjia/scratch/diffusion_weights")
pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism",
                                           motion_adapter=adapter,
                                           torch_dtype=torch.float16,
                                           cache_dir="/homes/55/runjia/scratch/diffusion_weights")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config,
                                          beta_schedule="linear")

pipe.load_lora_weights("wangfuyun/AnimateLCM",
                       weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                       adapter_name="lcm-lora")
pipe.set_adapters(["lcm-lora"], [0.8])

pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()
pipe = pipe.to("cuda")



def animate_lcm_generate(prompt,
                         negative_prompt: str="bad quality, worse quality, low resolution",
                         num_frames: int=32,
                         guidance_scale: float=2.0,
                         num_inference_steps: int=50,
                         generator: torch.Generator=torch.Generator("cuda").manual_seed(0)):
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )
    frames = output.frames[0]
    return frames



if __name__ == "__main__":
    test_output = animate_lcm_generate("an empty glass", num_frames=32)
    
    export_to_gif(test_output, "outputs/animatelcm.gif")