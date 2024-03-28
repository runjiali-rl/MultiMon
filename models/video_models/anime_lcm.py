import torch
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_video, export_to_gif



class AnimateLCMPipeline():
    def __init__(self, cache_dir: str = "/homes/55/runjia/scratch/diffusion_weights"):


        self.adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM",
                                                torch_dtype=torch.float16,
                                                cache_dir=cache_dir)
        self.pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism",
                                                motion_adapter=self.adapter,
                                                torch_dtype=torch.float16,
                                                cache_dir=cache_dir)
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config,
                                                beta_schedule="linear")

        self.pipe.load_lora_weights("wangfuyun/AnimateLCM",
                            weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                            adapter_name="lcm-lora")
        self.pipe.set_adapters(["lcm-lora"], [0.8])

        self.pipe.enable_vae_slicing()
        self.pipe.enable_model_cpu_offload()
        self.pipe = self.pipe.to("cuda")



    def generate(self,
                prompt,
                negative_prompt: str="bad quality, worse quality, low resolution",
                num_frames: int=32,
                guidance_scale: float=2.0,
                num_inference_steps: int=50,
                generator: torch.Generator=torch.Generator("cuda").manual_seed(0)):
        output = self.pipe(
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
    model = AnimateLCMPipeline()
    test_output = model.generate("an empty glass", num_frames=32)
    export_to_gif(test_output, "outputs/animatelcm.gif")
