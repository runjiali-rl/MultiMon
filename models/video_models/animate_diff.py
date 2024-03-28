import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file



class AnimatediffPipeline():
    def __init__(self,
                 cache_dir:str = "/homes/55/runjia/scratch/diffusion_weights",
                 device:str = "cuda",
                 dtype:torch.dtype = torch.float16,
                 step:int = 8,):

        self.device = device
        self.dtype = dtype
        self.step = step
        self.repo = "ByteDance/AnimateDiff-Lightning"
        self.ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
        self.base = "emilianJR/epiCRealism"
        self.adapter = MotionAdapter().to(device, dtype)
        self.adapter.load_state_dict(load_file(hf_hub_download(self.repo ,self.ckpt),
                                               device=device))
        self.pipe = AnimateDiffPipeline.from_pretrained(self.base, motion_adapter=self.adapter,
                                                         cache_dir=cache_dir,
                                                         torch_dtype=dtype).to(device)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config,
                                                                    timestep_spacing="trailing",
                                                                    beta_schedule="linear")

    def generate(self,
                prompt:str,
                guidance_scale:float=1.0):
        output = self.pipe(prompt=prompt,
                            guidance_scale=guidance_scale,
                            num_inference_steps=self.step)
        return output.frames[0]



if __name__ == "__main__":
    model = AnimatediffPipeline()
    test_output = model.generate("an empty glass")
    export_to_gif(test_output, "outputs/animatediff.gif")
    