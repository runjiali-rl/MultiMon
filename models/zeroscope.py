import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video, export_to_gif
from PIL import Image
import numpy as np

pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w",
                                         torch_dtype=torch.float16,
                                         cache_dir="/homes/55/runjia/scratch/diffusion_weights")
pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload()

# memory optimization
pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
pipe.enable_vae_slicing()

def zeroscope_generate(prompt,
                       negative_prompt: str="bad quality, worse quality, low resolution",
                       num_frames: int=32,
                       num_inference_steps: int=50,
                       generator: torch.Generator=torch.Generator("cuda").manual_seed(0)):
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )
    frames = output.frames[0]
    video_frames_PIL_list = [Image.fromarray((frame*255).astype(np.uint8)) for frame in frames]
    return video_frames_PIL_list



if __name__ == "__main__":
    test_output = zeroscope_generate("a cat outside a box",
                                     num_frames=32)
    export_to_gif(test_output, "outputs/zeroscope.gif")

# prompt = "a cat outside a box"
# output = pipe(prompt, num_frames=24)
# frames = output.frames[0]
# video_frames_PIL_list = [Image.fromarray((frame*255).astype(np.uint8)) for frame in frames]
# export_to_gif(video_frames_PIL_list, "outputs/zeroscope.gif")