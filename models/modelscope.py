import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video, export_to_gif
import numpy as np
from PIL import Image

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b",
                                         torch_dtype=torch.float16,
                                         variant="fp16",
                                         cache_dir="/homes/55/runjia/scratch/diffusion_weights")
pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()


def modelscope_generate(prompt,
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
    test_output = modelscope_generate("a cat outside a box",
                                      num_frames=32)
    export_to_gif(test_output, "outputs/modelscope.gif")


