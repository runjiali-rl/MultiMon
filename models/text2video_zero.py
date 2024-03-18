import torch
from diffusers.utils import load_image, export_to_gif
from diffusers import TextToVideoZeroPipeline
from PIL import Image


class Text2VideoZeroPipeline():
    def __init__(self, cache_dir: str = "/homes/55/runjia/scratch/diffusion_weights"):
        self.pipe = TextToVideoZeroPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                            torch_dtype=torch.float16,
                                                            cache_dir=cache_dir)
        self.pipe = self.pipe.to("cuda")
        # self.pipe.enable_model_cpu_offload()
        # self.pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
        # self.pipe.enable_vae_slicing()

    def generate(self, prompt):
        output = self.pipe(
            prompt=prompt
        )
        frames = output.images
        frames = [Image.fromarray((frame*255).astype("uint8")) for frame in frames]
        return frames


# model_id = "runwayml/stable-diffusion-v1-5"
# pipe = TextToVideoZeroPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# prompt = "A panda is playing guitar on times square"
# result = pipe(prompt=prompt).images
# result = [(r * 255).astype("uint8") for r in result]
# imageio.mimsave("video.mp4", result, fps=4)

if __name__ == "__main__":
    model = Text2VideoZeroPipeline()
    test_output = model.generate("a cat outside a box")
    export_to_gif(test_output, "outputs/text2video_zero.gif")
    