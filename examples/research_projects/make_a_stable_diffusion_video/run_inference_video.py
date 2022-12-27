#!/usr/bin/env python3

from diffusers import StableDiffusionVideoInpaintPipeline
import torch
import imageio

from PIL import Image, ImageDraw

model_id = "lxj616/make-a-stable-diffusion-video-timelapse"
pipe = StableDiffusionVideoInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_sequential_cpu_offload()

# notice this is only the video prompt, must be cloudscape, because I only trained on that, do not mention the cat
prompts = ["a fantasy sureal painting of cityscape and cloudscape, trending on artstation, colorful vibrant"]

# provide first frame, generated from elsewhere
# prompt = "a portrait of a cat, sitting on top of a tall building under sunset clouds"
init_image = Image.open("assets/cat.png").convert("RGB").resize((512, 512))
# provide first frame as a whole, or you could use custom mask, it is also supported to do inpaint while making video
mask_image = Image.new("L", (512,512), 0).convert("RGB")

counter_i = 0
for p in prompts:
    for i in range(100):
        images = pipe(p, image=init_image, mask_image=mask_image, num_inference_steps=100, guidance_scale=12.0, frames_length=120).images
        counter_j = 0
        #for i in images:
        #    counter_j += 1
        #    i.save("/tmp/test_timelapse/image_" + str(counter_i) + "_" + str(counter_j) + ".png")
        imageio.mimsave('/tmp/test_timelapse_final/gif_' + str(counter_i) + ".gif", images, fps = 12)
        counter_i += 1
