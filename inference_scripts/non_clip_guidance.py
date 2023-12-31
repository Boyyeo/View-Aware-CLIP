from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
GPU_ID = 2
pipe = pipe.to("cuda:{}".format(GPU_ID))
generator = torch.Generator(device="cuda:{}".format(GPU_ID)).manual_seed(0)

for i in range(20):
    prompt = "a photo of a bear right side view"
    image = pipe(prompt,generator=generator).images[0]  
        
    image.save("clip_non_guided_sd_image_{}.png".format(i))
