# !pip install opencv-python transformers accelerate
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import torch
import cv2
from PIL import Image
from transformers import pipeline
import torchvision.transforms as T

def transform_control_image(image:Image):
    original_shape = 256
    resize_shape = np.random.randint(low=original_shape,high=512) 
    image = image.resize(size=(resize_shape,resize_shape))
    image = T.ToTensor()(image)
    white_image = torch.zeros(size=(3,512,512))

    half_shape = (512 - resize_shape)//2
    shift_x = np.random.randint(low=-half_shape,high=half_shape) 
    shift_y = np.random.randint(low=-half_shape,high=half_shape)  
    
    start_x = 512//2 - resize_shape//2 + shift_x
    start_y = 512//2 - resize_shape//2 + shift_y
    white_image[:,start_x:start_x+resize_shape,start_y:start_y+resize_shape] = image
    transform_image = T.ToPILImage()(white_image)
    return transform_image




def depth_estimate(image_path):
    image = Image.open(image_path).convert('RGB')
    depth_estimator = pipeline("depth-estimation", model ="Intel/dpt-hybrid-midas" )
    image = depth_estimator(image)['predicted_depth'][0]
    image = image.numpy()
    image_depth = image.copy()
    image_depth -= np.min(image_depth)
    image_depth /= np.max(image_depth)
    bg_threhold = 0.4
    x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    x[image_depth < bg_threhold] = 0
    y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    y[image_depth < bg_threhold] = 0
    z = np.ones_like(x) * np.pi * 2.0
    image = np.stack([x, y, z], axis=2)
    image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
    image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
    depth_image = Image.fromarray(image).resize((512,512))
    depth_image.save('depth.png')
    return depth_image

def canny_detect(image_path):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    # get canny image
    image = cv2.Canny(image, 250, 250)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image).resize((512,512))
    canny_image.save('canny.png')
    return canny_image

device = torch.device('cuda:2')
control_image_path = 'mvdream_result/cat/a photo of a cat_sample_0_view_front.png'
control_image = canny_detect(control_image_path)

# load control net and stable diffusion v1-5
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
).to(device)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed
pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

# generate image
seed_list = [0,1,2,3,4,5,6,7,8,9]
for seed in seed_list:
    generator = torch.manual_seed(seed)

    transformed_control_image = transform_control_image(control_image)


    image = pipe(
        "a photo of a cat, front view in forest background", num_inference_steps=20, generator=generator, image=transformed_control_image
    ).images[0]
    image.save('control_result/control_seed{}.jpg'.format(seed))