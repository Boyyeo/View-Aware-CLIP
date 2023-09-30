from modeling_clip import MyCLIPModel
from PIL import Image
from transformers import AutoTokenizer, CLIPTextModel
from stable_diffusion_pipeline import MyStableDiffusionPipeline
import torch
pretrained_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
outputs = pretrained_model(**inputs)
last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooler_output



checkpoint_path=None
model = MyCLIPModel.from_pretrained("openai/clip-vit-large-patch14")
if checkpoint_path:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    print("loaded succesfully")

pretrained_model.text_model = model.text_model 
print("successfully~")
outputs = pretrained_model(**inputs)
last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooler_output

model_id = "runwayml/stable-diffusion-v1-5"
pipe = MyStableDiffusionPipeline.from_pretrained(model_id)
pipe.text_encoder = pretrained_model
pipe = pipe.to("cuda")

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

sample = 3

right_save_image = None
prompt = "a photo of 3d car model right view"
for _ in range(sample):
    image = pipe(prompt).images[0]  
    if right_save_image:
        right_save_image = get_concat_h(right_save_image,image)
    else:
        right_save_image = image

left_save_image = None
prompt = "a photo of 3d car model left view"
for _ in range(sample):
    image = pipe(prompt).images[0]  
    if left_save_image:
        left_save_image = get_concat_h(left_save_image,image)
    else:
        left_save_image = image

front_save_image = None
prompt = "a photo of 3d car model front view"
for _ in range(sample):
    image = pipe(prompt).images[0]  
    if front_save_image:
        front_save_image = get_concat_h(front_save_image,image)
    else:
        front_save_image = image

back_save_image = None
prompt = "a photo of 3d car model back view"

for _ in range(sample):
    image = pipe(prompt).images[0]  
    if back_save_image:
        back_save_image = get_concat_h(back_save_image,image)
    else:
        back_save_image = image

save_image = get_concat_v(get_concat_v(get_concat_v(front_save_image,back_save_image),left_save_image),right_save_image)
save_image.save("car.png")
