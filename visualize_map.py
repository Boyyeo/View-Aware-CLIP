#https://github.com/huggingface/diffusers/tree/9be94d9c6659f7a0a804874f445291e3a84d61d4/examples/community#clip-guided-stable-diffusion
#https://github.com/openai/CLIP/issues/82
#https://docs.openvino.ai/2023.2/notebooks/232-clip-language-saliency-map-with-output.html
from transformers import CLIPProcessor, CLIPModel
import torch
import torchvision.transforms as T
from PIL import Image 

GPU_ID = 2
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
checkpoint_path = 'checkpoint/control_view_best_train_acc_0.8944.pyt'
if checkpoint_path:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    clip_model.load_state_dict(checkpoint['model'])
    print("loaded succesfully")

text_inputs = ['a photo of a bear left view']
image_path = 'datasets/VIEW_DATA/Bear,  facing left side/left_0000.jpg'
image = Image.open(image_path).convert('RGB')

with torch.no_grad():
    inputs = processor(text=text_inputs, images=image, return_tensors="pt", padding=True).to(clip_model.device)
    outputs = clip_model(**inputs)
    print("outputs keys:",outputs.keys())
    text_embeds, image_embeds = outputs['text_embeds'], outputs['image_embeds']
    print("text_embeds:{} image_embeds:{}".format(text_embeds.shape,image_embeds.shape))
    print("logits_per_image:{} logits_per_text:{} text_model_output:{} vision_model_output:{}".format(outputs['logits_per_image'].shape,outputs['logits_per_text'].shape,outputs['text_model_output'][0].shape,outputs['vision_model_output'][0].shape))
#similarity = (100.0 * image_embeds @ text_embeds.T).softmax(dim=-1)
#values, indices = similarity[0].topk(5)
#num_correct = num_correct + 1 if class_id in indices else num_correct

