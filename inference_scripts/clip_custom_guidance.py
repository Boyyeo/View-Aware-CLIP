#https://github.com/huggingface/diffusers/tree/9be94d9c6659f7a0a804874f445291e3a84d61d4/examples/community#clip-guided-stable-diffusion

from diffusers import DiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPModel
import torch
from clip_guided_stable_diffusion import CLIPGuidedStableDiffusion

GPU_ID = 2
feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-large-patch14")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
checkpoint_path = 'checkpoint/control_view_best_train_acc_0.8944.pyt'
if checkpoint_path:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    clip_model.load_state_dict(checkpoint['model'])
    print("loaded succesfully")


guided_pipeline = CLIPGuidedStableDiffusion.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    #custom_pipeline="clip_guided_stable_diffusion",
    clip_model=clip_model,
    feature_extractor=feature_extractor,
    torch_dtype=torch.float16,
)
print("guided CLIP")
guided_pipeline.enable_attention_slicing()
guided_pipeline = guided_pipeline.to("cuda:{}".format(GPU_ID))

prompt = "a photo of a bear left side view"

generator = torch.Generator(device="cuda:{}".format(GPU_ID)).manual_seed(0)
images = []
for i in range(20):
    image = guided_pipeline(
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        clip_guidance_scale=100,
        num_cutouts=4,
        use_cutouts=False,
        generator=generator,
        clip_prompt='a photo of a bear left side view',
    ).images[0]
    images.append(image)

    if not checkpoint_path:
        image.save("clip_custom_guided_sd_image_{}.png".format(i))
    else:
        image.save("clip_custom_guided_sd_image_finetune_{}.png".format(i))
