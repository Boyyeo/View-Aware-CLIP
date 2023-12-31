from pathlib import Path
from typing import Tuple, Union, Optional
from urllib.request import urlretrieve
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from tqdm import tqdm 
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

def get_random_crop_params(
    image_height: int, image_width: int, min_crop_size: int
) -> Tuple[int, int, int, int]:
    crop_size = np.random.randint(min_crop_size, min(image_height, image_width))
    x = np.random.randint(image_width - crop_size + 1)
    y = np.random.randint(image_height - crop_size + 1)
    return x, y, crop_size


def get_cropped_image(
    im_tensor: np.array, x: int, y: int, crop_size: int
) -> np.array:
    return im_tensor[
        y : y + crop_size,
        x : x + crop_size,
        ...
    ]


def update_saliency_map(
    saliency_map: np.array, similarity: float, x: int, y: int, crop_size: int
) -> None:
    saliency_map[
        y : y + crop_size,
        x : x + crop_size,
    ] += similarity


def cosine_similarity(
    one: Union[np.ndarray, torch.Tensor], other: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    return one @ other.T / (np.linalg.norm(one) * np.linalg.norm(other))



model_checkpoint =  "openai/clip-vit-large-patch14"

model = CLIPModel.from_pretrained(model_checkpoint).eval()
processor = CLIPProcessor.from_pretrained(model_checkpoint)

checkpoint_path = None
if checkpoint_path:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print("loaded succesfully")

n_iters = 300
min_crop_size = 50

query = "dog"
image_path = 'datasets/control_view_data_filtered/Dog/a photo of a Dog_sample_0_view_left_seed_0.png'
image = Image.open(image_path)
im_tensor = np.array(image)

x_dim, y_dim = image.size
inputs = processor(text=[query], images=[im_tensor], return_tensors="pt")
with torch.no_grad():
    results = model(**inputs)
results.keys()
initial_similarity = cosine_similarity(results.text_embeds, results.image_embeds).item()  # 1. Computing query and image similarity
saliency_map = np.zeros((y_dim, x_dim))

for _ in tqdm(range(n_iters)):  # 6. Setting number of the procedure iterations
    x, y, crop_size = get_random_crop_params(y_dim, x_dim, min_crop_size)
    im_crop = get_cropped_image(im_tensor, x, y, crop_size)  # 2. Getting a random crop of the image

    inputs = processor(text=[query], images=[im_crop], return_tensors="pt")
    with torch.no_grad():
        results = model(**inputs)  # 3. Computing crop and query similarity

    similarity = cosine_similarity(results.text_embeds, results.image_embeds).item() - initial_similarity  # 4. Subtracting query and image similarity from crop and query similarity
    update_saliency_map(saliency_map, similarity, x, y, crop_size)  # 5. Updating the region on the saliency map

plt.figure(dpi=150)
plt.imshow(saliency_map, norm=colors.TwoSlopeNorm(vcenter=0), cmap='jet')
plt.colorbar(location="bottom")
plt.title(f'Query: \"{query}\"')
plt.axis("off")
plt.show()
plt.savefig('ss.jpg')

def plot_saliency_map(image_tensor: np.ndarray, saliency_map: np.ndarray, query: Optional[str]) -> None:
    fig = plt.figure(dpi=150)
    plt.imshow(image_tensor)
    plt.imshow(
        saliency_map,
        norm=colors.TwoSlopeNorm(vcenter=0),
        cmap="jet",
        alpha=0.5,  # make saliency map trasparent to see original picture
    )
    if query:
        plt.title(f'Query: "{query}"')
    plt.axis("off")
    plt.savefig('saliency.jpg')
    return fig



plot_saliency_map(im_tensor, saliency_map, query)
