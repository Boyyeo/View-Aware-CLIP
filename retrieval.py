import clip_openai
import torch
import os
from PIL import Image
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip_openai.load("ViT-B/32", device=device)


def calculate_similarity(query_embeddings, input_embeddings):
  similariries = query_embeddings @ input_embeddings.T
  return similariries

query_list = ['a photo of car front view','a photo of car back view','a photo of car left side view','a photo of car right side view']
query_token_list = clip_openai.tokenize(query_list).to(device)
query_embeddings_list = []

for query in query_list:
    query_tokens = clip_openai.tokenize([query]).to(device)  #Tokenize Before Embeddings
    with torch.no_grad():
        query_embeddings = model.encode_text(query_tokens)
        query_embeddings_list.append(query_embeddings)

total_num = 0
total_correct = 0
dir_list = ['front.png','back.png','left.png','right.png']
for folder in tqdm(os.listdir("car_orthogonal/test")):
    for i in range(len(dir_list)):
        total_num += 1
        file_path = 'car_orthogonal/test/{}/{}'.format(folder,dir_list[i])
        image = Image.open(file_path)
        image = preprocess(image).unsqueeze(0).to(device)

        logits_per_image, logits_per_text = model(image, query_token_list)
        probs = logits_per_image.softmax(dim=-1).cpu().detach()
        max_idx = torch.argmax(probs)
        total_correct += int(max_idx==i)
     

print("Accuracy: ",total_correct/total_num)

