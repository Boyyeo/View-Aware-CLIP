import os
from tqdm import tqdm 
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPTokenizer
from modeling_clip import MyCLIPModel
import shutil 
EPOCH = 25
BATCH_SIZE = 4
checkpoint_path='checkpoint/last.pyt'
device = "cuda:1" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model = MyCLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
#text_features = model.get_text_features(**inputs)
#print("text_features:",text_features.shape)






class image_title_dataset(Dataset):
    def __init__(self, list_image_path):
        self.prepare_data(list_image_path)

    def prepare_data(self,list_image_path):
        self.image_path = []
        self.list_txt = []
        for folder in os.listdir(list_image_path):
            for file in os.listdir(list_image_path+'/'+folder):
                self.image_path.append('{}/{}/{}'.format(list_image_path,folder,file))
                image_direction = file.split('.png')[0]
                caption = 'a photo of 3d car model {} view'.format(image_direction)
                self.list_txt.append(caption)

        #self.title = tokenizer(self.list_txt)
        #print("title:",self.title.keys())


    def __len__(self):
        return len(self.list_txt)

    def __getitem__(self, idx):
        #print("path:",self.image_path[idx])
        image = Image.open(self.image_path[idx])
        input_dict = preprocess(text=[self.list_txt[idx]], images=image, return_tensors="pt", padding=True)
        return input_dict


list_image_path ='car_orthogonal/train'
train_dataset = image_title_dataset(list_image_path)
train_dataloader = DataLoader(train_dataset,batch_size = BATCH_SIZE) 



def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 



loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
model.setup_finetune_text_encoder()
params = [params for params in model.parameters() if params.requires_grad]

optimizer = optim.Adam(params, lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset




if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        print("loaded checkpoint successfully")
else:
    start_epoch = 0
    best_acc = 0

#fixed_image = preprocess(Image.open('car_orthogonal/test/003000/front.png')).to(device) # Image from PIL module
# add your own code to track the training progress.
for epoch in range(start_epoch,EPOCH):
  model.train()
  for batch in tqdm(train_dataloader) :
      optimizer.zero_grad()

      input_dict = batch.to(device) 
      #print("input_dict:",input_dict.keys())
      #print("images:{} text_idx:{} attention_mask:{}".format(input_dict.pixel_values.shape,input_dict.input_ids.shape,input_dict.attention_mask.shape))
      outputs_dict = model(input_ids=input_dict.input_ids.squeeze(),pixel_values=input_dict.pixel_values.squeeze(),attention_mask=input_dict.attention_mask.squeeze(), return_loss=True, return_dict=True)
      loss = outputs_dict['loss']
      loss.backward()
      optimizer.step()
      break
 
  print("Epoch {}/{} Training Loss:{}".format(epoch,EPOCH,loss.item()))
  model.eval()
  total_num = 0
  total_correct = 0
  dir_list = ['front.png','back.png','left.png','right.png']
  query_list = ['a photo of 3d car model front view','a photo of 3d car model back view','a photo of 3d car model left view','a photo of 3d car model right view']
  #query_token_list = tokenizer(query_list).to(device)
  for folder in tqdm(os.listdir("car_orthogonal/test")):
        for i in range(len(dir_list)):
            total_num += 1
            file_path = 'car_orthogonal/test/{}/{}'.format(folder,dir_list[i])
            image = Image.open(file_path)
            #image = preprocess(image).unsqueeze(0).to(device)
            input_dict = preprocess(text=query_list, images=image, return_tensors="pt", padding=True).to(device)
            #print("images:{} text_idx:{} attention_mask:{}".format(input_dict.pixel_values.shape,input_dict.input_ids.shape,input_dict.attention_mask.shape))
            outputs_dict = model(**input_dict)
            #outputs_dict = model(input_ids=input_dict.input_ids,pixel_values=input_dict.pixel_values,attention_mask=input_dict.attention_mask, return_loss=True, return_dict=True)
            logits_per_image, logits_per_text = outputs_dict['logits_per_image'], outputs_dict['logits_per_text']
            #print("logits_per_image:",logits_per_image.shape,logits_per_image)
            probs = logits_per_image.softmax(dim=-1).cpu().detach()
            #print("probs:",probs)

            max_idx = torch.argmax(probs.squeeze())
            image.save('validate/{}/{}'.format(dir_list[max_idx].split('.png')[0],'{}_'.format(folder)+dir_list[i]))
            
            #print("max_idx:",max_idx)
            total_correct += int(max_idx==i)
    
  accuracy = total_correct / total_num
  print("Epoch {}/{} Accuracy:{}".format(epoch,EPOCH,total_correct/total_num))
  if accuracy > best_acc:
      best_acc = accuracy
      checkpoint = {
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc,
        }
      torch.save(checkpoint,'checkpoint/last.pyt')

  

    