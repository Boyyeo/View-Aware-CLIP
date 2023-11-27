from torch import nn
from transformers import Trainer
from transformers.trainer import *
import torch
from transformers import CLIPProcessor, CLIPTokenizer
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from finetune import image_title_dataset, infiniteloop
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

'''
# cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

'''

def infiniteloop(dataloader):
    while True:
        for  input_dict in iter(dataloader):
            yield  input_dict


class view_dataset(Dataset):
    def __init__(self, csv_path):
        self.prepare_view_data(csv_path)
        self.preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


    def prepare_view_data(self,csv_path):
        df = pd.read_csv(csv_path)
        self.image_path_list = df["image_path"].values.tolist()
        self.caption_list = df["caption"].values.tolist()


    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_path_list[idx])
        input_dict = self.preprocess(text=[self.caption_list[idx]], images=image, return_tensors="pt", padding='max_length', max_length=30)
        #print("input_dict:",input_dict['input_ids'].shape,input_dict['pixel_values'].shape)
        #print("image_path",self.image_path_list[idx])

        return input_dict



class CLIPTrainer(Trainer):
    def setup_view_loader(self):
        #list_image_path ='car_orthogonal/train'
        #train_view_dataset = image_title_dataset(list_image_path=list_image_path,view_data=True)
        train_view_dataset = view_dataset(csv_path='3dbicar_train.csv')
        train_view_dataloader = DataLoader(train_view_dataset,batch_size = 4) 
        self.view_looper = infiniteloop(train_view_dataloader)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.best_acc = 0.0
        self.global_steps = 0
        #from run_clip import Transform
        #image_size=224 
        #image_processor_mean = [0.48145466, 0.4578275, 0.40821073] 
        #image_processor_std=[0.26862954, 0.26130258, 0.27577711]
        #self.image_transformations = Transform(image_size, image_processor_mean, image_processor_std)
    
        #view_image, view_caption = next(self.view_looper)
        #print("view_image:{} view_caption:{}".format(view_image.shape,view_caption))

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs.pop('caption')
        #print("captions:",captions)
        device = inputs['input_ids'].device

        view_inputs = next(self.view_looper).to(device)
        #print('---------------------------------------\n')
        #view_inputs.pixel_values = view_inputs.pixel_values.squeeze()
        #view_inputs.input_ids = view_inputs.input_ids.squeeze()
        #view_inputs.attention_mask = view_inputs.attention_mask.squeeze()
        #print("images:{} text_idx:{} attention_mask:{}".format(view_inputs.pixel_values.shape,view_inputs.input_ids.shape,view_inputs.attention_mask.shape))

        #pixel_values = self.image_transformations(view_image)
        #print("view_image:{} view_caption:{}".format(view_image.shape,view_caption))
        #print("view_image:",view_image.type(),inputs['input_ids'].type())
        #view_image = view_image.to(device)
        #view_inputs = self.processor(text=list(view_caption), images=view_image, return_tensors="pt", padding="longest").to(device) 
        #view_inputs.pixel_values = pixel_values
        #print("view_image:",view_image.type(),inputs['input_ids'].type())
        #print("input:{} view_image:{}".format(inputs['input_ids'].device,view_image.device))
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        #outputs_view = model(**view_inputs,return_loss=True)
        outputs_view = model(input_ids=view_inputs.input_ids.squeeze(),pixel_values=view_inputs.pixel_values.squeeze(),attention_mask=view_inputs.attention_mask.squeeze(), return_loss=True, return_dict=True)

        #print("view_inputs.input_ids:",view_inputs.input_ids.shape)
        #print("output:{} outputs_view:{}".format(outputs.keys(),outputs_view.keys()))
        
        #print("output_view")
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if is_peft_available() and isinstance(model, PeftModel):
                model_name = unwrap_model(model.base_model)._get_name()
            else:
                model_name = unwrap_model(model)._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            #print("************************** outputs:{} ***************************************".format(outputs.keys()))
            loss_ori = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            loss_view = outputs_view["loss"] if isinstance(outputs_view, dict) else outputs_view[0]
            loss = loss_ori + 0.35 * loss_view
            #loss =  loss_view 
            #print("Loss:{} loss_ori:{} loss_view:{}".format(loss.item(),loss_ori.item(),loss_view.item()))

        self.global_steps += 1
        if self.global_steps % 100 == 0:
            self.evaluate()

        return (loss, outputs) if return_outputs else loss

    
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        args = self.args
        #prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        #batch_size = self.args.eval_batch_size

        model.eval()
       
        cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
        test_samples = 1000
        num_correct = 0
        for i in tqdm(range(test_samples)):
            image, class_id = cifar100[i]
            text_inputs = [(f"a photo of a {c}") for c in cifar100.classes]
            with torch.no_grad():
                inputs = self.processor(text=text_inputs, images=image, return_tensors="pt", padding=True).to(model.device)
                outputs = model(**inputs)
                text_embeds, image_embeds = outputs['text_embeds'], outputs['image_embeds']

            similarity = (100.0 * image_embeds @ text_embeds.T).softmax(dim=-1)
            values, indices = similarity[0].topk(5)
            num_correct = num_correct + 1 if class_id in indices else num_correct
            
        print("CIFAR100 Accuracy with {} samples is : {}".format(test_samples,num_correct/test_samples))
   
        df = pd.read_csv('3dbicar_test.csv')
        image_path_list = df['image_path'].to_list()
        caption_list = df['caption'].to_list()
        total_num = 0
        total_correct = 0

        template_list = ['a photo of a {} 3D model front view','a photo of a {} 3D model back view','a photo of a {} 3D model left side view','a photo of a {} 3D model right side view']



        for image_path, caption in zip(image_path_list, caption_list):
            subject_name = caption.split(' ')[4]
            image = Image.open(image_path)
            ground_truth = -1
            query_list = template_list.copy()
            
            for i in range(len(query_list)):
                query_list[i] = query_list[i].format(subject_name)
                if caption == query_list[i]:
                    ground_truth = i
                    #print("ground truth:{} caption:{}".format(ground_truth,caption))

            input_dict = self.processor(text=query_list, images=image, return_tensors="pt", padding=True).to(model.device)
            outputs_dict = model(**input_dict)
            logits_per_image, logits_per_text = outputs_dict['logits_per_image'], outputs_dict['logits_per_text']
            probs = logits_per_image.softmax(dim=-1).cpu().detach()
            max_idx = torch.argmax(probs.squeeze())
            total_correct += int(max_idx==ground_truth)
            total_num += 1

        accuracy = total_correct / total_num
        print("Orthogonal View Accuracy:{}".format(accuracy))

        if accuracy > self.best_acc:
            self.best_acc = accuracy
            checkpoint = {
                    'model': model.state_dict(),
                    #'optim': optimizer.state_dict(),
                    #'epoch': epoch,
                    'best_acc': self.best_acc,
                }
            torch.save(checkpoint,'checkpoint/3dbicar_view.pyt')

        

        total_num = 0
        total_correct = 0
        dir_list = ['front.png','back.png','left.png','right.png']
        query_list = ['a photo of car 3D model front view','a photo of car 3D model back view','a photo of car 3D model left side view','a photo of car 3D model right side view']
        for folder in tqdm(os.listdir("car_orthogonal/test")):
                for i in range(len(dir_list)):
                    total_num += 1
                    file_path = 'car_orthogonal/test/{}/{}'.format(folder,dir_list[i])
                    image = Image.open(file_path)
                    input_dict = self.processor(text=query_list, images=image, return_tensors="pt", padding=True).to(model.device)
                    outputs_dict = model(**input_dict)
                    logits_per_image, logits_per_text = outputs_dict['logits_per_image'], outputs_dict['logits_per_text']
                    probs = logits_per_image.softmax(dim=-1).cpu().detach()
                    max_idx = torch.argmax(probs.squeeze())
                    #image.save('validate/{}/{}'.format(dir_list[max_idx].split('.png')[0],'{}_'.format(folder)+dir_list[i]))
                    total_correct += int(max_idx==i)
            
            
        accuracy = total_correct / total_num
        print("Car View Accuracy:{}".format(accuracy))
        
        return {'eval_loss':1.000}