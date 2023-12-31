from torch import nn
from transformers import Trainer
from transformers.trainer import *
import torch
from transformers import CLIPProcessor, CLIPTokenizer
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize,ToTensor
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T

class Transform(torch.nn.Module):
    def __init__(self, image_size):
        super().__init__()
       
        self.transforms = T.Compose([T.Resize(size=(image_size,image_size),interpolation=InterpolationMode.BICUBIC),
                                     T.CenterCrop(size=(image_size,image_size)),
                                     T.ToTensor(),
                                     T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                                     ])

    def forward(self, x) -> torch.Tensor:
        """`x` should be an instance of `PIL.Image.Image`"""
        #print("x:",x)
        with torch.no_grad():
            x = self.transforms(x)
        return x

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
        #self.image_transform = Transform(image_size=224)


    def prepare_view_data(self,csv_path):
        df = pd.read_csv(csv_path)
        self.image_path_list = df["image_path"].values.tolist()
        self.caption_list = df["caption"].values.tolist()


    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_path_list[idx]).convert('RGB')
        #image = self.image_transform(image)
        input_dict = self.preprocess(text=[self.caption_list[idx]], images=image, return_tensors="pt", padding='max_length', max_length=30)
        #print("input_dict:",input_dict['input_ids'].shape,input_dict['pixel_values'].shape)
        #print("image_path",self.image_path_list[idx])

        return input_dict



class CLIPTrainer(Trainer):
    def setup_view_loader(self):
       
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.best_acc = 0.0
        self.global_steps = 0
        self.image_transform = Transform(image_size=224)


    def compute_loss(self, model, inputs, return_outputs=False):
        inputs.pop('caption')
        device = inputs['input_ids'].device
        view_inputs = inputs.pop('view_inputs') # next(self.view_looper).to(device)
       
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        outputs_view = model(input_ids=view_inputs.input_ids.squeeze(),pixel_values=view_inputs.pixel_values.squeeze(),attention_mask=view_inputs.attention_mask.squeeze(), return_loss=True, return_dict=True)
       
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
            loss_ori = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            loss_view = outputs_view["loss"] if isinstance(outputs_view, dict) else outputs_view[0]
            loss = loss_ori + 0.2 * loss_view
            #loss= loss_view

        self.global_steps += 1
        if self.global_steps % 10 == 0 or self.global_steps == 1:
            self.evaluate()

        return (loss, outputs) if return_outputs else loss

    
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        test_back=True,
    ) -> Dict[str, float]:
        print("Direction Test:{}".format('Front, Left, Right, Back' if test_back else 'Front, Left, Right'))
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


        model.eval()
        cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
        test_samples = 1000
        num_correct = 0
        for i in tqdm(range(test_samples)):
            image, class_id = cifar100[i]
            text_inputs = [(f"a photo of a {c}") for c in cifar100.classes]
            with torch.no_grad():
                inputs = self.processor(text=text_inputs, images=image, return_tensors="pt", padding=True).to(model.device)
                #inputs['use_prompt_embedding'] = False
                outputs = model(**inputs)
                text_embeds, image_embeds = outputs['text_embeds'], outputs['image_embeds']

            similarity = (100.0 * image_embeds @ text_embeds.T).softmax(dim=-1)
            values, indices = similarity[0].topk(5)
            num_correct = num_correct + 1 if class_id in indices else num_correct
            
        print("CIFAR100 Accuracy with {} samples is : {}".format(test_samples,num_correct/test_samples))

        if test_back:
            df = pd.read_csv('datasets/crawl_eval_have_back.csv')
        else:
            df = pd.read_csv('datasets/crawl_eval.csv')

        image_path_list = df['image_path'].to_list()
        caption_list = df['caption'].to_list()
        total_num = 0
        total_correct = 0
        template_list = ['a photo of a {} front view','a photo of a {} left side view','a photo of a {} right side view']
        confusion_matrix = torch.zeros(size=(4,4))
        if test_back:
            template_list.append('a photo of a {} back view')

        for image_path, caption in zip(image_path_list, caption_list):
            subject_name = caption.split(' ')[4]
            #print("crawl subject_name:",subject_name)
            image = Image.open(image_path)
            #image = self.image_transform(image)
            ground_truth = -1
            query_list = template_list.copy()
            
            for i in range(len(query_list)):
                query_list[i] = query_list[i].format(subject_name)
                if caption == query_list[i]:
                    ground_truth = i
                    #print("crawl ground truth:{} caption:{}".format(ground_truth,caption))

            input_dict = self.processor(text=query_list, images=image, return_tensors="pt", padding=True).to(model.device)
            outputs_dict = model(**input_dict)
            logits_per_image, logits_per_text = outputs_dict['logits_per_image'], outputs_dict['logits_per_text']
            probs = logits_per_image.softmax(dim=-1).cpu().detach()
            max_idx = torch.argmax(probs.squeeze())
            total_correct += int(max_idx==ground_truth)
            confusion_matrix[ground_truth][max_idx] += 1
            total_num += 1

        accuracy = total_correct / total_num
        print("Crawling Data Accuracy:{}".format(accuracy))
        print("Confusion Matrix\n:{}".format(confusion_matrix))

        if test_back:
            df = pd.read_csv('datasets/control_view_train_have_back.csv')
        else:
            df = pd.read_csv('datasets/control_view_train_edit.csv')

        image_path_list = df['image_path'].to_list()
        caption_list = df['caption'].to_list()
        total_num = 0
        total_correct = 0
        template_list = ['a photo of a {} front view','a photo of a {} left side view','a photo of a {} right side view']
        confusion_matrix = torch.zeros(size=(4,4))

        if test_back:
            template_list.append('a photo of a {} back view')
        for image_path, caption in zip(image_path_list, caption_list):
            subject_name = caption.split(' ')[4]
            #print("control subject_name:",subject_name)

            image = Image.open(image_path)
            #image = self.image_transform(image)

            ground_truth = -1
            query_list = template_list.copy()
            
            for i in range(len(query_list)):
                query_list[i] = query_list[i].format(subject_name)
                if caption == query_list[i]:
                    ground_truth = i
                    #print("control ground truth:{} caption:{}".format(ground_truth,caption))

            input_dict = self.processor(text=query_list, images=image, return_tensors="pt", padding=True).to(model.device)
            outputs_dict = model(**input_dict)
            logits_per_image, logits_per_text = outputs_dict['logits_per_image'], outputs_dict['logits_per_text']
            probs = logits_per_image.softmax(dim=-1).cpu().detach()
            max_idx = torch.argmax(probs.squeeze())
            total_correct += int(max_idx==ground_truth)
            confusion_matrix[ground_truth][max_idx] += 1
            total_num += 1

        accuracy = total_correct / total_num
        print("Control Train View Data Accuracy:{}".format(accuracy))
        print("Confusion Matrix\n:{}".format(confusion_matrix))

        if accuracy > self.best_acc:
                self.best_acc = accuracy
                checkpoint = {
                        'model': model.state_dict(),
                        #'optim': optimizer.state_dict(),
                        #'epoch': epoch,
                        'best_acc': self.best_acc,
                    }
                torch.save(checkpoint,'checkpoint/control_view_best_train_acc_{}.pyt'.format(round(self.best_acc,4)))


        if test_back:
            df = pd.read_csv('datasets/control_view_test_have_back.csv')
        else:
            df = pd.read_csv('datasets/control_view_test_edit.csv')
        image_path_list = df['image_path'].to_list()
        caption_list = df['caption'].to_list()
        total_num = 0
        total_correct = 0
        template_list = ['a photo of a {} front view','a photo of a {} left side view','a photo of a {} right side view']
        confusion_matrix = torch.zeros(size=(4,4))

        if test_back:
            template_list.append('a photo of a {} back view')
        for image_path, caption in zip(image_path_list, caption_list):
            subject_name = caption.split(' ')[4]
            #print("control subject_name:",subject_name)

            image = Image.open(image_path)
            #image = self.image_transform(image)

            ground_truth = -1
            query_list = template_list.copy()
            
            for i in range(len(query_list)):
                query_list[i] = query_list[i].format(subject_name)
                if caption == query_list[i]:
                    ground_truth = i
                    #print("control ground truth:{} caption:{}".format(ground_truth,caption))

            input_dict = self.processor(text=query_list, images=image, return_tensors="pt", padding=True).to(model.device)
            outputs_dict = model(**input_dict)
            logits_per_image, logits_per_text = outputs_dict['logits_per_image'], outputs_dict['logits_per_text']
            probs = logits_per_image.softmax(dim=-1).cpu().detach()
            max_idx = torch.argmax(probs.squeeze())
            total_correct += int(max_idx==ground_truth)
            confusion_matrix[ground_truth][max_idx] += 1
            total_num += 1

        accuracy = total_correct / total_num
        print("Control Test View Data Accuracy:{}".format(accuracy))
        print("Confusion Matrix\n:{}".format(confusion_matrix))

        
        return {'eval_loss':1.000}