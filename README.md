## 112 Semester IMVFX Final Project : View-Aware CLIP

#### Abstract: 
CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a
variety of (image, text) pairs, having a strong capability in representing text and images
after extensive learning. However, it still faces challenges with more complex concepts.
CLIP finds it difficult to recognize photos of object taken from different perspectives. Our
project aims to enhance its understanding of object from different angles.

#### Data Collection (Please follow the instructions in folder ```dataset_scripts``` to generate the dataset)
We have to collect a view-aligned image-text pair dataset by ourselves because
this is a new task.

#### To train our View-Aware CLIP 
```
bash scripts/run.sh
```

#### Image generation with View-Aware CLIP-guidance Stable Diffusion
```
bash scripts/demo.sh
```

#### To visualize the heatmap 
```
python visualize.py
```


#### View-Aware CLIP checkpoint can be downloaded from [here](https://drive.google.com/file/d/10tIFHh7Rcmnv4m8osOHdWY6j8EzMJ3Wf/view?usp=sharing)
