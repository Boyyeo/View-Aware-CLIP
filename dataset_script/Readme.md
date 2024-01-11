## Collecting View-Alinged Dataset

Our method is to first render 3D model into 2D images. However, datasets collected in this manner will cause a
domain gap between domain of 3D models rendered images and domain in real photos.
Our solution is to use the images rendered from the 3D models as a conditions for
the controlNet in stable diffusion, and finally generating a more realistic image to alleviate
the domain gap problems

## To generate the a view-aligned dataset with stable diffusion(ControlNET)
```
python execute_control_gpu.py (please change the gpu id you want to use in the script)
```

## Generating the a single image with stable diffusion 
```
bash generate.sh
```

Notice: All the categories we had collected and the scene prompt that were used to generate realistic image listed in folder ```text_file```
