import os
import pandas as pd
rootdir = 'mvdream_result'
subdirs_list = []
for rootdir, dirs, files in os.walk(rootdir):
    for subdir in dirs:
        subdirs_list.append(os.path.join(rootdir, subdir))

print(subdirs_list)

image_path_list = []
caption_list = []
for subdir in subdirs_list:
    for file in os.listdir(subdir):
        if file.endswith(".png"):
            #print(os.path.join(subdir, file))
            image_path = os.path.join(subdir, file)
            split_list = file.split('_')
            caption = split_list[0] + ' ' + split_list[-1]
            if 'left' in caption or 'right' in caption:
                caption = caption.replace('.png',' side view')
            else:
                caption = caption.replace('.png',' view')

            print("caption:",caption)
            image_path_list.append(image_path)
            caption_list.append(caption)

# dictionary of lists 
df_dict = {'image_path': image_path_list, 'caption': caption_list} 
   
df = pd.DataFrame(df_dict)
df.to_csv('mvdream_train.csv',index=False)