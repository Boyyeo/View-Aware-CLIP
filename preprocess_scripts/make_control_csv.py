import os
import pandas as pd
rootdir = 'datasets/control_view_data_filtered_eval'
subdirs_list = []
for rootdir, dirs, files in os.walk(rootdir):
    for subdir in dirs:
        subdirs_list.append(os.path.join(rootdir, subdir))

#print(subdirs_list)

image_path_list = []
caption_list = []

for subdir in subdirs_list:
    for file in os.listdir(subdir):
        if file.endswith(".png"):
            #print(os.path.join(subdir, file))
            image_path = os.path.join(subdir, file)
            print('image_path:',image_path)
            
            caption = image_path.split('/')[-1].split('_')[0].lower()

            view_prompt = ''
            if 'left' in image_path:
                caption = caption + ' left side view'
            
            elif 'right' in image_path:
                caption = caption + ' right side view'

            elif 'front' in image_path:
                caption = caption + ' front view'
            
            elif 'back' in image_path:
                continue
                caption = caption + ' back view'
            
            print("caption:",caption)

            #caption = image_path.split('/')[1]
            #caption = 'a photo of a ' + caption.lower().replace(',','').replace('facing right side','right side view').replace('facing left side','left side view')
            image_path_list.append(image_path)
            caption_list.append(caption)

#dictionary of lists 
df_dict = {'image_path': image_path_list, 'caption': caption_list}   
df = pd.DataFrame(df_dict)
df.to_csv('control_view_test_edit.csv',index=False)