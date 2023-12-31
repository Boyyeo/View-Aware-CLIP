import os
import pandas as pd
rootdir = 'datasets/VIEW_DATA_EVAL'
subdirs_list = []
for rootdir, dirs, files in os.walk(rootdir):
    for subdir in dirs:
        subdirs_list.append(os.path.join(rootdir, subdir))

print(subdirs_list)


image_path_list = []
caption_list = []
for subdir in subdirs_list:
    for file in os.listdir(subdir):
        if file.endswith(".jpg"):
            #print(os.path.join(subdir, file))
            image_path = os.path.join(subdir, file)
            #if 'back' in image_path:
            #    continue
            print('image_path:',image_path)
            caption = image_path.split('/')[2]
            caption = 'a photo of a ' + caption.lower().replace(',','').replace('facing right side','right side view').replace('facing left side','left side view')
            image_path_list.append(image_path)
            caption_list.append(caption)

# dictionary of lists 
df_dict = {'image_path': image_path_list, 'caption': caption_list} 
   
df = pd.DataFrame(df_dict)
df.to_csv('crawl_eval_have_back.csv',index=False)