
import os 
import json
from tqdm import tqdm
DATASET_PATH = '3dbicar'
SUBFOLDER_NAME = '16'

subfolder_list = [name for name in os.listdir(DATASET_PATH)]
subfolder_list.sort()

for SUBFOLDER_NAME in tqdm(subfolder_list):
    #print(SUBFOLDER_NAME)

    eobj_filename = '/{}/pose/e.obj'.format(SUBFOLDER_NAME)
    mobj_filename = '/{}/pose/m.obj'.format(SUBFOLDER_NAME)

    EOBJ_PATH = DATASET_PATH + eobj_filename
    MOBJ_PATH = DATASET_PATH + mobj_filename

    ############ TODO: Material m.BMP to m.bmp and e.BMP to e.bmp #############
    row_content_saved = []
    normal_flag = False
    for line in reversed(list(open(EOBJ_PATH))):
        if (line[0] == 'f' and len(line.split(' ')) >= 5) or normal_flag:
            normal_flag = True 
            row_content_saved.append(line)


    os.remove(EOBJ_PATH)
    f = open(EOBJ_PATH, "a")
    for line in reversed(row_content_saved):
        f.write(line)
    f.close()

        
    ###########  TODO: delete the last few rows in e.obj #############
    bmp_m_filename = DATASET_PATH + '/{}/pose/e.BMP'.format(SUBFOLDER_NAME)
    bmp_e_filename = DATASET_PATH + '/{}/pose/m.BMP'.format(SUBFOLDER_NAME)

    os.rename(bmp_m_filename,bmp_m_filename.replace('BMP','bmp'))
    os.rename(bmp_e_filename,bmp_e_filename.replace('BMP','bmp'))

    ###########  TODO: write the information to json file #############
    render_dict = {
                    "image_size": 512,
                    "camera_dist": 1.0,   
                    "elevation": [10],
                    "azim_angle": [0, 90, 180, 270],
                    "obj_filename": MOBJ_PATH,
                    "z_coord": 0     
                    }

    with open("render_params_list.json", "a") as outfile:
        json.dump(render_dict, outfile)
        outfile.write('\n')
