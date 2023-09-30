import os
from shutil import copy


dir_list = ['right.png','back.png','left.png','front.png']
dir_file_list = ['08.png','17.png','26.png','35.png']

folder_list = [] 
for file in os.listdir("car"):
    if 'DS_Store' in file:
        print(file)
    else:
        folder_list.append(file)


for count, folder in enumerate(folder_list):
    os.mkdir('car_orthogonal/{}'.format(str(count).zfill(6)))

    for i in range(len(dir_file_list)):
        old_file_path = 'car/{}/easy/{}'.format(folder,dir_file_list[i])
        new_file_path = 'car_orthogonal/{}/{}'.format(str(count).zfill(6),dir_list[i])
        copy(old_file_path, new_file_path)
        print(old_file_path,new_file_path)
 
    


