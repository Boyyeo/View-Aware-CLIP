import os 
import subprocess

FOLDER_PATH = 'mvdream_result'

subfolder_list = []
for subfolder_name in os.listdir(FOLDER_PATH):
        subfolder_list.append(subfolder_name)

gpu_list = [0,1]
count = 0
while len(subfolder_list) > 0:

    sub_list = []
    for i in range(len(gpu_list)):
        sub_list.append(subfolder_list.pop())
        if len(subfolder_list) == 0:
             break
    if len(subfolder_list) == 0:
             break
    for i,subject_name in enumerate(sub_list[:-1]):
           print('python generate_all_control.py --subject_folder_name {} --gpu {}'.format(subject_name,gpu_list[i]))
           subprocess.Popen('python generate_all_control.py --subject_folder_name {} --gpu {}'.format(subject_name,gpu_list[i]), shell=True)

    print('python generate_all_control.py --subject_folder_name {} --gpu {}'.format(sub_list[-1],gpu_list[-1]))
    p = subprocess.Popen('python generate_all_control.py --subject_folder_name {} --gpu {}'.format(sub_list[-1],gpu_list[-1]), shell=True)
    p.wait()
    count += 1
    print("Folder count:",count)
    break
  


  