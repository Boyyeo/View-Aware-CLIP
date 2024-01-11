import os 
import subprocess

import argparse


if __name__ == "__main__":
    FOLDER_PATH = 'mvdream_result'
    parser = argparse.ArgumentParser()
    parser.add_argument("--control_file", type=str, default=None,required=True)
    parser.add_argument("--gpu", type=str, default=None,required=True)
    args = parser.parse_args()

    f = open(args.control_file, "r")
    count = 0
    for subject_name in f:
        subject_name = subject_name.replace('\n','')
        print('Executing subject name :{}'.format(subject_name))
        p = subprocess.Popen('python generate_all_control.py --subject_folder_name {} --gpu {}'.format(subject_name,args.gpu), shell=True)
        p.wait()
        count += 1
        print("Folder count:",count)
       
  


  