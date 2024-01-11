
# Importing required module
import subprocess
 
# Using system() method to
# execute shell commands
count = 0
f = open("object_eval.txt", "r")
for name in f:
  subject_name = name.replace('\n','') 
  p = subprocess.Popen('CUDA_VISIBLE_DEVICES=2 python scripts/t2i.py --text "a photo of a {}" --subject_name {} --seed 42'.format(subject_name,subject_name), shell=True)
  p.wait()
  count += 1
  print("count:",count)
  #if count == 2:
  #  break
#https://bobbyhadz.com/blog/wait-process-until-all-subprocesses-finish-in-python