""" 
    The order of image frames of this dataset is not regular.
    Some clips include not evenly ordered file names. 
    Align the names in sequential.
    (i.e) 00001.jpg, 00003.jpg, 00004.jpg -> 00001.jpg, 00002.jpg, 00003.jpg
"""
import os
from tqdm import tqdm 

def new_name(target_dir):
    fileEx = ".jpg"

    files = sorted([file_ for file_ in os.listdir(target_dir) if file_.endswith(fileEx)])
    for idx, item in enumerate(files):
        old_name = os.path.join(target_dir, item)
        new_name = os.path.join(target_dir, f"{idx+1:05}.jpg")
        os.rename(old_name, new_name)

def main(): 
    dataset_path = os.path.join('../../datasets/JESTER-V1/images')
    
    # list of all files to rename 
    path_dirs = [os.path.join(dataset_path, dir) for dir in os.listdir(dataset_path)]

    for target_path in tqdm(path_dirs):
        new_name(target_path)

if __name__ == '__main__':
    print('\nStarting change_name.py')
    
    main()
    
    print('Change name completely!!!')