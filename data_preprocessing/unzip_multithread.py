"""
    Unzip a large .zip file
"""
import sys
from pathlib import Path 
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor

# unzip files from an archive 
def unzip_files(zipObj, filenames, path):
    # unzip multiple files 
    for file in filenames:
        # unzip the file 
        zipObj.extract(file, path)

# unzip a large number of files
def main(
    src_path = '/root/Hand_Gesture/datasets/20bn-jester-v1-videos.zip', 
    loc_path = '/root/Hand_Gesture/datasets/JESTER-V1/images'
):
    Path(loc_path).mkdir(exist_ok = True)

    # == open the zip file == # 
    with ZipFile(src_path, 'r') as zipObj:

        # list of all files to unzip 
        files = zipObj.namelist() 
        
        # determine chunksize 
        n_workers = 100 
        chunksize = round(len(files) / n_workers)

        # start the thread pool 
        with ThreadPoolExecutor(n_workers) as exe: 
            # split the copy operations into chunks 
            for i in range(0, len(files), chunksize):
                # select a chunk(= mini-batch) of filenames
                filenames = files[i:(i+chunksize)]

                # submit the batch copy task 
                _ = exe.submit(unzip_files, zipObj, filenames, loc_path)

if __name__ == '__main__':
    # Get argument
    arg = sys.argv[1]
    arg = str.upper(arg)
    
    print('Starting unzip_multithread.py')
    
    # HAGRID_YOLO-V1
    src_path_hagrid = '/root/Hand_Gesture/datasets/hagrid-yolo-v1.zip'
    loc_path_hagrid = '/root/Hand_Gesture/datasets'
    
    # JESTER-V1
    src_path_jester = '/root/Hand_Gesture/datasets/20bn-jester-v1-videos.zip'
    # loc_path_jester = '/root/Hand_Gesture/datasets/JESTER-V1/images'
    loc_path_jester = 'D:/Khanh/Others/Hand_Gesture/datasets/JESTER-V1/images' # Delete
    
    if arg == 'JESTER':
        main(
            src_path = src_path_jester,
            loc_path = loc_path_jester
        )
        print('Unzip JESTER-V1 completely!!!')
    elif arg == 'HAGRID':
        main(
            src_path = src_path_hagrid,
            loc_path = loc_path_hagrid
        )
        print('Unzip HAGRID-YOLO completely!!!')
    else:
        raise ValueError('Invalid mode!')