"""
    Unzip a large .zip file
"""
from pathlib import Path 
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor

# unzip files from an archive 
def unzip_files(zipObj, filenames, path):
    # unzip multiple files 
    for file in filenames:
        # unzip the file 
        zipObj.extract(file, path)

        # report progress
        print(f".unzipped {file}")

# unzip a large number of files 
def main(loc_path = "../datasets/JESTER"):
    Path(loc_path).mkdir(exist_ok = True)

    # == open the zip file == # 
    with ZipFile("../datasets/JESTER-V1.zip", "r") as zipObj: 

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
    main()
