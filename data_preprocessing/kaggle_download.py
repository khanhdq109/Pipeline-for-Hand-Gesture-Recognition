import sys
import kaggle
from pathlib import Path

def main():
    # Get argument
    arg = sys.argv[1]
    arg = str.upper(arg)
    
    # Install the Kaggle API client
    kaggle.api.authenticate()

    # Specify the dataset you want to download (replace with the dataset name)
    if arg == 'JESTER':
        dataset_name = 'kylecloud/20bn-jester-v1-videos'
        destination_folder = '../../datasets/JESTER-V1/images'
    elif arg == 'HAGRID':
        dataset_name = 'khnhoquc/hagrid-yolo-v1'
        destination_folder = '../../datasets'
    else:
        raise ValueError('Invalid mode!')

    # Create path if not exist
    Path(destination_folder).mkdir(exist_ok = True)

    # Download the dataset
    print('Downloading the dataset...')
    kaggle.api.dataset_download_files(dataset_name, path = destination_folder, unzip = False)
    print('Dataset downloaded successfully!') 

if __name__ == '__main__':
    main()