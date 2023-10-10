import kaggle

# Install the Kaggle API client
kaggle.api.authenticate()

# Specify the dataset you want to download (replace with the dataset name)
# dataset_name = 'kylecloud/20bn-jester-v1-videos'
dataset_name = 'khnhoquc/hagrid-yolo-v1'

# Set the destination folder where you want to save the dataset
destination_folder = '../../datasets/JESTER-V1/images'

# Download the dataset
print('Downloading the dataset...')
kaggle.api.dataset_download_files(dataset_name, path = destination_folder, unzip = False)
print('Dataset downloaded successfully!')
