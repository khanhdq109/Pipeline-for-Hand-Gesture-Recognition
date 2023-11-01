# Instructions to pre-process JESTER-V1 dataset

# Download dataset
python kaggle_download.py JESTER
# Unzip .zip file
python unzip_multithread.py JESTER
# Remove .zip file
rm /root/Hand_Gesture/datasets/JESTER-V1/images/20bn-jester-v1-videos.zip
# Change name of images
python change_name.py
# Generate annotation files
python gen_annotation.py
# Create small version of dataset
python jester-v1-small.py