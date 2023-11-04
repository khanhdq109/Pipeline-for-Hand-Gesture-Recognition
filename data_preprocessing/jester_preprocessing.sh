# Instructions to pre-process JESTER-V1 dataset

# Download dataset
python kaggle_download.py JESTER
# Unzip .zip file
python unzip_multithread.py JESTER
# Change name of images
# python change_name.py
# Generate annotation files
python gen_annotation.py