# Instructions to pre-process JESTER-V1 dataset

# Download dataset
pwd
python kaggle_download.py JESTER
# Unzip .zip file
python unzip_multithread.py JESTER
# Generate annotation files
python gen_annotation.py