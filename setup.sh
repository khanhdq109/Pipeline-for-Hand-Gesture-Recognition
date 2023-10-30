#######################################
###### Quickly setup the server #######
#######################################

# Install packages
pip install ultralytics
pip install scikit-image
pip install kaggle

sudo apt update
sudo apt search vim
sudo apt install vim

sudo apt-get install libl1-mesa-glx
sudo ldconfig
ls -l /usr/lib/x86_64-linux-gnu/libGL.so.1

# Setup directory
cd /root/Hand_Gesture
mkdir datasets
cd datasets
mkdir JESTER-V1
cd JESTER-V1
mkdir images annotations

# Setup KAGGLE
export KAGGLE_CONFIG_DIR=/root/Hand_Gesture/src/data_preprocessing
chmod 600 /root/Hand_Gesture/src/data_preprocessing/kaggle.json
cd /root/Hand_Gesture/src/data_preprocessing

# Download and setup JESTER-V1
chmod +x jester_preprocessing.sh
./jester_preprocessing.sh
rm /root/Hand_Gesture/datasets/JESTER-V1/images/20bn-jester-v1-videos.zip