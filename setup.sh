#######################################
###### Quickly setup the server #######
#######################################

# Install packages
pip install kaggle
pip install numpy pandas matplotlib tqdm
pip install opencv-python
pip install scikit-image
pip install torchsummary

sudo apt update
sudo apt search vim
sudo apt install vim

sudo apt update
sudo apt install zip

# Setup directory
# cd /root/Hand_Gesture
cd D:/Khanh/Others/Hand_Gesture # Delete
mkdir datasets models
cd datasets
mkdir JESTER-V1
cd JESTER-V1
mkdir images annotations
# cd /root/Hand_Gesture/models
cd D:/Khanh/Others/Hand_Gesture/models # Delete
mkdir classify

# Setup KAGGLE
# export KAGGLE_CONFIG_DIR=/root/Hand_Gesture/source/data_preprocessing
export KAGGLE_CONFIG_DIR=D:/Khanh/Others/Hand_Gesture/source/data_preprocessing # Delete
# chmod 600 /root/Hand_Gesture/source/data_preprocessing/kaggle.json
chmod 600 D:/Khanh/Others/Hand_Gesture/source/data_preprocessing/kaggle.json # Delete
# cd /root/Hand_Gesture/source/data_preprocessing
cd D:/Khanh/Others/Hand_Gesture/source/data_preprocessing # Delete

# Download and setup JESTER-V1
chmod +x jester_preprocessing.sh
./jester_preprocessing.sh
# rm /root/Hand_Gesture/datasets/20bn-jester-v1-videos.zip
rm D:/Khanh/Others/Hand_Gesture/datasets/20bn-jester-v1-videos.zip # Delete