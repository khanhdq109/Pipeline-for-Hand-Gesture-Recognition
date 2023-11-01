#######################################
###### Quickly setup the server #######
#######################################

# Install packages
pip install kaggle
pip install numpy pandas matplotlib tqdm
pip install opencv-python
pip install scikit-image

sudo apt update
sudo apt search vim
sudo apt install vim

sudo apt update
sudo apt install zip

# Setup directory
cd /root/Hand_Gesture
mkdir datasets models
cd datasets
mkdir JESTER-V1
cd JESTER-V1
mkdir images annotations
cd /root/Hand_Gesture/models
mkdir classify

# Setup KAGGLE
export KAGGLE_CONFIG_DIR=/root/Hand_Gesture/source/data_preprocessing
chmod 600 /root/Hand_Gesture/source/data_preprocessing/kaggle.json
cd /root/Hand_Gesture/source/data_preprocessing

# Download and setup JESTER-V1
chmod +x jester_preprocessing.sh
./jester_preprocessing.sh