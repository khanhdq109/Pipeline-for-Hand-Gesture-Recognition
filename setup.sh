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
export KAGGLE_CONFIG_DIR=/root/Hand_Gesture/src/data_preprocessing
chmod 600 /root/Hand_Gesture/src/data_preprocessing/kaggle.json
cd /root/Hand_Gesture/src/data_preprocessing

# Download and setup JESTER-V1
chmod +x jester_preprocessing.sh
./jester_preprocessing.sh
python jester-v1-small.py 0.1 # Create a small version of JESTER-V1
rm /root/Hand_Gesture/datasets/JESTER-V1/images/20bn-jester-v1-videos.zip