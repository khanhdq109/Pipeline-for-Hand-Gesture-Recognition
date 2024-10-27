#######################################
###### Quickly setup the server #######
#######################################

# Install packages
pip install kaggle
pip install numpy pandas matplotlib tqdm
pip install seaborn
pip install opencv-python
pip install scikit-image
pip install scikit-learn
pip install torchsummary
# pip install ultralytics

sudo apt update
sudo apt search vim
sudo apt install vim
sudo apt install zip

# Setup directory
cd ..
mkdir datasets models
cd datasets
mkdir JESTER-V1
cd JESTER-V1
mkdir images annotations
cd ../../models
mkdir detect classify
cd ../src

# Setup KAGGLE
export KAGGLE_CONFIG_DIR="$(realpath data_preprocessing)"
chmod 600 data_preprocessing/kaggle.json
cd data_preprocessing

# Download and setup JESTER-V1
chmod +x jester_preprocessing.sh
./jester_preprocessing.sh
# ./hagrid_preprocessing.sh
python jester-v1-small.py 0.01 # Create a small version of JESTER-V1
rm ../../datasets/JESTER-V1/images/20bn-jester-v1-videos.zip