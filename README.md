# Pipeline for Hand Gesture Recognition
Develops a real-time dynamic hand gesture recognition system using the HaGRID and JESTER datasets. It includes detection and classification components for accurate performance. Below are the project structure, setup, and execution details.

## Project structure
* [setup.sh](./setup.sh)                    : bash file to setup environment and download dataset
* [detect_train.py](./detect_train.py)      : train detection models
* [train.sh](./train.sh)                    : bash file to train classify models from scratch or pre-trained models
* [detect_eval.py](./detect_eval.py)        : evaluate detection models
* [classify_eval.py](./classify_eval.py)    : evalutate classification models
* [program.py](./program.py)                : run the program on real-time camera

## Dataset
### Detection
In this project, we use [HaGRID](https://github.com/hukenovs/hagrid) dataset for detection task. You can download the version of the dataset for YOLO here:
* [HAGRID-YOLO-V1](https://www.kaggle.com/datasets/khnhoquc/hagrid-yolo-v1)

### Classification
For classification task, we use ***JESTER*** dataset. You can download it here:
* [20BN_jester_V1_videos](https://www.kaggle.com/datasets/kylecloud/20bn-jester-v1-videos)

## Setup
Setup directories: `Hand_Gesture/src`.  
After clone this repository in `src`, run this command to install requirement packages, download and setup dataset:
```
./setup.sh
```
> **CAUTION:** You must have at least 70GB of memory available.

## Execute
To train, evaluate models as well as run the program, you have to manually tune parameters from the source code.
### Detection
Train:
```
python detect_train.py
```
Evaluate (***image*** or ***video*** mode):
```
python detect_eval.py
```
### Classification
Run this command to train model as well as save metrics:
> You can set ***small_version = True*** in ***classify_train.py*** to run a demo with a small version of the dataset.
```
./train.sh
```
Evaluate:
```
python classify_eval.py
```
Run the model with real-time camera:
```
python program.py
```

## Author
Quoc Khanh