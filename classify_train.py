import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import JesterV1
from network.R3D import R3D
from network.R2Plus1D import R2Plus1D

import warnings
warnings.filterwarnings('ignore', message = 'The default value of the antialias parameter of all the resizing transforms*')

""" 
    Full command to train model:
        python classify_train.py <model_arch> <block_arch> <pre_trained>
        
    Example:
        python classify_train.py r3d 18 0 --> Train R3D-18 model from scratch
        python classify_train.py r3d 18 1 --> Train R3D-18 model from pre-trained model
"""

# Categories
class_names = [
    'Swiping Left',
    'Swiping Right',
    'Swiping Down',
    'Swiping Up',
    'Pushing Hand Away',
    'Pulling Hand In',
    'Sliding Two Fingers Left',
    'Sliding Two Fingers Right',
    'Sliding Two Fingers Down',
    'Sliding Two Fingers Up',
    'Pushing Two Fingers Away',
    'Pulling Two Fingers In',
    'Rolling Hand Forward',
    'Rolling Hand Backward',
    'Turning Hand Clockwise',
    'Turning Hand Counterclockwise',
    'Zooming In With Full Hand',
    'Zooming Out With Full Hand',
    'Zooming In With Two Fingers',
    'Zooming Out With Two Fingers',
    'Thumb Up',
    'Thumb Down',
    'Shaking Hand',
    'Stop Sign',
    'Drumming Fingers',
    'No gesture',
    'Doing other things',
]

# Model architecture
if len(sys.argv) >= 2:
    arg1 = sys.argv[1]
    arg1 = str.lower(arg1)
else:
    arg1 = 'r3d'
# Block architecture
if len(sys.argv) >= 3:
    arg2 = sys.argv[2]
    arg2 = int(arg2)
else:
    arg2 = 18
# Training pre-trained model model
if len(sys.argv) >= 4:
    arg3 = sys.argv[3]
    if int(arg3) == 1:
        arg3 = True
    else:
        arg3 = False

# Check for GPU availability
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('GPU is available')
    # Get the name of the GPU
    print('GPU Device Name:', torch.cuda.get_device_name(0)) # Change the device index if you have multiple GPUs
else:
    device = torch.device('cpu')
    print('GPU not available, using CPU instead')
    
print('Selected device:', device)

# Set training parameters
## Data parameters
resize = (112, 112)
num_frames = 30
batch_size = 1
num_workers = 4 # Number of threads for data loading
small_version = False
## Model parameters
model_arch = arg1
block_arch = arg2
pre_trained = arg3
pre_trained_path = '/root/Hand_Gesture/models/classify/R3D/r3d-18_0-mp_10-epochs.pth'
if pre_trained:
    pre_trained_epochs = int(pre_trained_path.split('_')[-1].split('-')[0])
else:
    pre_trained_epochs = 0
no_max_pool = True
widen_factor = 1.0
n_classes = 28
## Training parameters
num_epochs = 10
learning_rate = 0.001
decay_step = 5 # Decay the learning rate after n epochs
gamma = 0.1 # Decay the learning rate by gamma
validation_interval = 1 # Perform validation after every n epochs
save_interval = 1 # Save model after every n epochs

# Define dataset
data_dir = '/root/Hand_Gesture/datasets/JESTER-V1'

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(resize),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])

# Create an instance of the dataset
train_dataset = JesterV1(
    data_dir = data_dir,
    num_frames = num_frames,
    transform = transform,
    mode = 'train',
    small = small_version
) # Train dataset
val_dataset = JesterV1(
    data_dir = data_dir,
    num_frames = num_frames,
    transform = transform,
    mode = 'val',
    small = small_version
) # Validation dataset

# Create a DataLoader
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers) # Train data loader
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers) # Validation data loader

# Define model
if model_arch == 'r3d':
    model = R3D(
        block_arch,
        n_input_channels = 3,
        conv1_t_size = 7,
        conv1_t_stride = 1,
        no_max_pool = no_max_pool,
        widen_factor = widen_factor,
        n_classes = n_classes
    ).to(device)
elif model_arch == 'r2plus1d':
    model = R2Plus1D(
        block_arch,
        n_input_channels = 3,
        conv1_t_size = 7,
        conv1_t_stride = 1,
        no_max_pool = no_max_pool,
        widen_factor = widen_factor,
        n_classes = n_classes
    ).to(device)
else:
    raise ValueError('Invalid model architecture!')

# Load pre-trained weights if pre_trained is True
if pre_trained:
    model.load_state_dict(torch.load(pre_trained_path), strict = False)
    print('Pre-trained model loaded successfully!')

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Create a learning rate scheduler
scheduler = StepLR(optimizer, step_size = decay_step, gamma = gamma)

# Lists to store the true and predicted labels for training and validation
train_true_labels = []
train_predicted_labels = []
val_true_labels = []
val_predicted_labels = []

# Start training
epochs, train_loss, train_acc, val_acc, train_precision_list, train_recall_list, train_f1_list, val_precision_list, val_recall_list, val_f1_list = [], [], [], [], [], [], [], [], [], [] # Define lists to store the training and validation metrics
print('Start training...')

total_train_batches = len(train_loader) # Total number of train batches
total_val_batches = len(val_loader) # Total number of validation batches
for epoch in range(num_epochs):
    # Set the model to train mode
    model.train()
    total_loss = 0.0 # Total loss for the epoch
    total_correct = 0
    
    with tqdm(total = total_train_batches, unit = 'batch') as pbar: # Initialize the progress bar
        pbar.set_description(f'Epoch {epoch + pre_trained_epochs + 1} - Training')
        for batch_idx, (frames, labels) in enumerate(train_loader):
            # Move the frames and labels to device
            frames, labels = frames.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(frames)
            # Calculate the loss
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Update the parameters
            optimizer.step()
            # Add the loss to the total loss
            total_loss += loss.item()
            # Get the predictions
            _, predictions = outputs.max(1)
            # Update the total number of correct predictions
            total_correct += (predictions == labels).sum().item()
            # Append the true and predicted labels
            train_true_labels.extend(labels.cpu().numpy())
            train_predicted_labels.extend(predictions.cpu().numpy())
            # Update the progress bar
            pbar.update(1)
            pbar.set_postfix({'Train Loss': loss.item()}, refresh = False)
        
        # Calculate the average loss and training accuracy
        average_loss = total_loss / total_train_batches
        training_accuracy = total_correct / total_train_batches
        # Update the progress bar
        pbar.set_postfix({'Average Loss': average_loss, 'Training Accuracy': training_accuracy})
        # Calculate Precision, Recall and F1-score for training
        train_precision = precision_score(train_true_labels, train_predicted_labels, average = 'weighted')
        train_recall = recall_score(train_true_labels, train_predicted_labels, average = 'weighted')
        train_f1 = f1_score(train_true_labels, train_predicted_labels, average = 'weighted')
        # Append the training metrics to the lists
        epochs.append(epoch + pre_trained_epochs + 1)
        train_loss.append(average_loss)
        train_acc.append(training_accuracy)
        train_precision_list.append(train_precision)
        train_recall_list.append(train_recall)
        train_f1_list.append(train_f1)
        # Update the learning rate at the end of each epoch
        scheduler.step()
        
    # Perform validation every <validation_interval> epochs
    if (epoch + pre_trained_epochs + 1) % validation_interval == 0:
        # Set the model to evaluation mode
        model.eval()
        total_correct = 0
        
        with tqdm(total = total_val_batches, unit = 'batch') as pbar: # Initialize the progress bar
            pbar.set_description(f'Epoch {epoch + 1} - Validation')
            with torch.no_grad():
                for batch_idx, (frames, labels) in enumerate(val_loader):
                    # Move the frames and labels to GPU
                    frames, labels = frames.to(device), labels.to(device)
                    # Forward pass
                    outputs = model(frames)
                    # Get the predictions
                    _, predictions = outputs.max(1)
                    # Update the total number of correct predictions
                    total_correct += (predictions == labels).sum().item()
                    # Append true and predicted labels
                    val_true_labels.extend(labels.cpu().numpy())
                    val_predicted_labels.extend(predictions.cpu().numpy())
                    # Update the progress bar
                    pbar.update(1)
            
            # Calculate validation accuracy
            validation_accuracy = total_correct / total_val_batches
            # Update the progress bar
            pbar.set_postfix({'Validation Accuracy': validation_accuracy})
            # Calculate Precision, Recall and F1-score for validation
            val_precision = precision_score(val_true_labels, val_predicted_labels, average = 'weighted')
            val_recall = recall_score(val_true_labels, val_predicted_labels, average = 'weighted')
            val_f1 = f1_score(val_true_labels, val_predicted_labels, average = 'weighted')
            # Append the validation metrics to the list
            val_acc.append(validation_accuracy)
            val_precision_list.append(val_precision)
            val_recall_list.append(val_recall)
            val_f1_list.append(val_f1)
            
    # Save the model
    if (epoch + 1) % save_interval == 0:
        # Define model name
        if no_max_pool:
            nmp = '_0-mp'
        else:
            nmp = '_1-mp'
        name = f'/root/Hand_Gesture/models/classify/{model_arch}-{block_arch}{nmp}_{epoch + pre_trained_epochs + 1}-epochs.pth'
        # Save model
        torch.save(model.state_dict(), name)

print('Training sucessfully!!!')

# Save the model for the last time
if no_max_pool:
    nmp = '_0-mp'
else:
    nmp = '_1-mp'
name = f'/root/Hand_Gesture/models/classify/{model_arch}-{block_arch}{nmp}_{epoch + pre_trained_epochs + 1}-epochs.pth'
# Save model
torch.save(model.state_dict(), name)

# Save the training metrics
if pre_trained:
    # Load the previous log
    log = pd.read_csv('logs/' + model_arch + '-' + str(block_arch) + nmp + '.csv')
    # Append the new log
    new_metrics = {
        'epochs': epochs,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'train_precision': train_precision_list,
        'train_recall': train_recall_list,
        'train_f1': train_f1_list,
        'val_acc': val_acc,
        'val_precision': val_precision_list,
        'val_recall': val_recall_list,
        'val_f1': val_f1_list
    }
    log = log.append(pd.DataFrame(new_metrics), ignore_index = True)
    # Save the log
    log.to_csv('logs/' + model_arch + '-' + str(block_arch) + nmp + '.csv', index = False)
else:     
    Path('logs').mkdir(exist_ok = True)
    log = {
        'epochs': epochs,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'train_precision': train_precision_list,
        'train_recall': train_recall_list,
        'train_f1': train_f1_list,
        'val_acc': val_acc,
        'val_precision': val_precision_list,
        'val_recall': val_recall_list,
        'val_f1': val_f1_list
    }
    log = pd.DataFrame(log)
    log.to_csv('logs/' + model_arch + '-' + str(block_arch) + nmp + '.csv', index = False)
    
# Calculate the confusion matrix for validation data
val_conf_matrix = confusion_matrix(val_true_labels, val_predicted_labels)
# Visualize and save confusion matrix after training
plt.figure(figsize = (12, 12))
sns.heatmap(val_conf_matrix, annot = True, fmt = 'd', xticklabels = class_names, yticklabels = class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('logs/plots/' + model_arch + '-' + str(block_arch) + nmp + '/confusion_matrix.png')

# Visualize and save metrics for training and validation
## Read the log file
logs = pd.read_csv('logs/' + model_arch + '-' + str(block_arch) + nmp + '.csv')
## Extract the metrics
epochs = logs['epochs']
train_loss = logs['train_loss']
train_acc = logs['train_acc']
train_precision = logs['train_precision']
train_recall = logs['train_recall']
train_f1 = logs['train_f1']
val_acc = logs['val_acc']
val_precision = logs['val_precision']
val_recall = logs['val_recall']
val_f1 = logs['val_f1']
## Create folder to save plots
save_folder = 'logs/plots/' + model_arch + '-' + str(block_arch) + nmp
Path(save_folder).mkdir(exist_ok = True)
## Train loss
plt.figure(figsize = (12, 6))
plt.plot(epochs, train_loss, label = 'Train Loss', color = 'red')
plt.title('Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(save_folder, 'train_loss.png'))
## Accuracy
plt.figure(figsize = (12, 6))
plt.plot(epochs, train_acc, label = 'Train Accuracy', color = 'red')
plt.plot(epochs, val_acc, label = 'Validation Accuracy', color = 'blue')
plt.title('Accuracy for Training and Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(save_folder, 'accuracy.png'))
## Precision
plt.figure(figsize = (12, 6))
plt.plot(epochs, train_precision, label = 'Train Precision', color = 'red')
plt.plot(epochs, val_precision, label = 'Validation Precision', color = 'blue')
plt.title('Precision for Training and Validation')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()
plt.savefig(os.path.join(save_folder, 'precision.png'))
## Recall
plt.figure(figsize = (12, 6))
plt.plot(epochs, train_recall, label = 'Train Recall', color = 'red')
plt.plot(epochs, val_recall, label = 'Validation Recall', color = 'blue')
plt.title('Recall for Training and Validation')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.savefig(os.path.join(save_folder, 'recall.png'))
## F1-score
plt.figure(figsize = (12, 6))
plt.plot(epochs, train_f1, label = 'Train F1-score', color = 'red')
plt.plot(epochs, val_f1, label = 'Validation F1-score', color = 'blue')
plt.title('F1-score for Training and Validation')
plt.xlabel('Epochs')
plt.ylabel('F1-score')
plt.legend()
plt.savefig(os.path.join(save_folder, 'f1-score.png'))
