import json
import sys
<<<<<<< HEAD
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
=======
>>>>>>> 74e9a99db9159bcd9c33a42ae704f8367e65b7d3
from tqdm import tqdm

import torch
import torch.nn as nn 
import torch.optim as optim
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
num_workers = 8 # Number of threads for data loading
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
n_classes = 27
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
        train_accuracy = total_correct / total_train_batches
        # Update the progress bar
        pbar.set_postfix({'Average Loss': average_loss, 'Training Accuracy': train_accuracy})
        
        # Calculate Precision, Recall and F1-score for training
        correct_by_class = [0] * n_classes
        actual_by_class = [0] * n_classes
        predicted_by_class = [0] * n_classes
        
        for true_label, predicted_label in zip(train_true_labels, train_predicted_labels):
            actual_by_class[true_label] += 1
            predicted_by_class[predicted_label] += 1
            if true_label == predicted_label:
                correct_by_class[true_label] += 1
                
        train_precision = sum(correct_by_class) / sum(predicted_by_class) if sum(predicted_by_class) != 0 else 0
        train_recall = sum(correct_by_class) / sum(actual_by_class) if sum(actual_by_class) != 0 else 0
        train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) != 0 else 0
        
        # Append the training metrics to the lists
        epochs.append(epoch + pre_trained_epochs + 1)
        train_loss.append(average_loss)
        train_acc.append(train_accuracy)
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
            val_accuracy = total_correct / total_val_batches
            # Update the progress bar
            pbar.set_postfix({'Validation Accuracy': val_accuracy})
            
            # Calculate Precision, Recall and F1-score for validation
            correct_by_class = [0] * n_classes
            actual_by_class = [0] * n_classes
            predicted_by_class = [0] * n_classes
                
            for true_label, predicted_label in zip(val_true_labels, val_predicted_labels):
                actual_by_class[true_label] += 1
                predicted_by_class[predicted_label] += 1
                if true_label == predicted_label:
                    correct_by_class[true_label] += 1
                    
            val_precision = sum(correct_by_class) / sum(predicted_by_class) if sum(predicted_by_class) != 0 else 0
            val_recall = sum(correct_by_class) / sum(actual_by_class) if sum(actual_by_class) != 0 else 0
            val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) != 0 else 0
            
            # Append the validation metrics to the list
            val_acc.append(val_accuracy)
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

# Save all metrics as a json file
metrics_dict = {
    # Model parameters
    'model_arch': model_arch,
    'block_arch': block_arch,
    'nmp': nmp,
    'pre_trained': pre_trained,
    'pre_trained_epochs': pre_trained_epochs,
    # Temporary variables
    'val_true_labels': val_true_labels,
    'val_predicted_label': val_predicted_labels,
    # Metrics
    'epochs': epochs,
    'train_loss': train_loss,
    'train_acc': train_acc,
    'val_acc': val_acc,
    'train_precision_list': train_precision_list,
    'train_recall_list': train_recall_list,
    'train_f1_list': train_f1_list,
    'val_precision_list': val_precision_list,
    'val_recall_list': val_recall_list,
    'val_f1_list': val_f1_list
}
json_path = 'metrics.json'
with open(json_path, 'w') as json_file:
    json.dump(metrics_dict, json_file)
