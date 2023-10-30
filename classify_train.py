import matplotlib.pyplot as plt 

import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader 
from torchvision import transforms

from data_loader import JesterV1
from network.R3D import R3D

import warnings
warnings.filterwarnings('ignore', message = 'The default value of the antialias parameter of all the resizing transforms*')

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
num_frames = 30
batch_size = 1
num_epochs = 1
learning_rate = 0.001
num_workers = 4 # Number of threads for data loading
validation_interval = 1 # Perform validation every n epochs

# Define dataset
data_dir = '/root/Hand_Gesture/datasets/JESTER-V1'

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((96, 96)),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])

# Create an instance of the dataset
train_dataset = JesterV1(
    data_dir = data_dir,
    num_frames = 30,
    transform = transform,
    mode = 'train'
) # Train dataset
val_dataset = JesterV1(
    data_dir = data_dir,
    num_frames = 30,
    transform = transform,
    mode = 'val'
) # Validation dataset

# Create a DataLoader
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers) # Train data loader
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers) # Validation data loader

# Define model
model = R3D(
    18,
    n_input_channels = 3,
    conv1_t_size = 7,
    conv1_t_stride = 1,
    no_max_pool = False,
    widen_factor = 1.0,
    n_classes = 28
).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Start training
print('Start training...')
for epoch in range(num_epochs):
    # Set the model to train mode
    model.train()
    total_loss = 0.0 # Total loss for the epoch
    # Iterate over the train_loader
    for frames, labels in train_loader:
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
    # Calculate the average loss
    average_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1} / {num_epochs}] - Train Loss: {average_loss:.4f}")
    
    # Perform validation every <validation_interval> epochs
    if (epoch + 1) % validation_interval == 0:
        # Set the model to evaluation mode
        model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for frames, labels in val_loader:
                # Move the frames and labels to GPU
                frames, labels = frames.to(device), labels.to(device)
                # Forward pass
                outputs = model(frames)
                # Get the predictions
                _, predictions = torch.max(outputs.data, 1)
                # Update the total number of correct predictions
                total_correct += (predictions == labels).sum().item()
                # Update the total number of samples
                total_samples += predictions.shape[0]
        # Calculate validation accuracy
        validation_accuracy = total_correct / total_samples
        print(f"Validation Accuracy after {epoch + 1} epochs: {validation_accuracy:.2f}%")
print('Training sucessfully!!!')

# Save the model
torch.save(model.state_dict(), 'model/classify/r3d-18_demo.pth')