import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import JesterV1, collate_fn
from network.R3D import R3D

"""
    Support functions...
"""
def load_pretrained_weights(model, pre_trained_path, device):
    if pre_trained_path:
        model.load_state_dict(
            torch.load(pre_trained_path, map_location = device),
            strict = False
        )
        print('Pre-trained model loaded successfully!')

def save_model(model, model_arch, block_arch, nmp, epoch, pre_trained_epochs, num_frames, nl_nums = 0):
    if nl_nums == 0:
        name = f'../models/classify/{model_arch.upper()}/{model_arch}-{block_arch}{nmp}_{epoch + pre_trained_epochs + 1}-epochs_{num_frames}frs.pth'
    else:
        name = f'../models/classify/{model_arch.upper()}/{model_arch}-{block_arch}{nmp}_{nl_nums}-nl_{epoch + pre_trained_epochs + 1}-epochs_{num_frames}frs.pth'
    torch.save(model.state_dict(), name)
    
def save_metrics(metrics_dict, json_path):
    with open(json_path, 'w') as json_file:
        json.dump(metrics_dict, json_file)

# Check for GPU availability
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('GPU is available')
    # Get the name of the GPU
    print('GPU Device Name:', torch.cuda.get_device_name(0))
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
small_version = True
## Model parameters
model_arch = 'r3d'
block_arch = 34
phi = 0.5
growth_rate = 12
nl_nums = 3
pre_trained = False
pre_trained_path = ''
if pre_trained:
    pre_trained_epochs = int(pre_trained_path.split('_')[-1].split('-')[0])
else:
    pre_trained_epochs = 0
no_max_pool = True
if no_max_pool:
    nmp = '_0-mp'
else:
    nmp = '_1-mp'
widen_factor = 1.0
dropout = 0.0
n_classes = 27
## Training parameters
num_epochs = 1
learning_rate = 0.001
decay_step = 5 # Decay the learning rate after n epochs
gamma = 0.1 # Decay the learning rate by gamma
validation_interval = 1 # Perform validation after every n epochs
save_interval = 1 # Save model after every n epochs

# Define dataset
data_dir = '../datasets/JESTER-V1'

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
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, collate_fn = collate_fn) # Train data loader
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, collate_fn = collate_fn) # Validation data loader

# Define model
model = R3D(
    block_arch,
    n_input_channels = 3,
    conv1_t_size = 7,
    conv1_t_stride = 1,
    no_max_pool = no_max_pool,
    widen_factor = widen_factor,
    nl_nums = nl_nums,
    n_classes = n_classes
).to(device)

# Load pre-trained weights if pre_trained is True
if pre_trained:
    load_pretrained_weights(model, pre_trained_path, device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Create a learning rate scheduler
scheduler = StepLR(optimizer, step_size = decay_step, gamma = gamma)

# Start training
epochs, train_loss, train_acc, val_acc = [], [], [], [] # Define lists to store the training and validation metrics
print(f'Start training {model_arch.upper()}-{block_arch}{nmp} for {num_epochs} epochs')

total_train_batches = len(train_loader) # Total number of train batches
total_val_batches = len(val_loader) # Total number of validation batches
total_train_samples = train_dataset.__len__() # Total number of train samples
total_val_samples = val_dataset.__len__() # Total number of validation samples
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
            # Update the progress bar
            pbar.update(1)
            pbar.set_postfix({'Train Loss': loss.item()}, refresh = False)
        
        # Calculate the average loss and training accuracy
        average_loss = total_loss / total_train_batches
        train_accuracy = total_correct / total_train_samples
        # Update the progress bar
        pbar.set_postfix({'Average Loss': average_loss, 'Training Accuracy': train_accuracy})
        
        # Append the training metrics to the lists
        epochs.append(epoch + pre_trained_epochs + 1)
        train_loss.append(average_loss)
        train_acc.append(train_accuracy)
        
        # Update the learning rate at the end of each epoch
        scheduler.step()
      
    # Perform validation every <validation_interval> epochs
    if (epoch + pre_trained_epochs + 1) % validation_interval == 0:
        # Set the model to evaluation mode
        model.eval()
        total_correct = 0
        
        with tqdm(total = total_val_batches, unit = 'batch') as pbar: # Initialize the progress bar
            pbar.set_description(f'Epoch {epoch + pre_trained_epochs + 1} - Validation')
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
                    # Update the progress bar
                    pbar.update(1)
            
            # Calculate validation accuracy
            val_accuracy = total_correct / total_val_samples
            # Update the progress bar
            pbar.set_postfix({'Validation Accuracy': val_accuracy})
            
            # Append the validation metrics to the list
            val_acc.append(val_accuracy)
            
    # Save the model
    if (epoch + 1) % save_interval == 0:
        save_model(
            model,
            model_arch, block_arch, nmp,
            epoch, pre_trained_epochs,
            num_frames,
            nl_nums
        )

print('Training sucessfully!!!')

# Save all metrics as a json file
metrics_dict = {
    # Model parameters
    'model_arch': model_arch,
    'block_arch': block_arch,
    'nmp': nmp,
    'num_frames': num_frames,
    'nl_nums': nl_nums,
    'pre_trained': pre_trained,
    # Metrics
    'epochs': epochs,
    'train_loss': train_loss,
    'train_acc': train_acc,
    'val_acc': val_acc,
}
save_metrics(metrics_dict, 'metrics.json')