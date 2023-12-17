""" 
    Save the training metrics and visualize the metrics after training.
"""

import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import JesterV1
from network.T3D import D3D
# from network.R3D import R3D

def save_training_metrics(
    model_arch, block_arch, nmp, pre_trained,
    epochs, train_loss, train_acc, val_acc
):  
    print('=' * 80)
    print('Saving training metrics...')
    
    # Save the training metrics
    if pre_trained:
        # Load the previous log
        log = pd.read_csv(f'logs/{model_arch}-{block_arch}{nmp}/train.csv')
        # Append the new log
        new_metrics = {
            'epochs': epochs,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }
        new_metrics_df = pd.DataFrame(new_metrics)
        log = pd.concat([log, new_metrics_df], ignore_index = True)
        # Save the log
        log.to_csv(f'logs/{model_arch}-{block_arch}{nmp}/train.csv', index = False)
    else:
        Path(f'logs/{model_arch}-{block_arch}{nmp}').mkdir(exist_ok = True)
        log = {
            'epochs': epochs,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }
        log = pd.DataFrame(log)
        log.to_csv(f'logs/{model_arch}-{block_arch}{nmp}/train.csv', index = False)

    print('Training metrics saved!!!')
    print('=' * 80)

def visualize_training_metrics(
    model_arch, block_arch, nmp
):
    print('Visualizing training metrics...')
    
    # Read the log file
    logs = pd.read_csv(f'logs/{model_arch}-{block_arch}{nmp}/train.csv')
    # Extract the metrics
    epochs = logs['epochs']
    train_loss = logs['train_loss']
    train_acc = logs['train_acc']
    val_acc = logs['val_acc']
    # Create folder to save plots
    save_folder = f'logs/{model_arch}-{block_arch}{nmp}/plots'
    Path(save_folder).mkdir(exist_ok = True)
    
    # Train loss
    plt.figure(figsize = (12, 6))
    plt.plot(epochs, train_loss, label = 'Train Loss', color = 'red')
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_folder, 'train_loss.png'))
    # Accuracy
    plt.figure(figsize = (12, 6))
    plt.plot(epochs, train_acc, label = 'Train Accuracy', color = 'red')
    plt.plot(epochs, val_acc, label = 'Validation Accuracy', color = 'blue')
    plt.title('Accuracy for Training and Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_folder, 'accuracy.png'))
    
    print('Training metrics visualized!!!')
    print('=' * 80)

def eval_on_test(
    model_arch, block_arch, epoch
):
    print('Evaluating on test data...')
    
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
    
    # Set parameters
    resize = (112, 112)
    num_frames = 30
    batch_size = 1
    num_workers = 4 # Number of threads for data loading
    small_version = False
    phi = 0.5
    growth_rate = 12
    no_max_pool = True
    # widen_factor = 1.0
    if no_max_pool:
        nmp = '_0-mp'
    else:
        nmp = '_1-mp'
    dropout = 0.0
    n_classes = 27
    
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
    
    # Create an instance of the dataset an DataLoader
    test_dataset = JesterV1(
        data_dir = data_dir,
        num_frames = num_frames,
        transform = transform,
        mode = 'test',
        small = small_version
    )
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    
    # Load model
    """
    model = R3D(
        block_arch,
        n_input_channels = 3,
        conv1_t_size = 7,
        conv1_t_stride = 1,
        no_max_pool = no_max_pool,
        widen_factor = widen_factor,
        n_classes = n_classes
    ).to(device)
    """
    
    model = D3D(
        block_arch,
        phi = phi,
        growth_rate = growth_rate,
        n_input_channels = 3,
        conv1_t_size = 3,
        conv1_t_stride = 1,
        no_max_pool = no_max_pool,
        n_classes = n_classes,
        dropout = dropout,
    ).to(device)
    
    """
    model = T3D(
        block_arch,
        phi = phi,
        growth_rate = growth_rate,
        temporal_expansion = 1,
        transition_t1_size = [1, 3, 6],
        transition_t_size = [1, 3, 4],
        n_input_channels = 3,
        conv1_t_size = 3,
        conv1_t_stride = 1,
        no_max_pool = no_max_pool,
        n_classes = n_classes,
        dropout = dropout
    ).to(device)
    """
    
    name = f'../models/classify/{model_arch.upper()}/{model_arch}-{block_arch}{nmp}_{epoch}-epochs.pth'
    model.load_state_dict(
        torch.load(name, map_location = device),
        strict = False
    )
    print(f'Model loaded from: {name}')
    
    # Set model to evaluation mode
    model.eval()
    total_correct = 0
    total_test_batches = len(test_loader)
    total_test_samples = test_dataset.__len__()
    # Create empty lists to store true and predicted labels
    true_labels = []
    predicted_labels = []
    
    with tqdm(total = total_test_batches, unit = 'batch') as pbar:
        pbar.set_description(f'Testing')
        # Create an empty confusion matrix
        conf_matrix = np.zeros((n_classes, n_classes), dtype = np.int32)
        # Evaluate the model on test set
        with torch.no_grad():
            for batch_idx, (frames, labels) in enumerate(test_loader):
                frames, labels = frames.to(device), labels.to(device)
                outputs = model(frames)
                _, predicted = outputs.max(1)
                # Update true and predicted labels
                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())
                # Update confusion matrix
                for i in range(len(predicted)):
                    conf_matrix[labels[i], predicted[i]] += 1
                # Update total correct
                total_correct += (predicted == labels).sum().item()
                pbar.update(1)
                
        # Calculate test accuracy
        test_accuracy = total_correct / total_test_samples
        pbar.set_postfix({'Test Accuracy': test_accuracy})
        
        # Calculate Precision, Recall and F1-score for each class
        precision = precision_score(y_true = true_labels, y_pred = predicted_labels, average = None)
        recall = recall_score(y_true = true_labels, y_pred = predicted_labels, average = None)
        f1 = f1_score(y_true = true_labels, y_pred = predicted_labels, average = None)
        # Create a DataFrame to store metrics
        metrics_df = pd.DataFrame({
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1
        })
        # Save metrics to a csv file
        name = f'logs/{model_arch}-{block_arch}{nmp}/test.csv'
        metrics_df.to_csv(name, index_label = 'Class')
        
        # Plot the confusion matrix
        plt.figure(figsize = (12, 12))
        sns.heatmap(conf_matrix, annot = True, fmt = 'g', cmap = 'Blues', xticklabels = class_names, yticklabels = class_names)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        name = f'logs/{model_arch}-{block_arch}{nmp}/confusion_matrix.png'
        plt.savefig(name)
        
    print('Testing completed!!!')
        
def main():
    # Load metrics from the json file
    with open('metrics.json', 'r') as json_file:
        loaded_metrics = json.load(json_file)
    
    # Extract model parameter
    model_arch = loaded_metrics['model_arch']
    block_arch = loaded_metrics['block_arch']
    nmp = loaded_metrics['nmp']
    pre_trained = loaded_metrics['pre_trained']
    # Extract individual metrics lists
    epochs = loaded_metrics['epochs']
    train_loss = loaded_metrics['train_loss']
    train_acc = loaded_metrics['train_acc']
    val_acc = loaded_metrics['val_acc']
    
    '''
    # Extract model parameter
    model_arch = 'r3d'
    block_arch = 34
    nmp = '_0-mp'
    pre_trained = False
    # Extract individual metrics lists
    epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    train_loss = [1.6, 0.447, 0.281, 0.188, 0.127, 0.028, 0.00859, 0.00217, 0.000491, 0.0000373]
    train_acc = [0.511, 0.855, 0.91, 0.939, 0.958, 0.992, 0.998, 1, 1, 1]
    val_acc = [0.708, 0.772, 0.81, 0.84, 0.847, 0.866, 0.863, 0.869, 0.866, 0.878]
    '''
    
    # Save the training metrics
    save_training_metrics(
        model_arch, block_arch, nmp, pre_trained,
        epochs, train_loss, train_acc, val_acc
    )

    # Visualize the training metrics
    visualize_training_metrics(
        model_arch, block_arch, nmp
    )
    
    # Evaluate the model on test set
    eval_on_test(
        model_arch, block_arch, epochs[-1]
    )

if __name__ == '__main__':
    # Define categories
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
    
    main()
