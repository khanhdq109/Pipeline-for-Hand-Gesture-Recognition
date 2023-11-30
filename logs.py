""" 
    Write something here (reasons why I create this file)
"""

import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from pathlib import Path

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def main():
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
    
    # Load metrics from the json file
    with open('metrics.json', 'r') as json_file:
        loaded_metrics = json.load(json_file)
    
    # Extract model parameter
    model_arch = loaded_metrics['model_arch']
    block_arch = loaded_metrics['block_arch']
    nmp = loaded_metrics['nmp']
    pre_trained = loaded_metrics['pre_trained']
    pre_trained_epochs = loaded_metrics['pre_trained_epochs']
    # Extract temporary variables
    val_true_labels = loaded_metrics['val_true_labels']
    val_predicted_labels = loaded_metrics['val_predicted_label']
    # Extract individual metrics lists
    epochs = loaded_metrics['epochs']
    train_loss = loaded_metrics['train_loss']
    train_acc = loaded_metrics['train_acc']
    val_acc = loaded_metrics['val_acc']
    train_precision_list = loaded_metrics['train_precision_list']
    train_recall_list = loaded_metrics['train_recall_list']
    train_f1_list = loaded_metrics['train_f1_list']
    val_precision_list = loaded_metrics['val_precision_list']
    val_recall_list = loaded_metrics['val_recall_list']
    val_f1_list = loaded_metrics['val_f1_list']
    
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

if __name__ == '__main__':
    main()