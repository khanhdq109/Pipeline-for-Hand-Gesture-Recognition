import os
import pandas as pd
import matplotlib.pyplot as plt

r3d_18 = pd.read_csv('logs/r3d-18_0-mp/train.csv')
r3d_34 = pd.read_csv('logs/r3d-34_0-mp/train.csv')
r3d_50 = pd.read_csv('logs/r3d-50_0-mp/train.csv')
r2plus1d_18 = pd.read_csv('logs/r2plus1d-18_0-mp/train.csv')
d3d_121 = pd.read_csv('logs/d3d-121_0-mp/train.csv')
t3d_121 = pd.read_csv('logs/t3d-121_0-mp/train.csv')

epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

r3d_18_train = r3d_18['train_acc']
r3d_18_val = r3d_18['val_acc']

r3d_34_train = r3d_34['train_acc']
r3d_34_val = r3d_34['val_acc']

r3d_50_train = r3d_50['train_acc']
r3d_50_val = r3d_50['val_acc']

r2plus1d_18_train = r2plus1d_18['train_acc']
r2plus1d_18_val = r2plus1d_18['val_acc']

d3d_121_train = d3d_121['train_acc']
d3d_121_val = d3d_121['val_acc']

t3d_121_train = t3d_121['train_acc'][:10]
t3d_121_val = t3d_121['val_acc'][:10]

# Train accuracy
plt.figure(figsize = (12, 6))
plt.plot(epochs, r3d_18_train, label = 'R3D-18', color = 'blue')
plt.plot(epochs, r3d_34_train, label = 'R3D-34', color = 'orange')
plt.plot(epochs, r3d_50_train, label = 'R3D-50', color = 'green')
plt.plot(epochs, r2plus1d_18_train, label = 'R(2+1)D-18', color = 'red')
plt.plot(epochs, d3d_121_train, label = 'D3D-121', color = 'black')
plt.plot(epochs, t3d_121_train, label = 'T3D-121', color = 'purple')
plt.title('Train accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('logs/train_acc.png')

# Val accuracy
plt.figure(figsize = (12, 6))
plt.plot(epochs, r3d_18_val, label = 'R3D-18', color = 'blue')
plt.plot(epochs, r3d_34_val, label = 'R3D-34', color = 'orange')
plt.plot(epochs, r3d_50_val, label = 'R3D-50', color = 'green')
plt.plot(epochs, r2plus1d_18_val, label = 'R(2+1)D-18', color = 'red')
plt.plot(epochs, d3d_121_val, label = 'D3D-121', color = 'black')
plt.plot(epochs, t3d_121_val, label = 'T3D-121', color = 'purple')
plt.title('Val accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('logs/val_acc.png')