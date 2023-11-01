import pandas as pd
from pathlib import Path

f = 'logs'
Path(f).mkdir(exist_ok = True)

epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train_loss = [1.53, 0.45, 0.285, 0.194, 0.13, 0.0895, 0.062, 0.0479, 0.0373, 0.0323]
train_acc = [0.526, 0.855, 0.908, 0.937, 0.957, 0.97, 0.979, 0.983, 0.988, 0.989]
val_acc = [0.632, 0.799, 0.809, 0.834, 0.826, 0.83, 0.82, 0.839, 0.839, 0.818]

log = {
    'epochs': epochs,
    'train_loss': train_loss,
    'train_acc': train_acc,
    'val_acc': val_acc
}
log = pd.DataFrame(log)

log.to_csv('logs/r3d-18_0-mp.csv', index = False)