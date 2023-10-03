from ultralytics import YOLO

# Dataset
dataset = 'D:\Khanh\Others\Hand_Gesture\datasets\HAGRID_YOLO\hagrid.yaml'

# Load model
model = YOLO('model/v8n_20.pt')

# Start training
model.train(
    task = 'detect',
    data = dataset, 
    imgsz = (640, 480),
    epochs = 10,
    lr0 = 0.001,
)