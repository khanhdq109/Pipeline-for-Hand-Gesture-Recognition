from ultralytics import YOLO

# Dataset
dataset = '/root/Hand_Gesture/datasets/HAGRID_YOLO-V1/hagrid.yaml'

# Load model
model = YOLO('model/detect/v8n_30.pt')

# Start training
model.train(
    task = 'detect',
    data = dataset, 
    imgsz = (640, 480),
    epochs = 10,
    lr0 = 0.001,
)