from ultralytics import YOLO

# Dataset
dataset = '/root/Hand_Gesture/datasets/HAGRID_YOLO-V1/hagrid.yaml'

# Load model
model = YOLO('yolov8m.pt')

# Start training
model.train(
    task = 'detect',
    data = dataset,
    imgsz = (640, 480),
    epochs = 50,
    lr0 = 0.0005,
)