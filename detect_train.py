from ultralytics import YOLO

# Dataset
dataset = '../datasets/HAGRID_YOLO-V1/hagrid.yaml'

# Load model
model = YOLO('model/detect/v8m_150.pt')

# Start training
model.train(
    task = 'detect',
    data = dataset,
    imgsz = (640, 480),
    epochs = 50,
    lr0 = 0.0002,
)