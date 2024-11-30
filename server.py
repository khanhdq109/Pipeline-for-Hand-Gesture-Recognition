from flask import Flask, request, jsonify
import torch
from torchvision import transforms
import numpy as np
from network.T3D import T3D
import pickle  # For deserializing frames
import logging

# Initialize Flask app and logging
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Load model
model_path = '../models/classify/T3D/t3d-121_0-mp_24-epochs_30frs.pth'
model = T3D(
    121,
    phi=0.5,
    growth_rate=12,
    temporal_expansion=1,
    transition_t1_size=[1, 3, 6],
    transition_t_size=[1, 3, 4],
    n_input_channels=3,
    conv1_t_size=3,
    conv1_t_stride=1,
    no_max_pool=True,
    n_classes=27,
    dropout=0.0
).to(device)

# Load model weights and move to the appropriate device
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()
logging.info("Model loaded successfully and moved to the device.")

@app.route('/infer', methods=['POST'])
def infer():
    try:
        # Deserialize the received data
        frames_data = pickle.loads(request.data)

        # Ensure the correct shape (batch_size, channels, num_frames, height, width)
        frames_tensor = torch.tensor(frames_data).float()  # Should have shape (1, 3, num_frames, height, width)
        frames_tensor = frames_tensor.to(device)  # Move to the device

        # Perform inference
        with torch.no_grad():
            outputs = model(frames_tensor)
            _, pred = outputs.max(1)
            predicted_label = str(pred.item())

        return jsonify({"predicted_label": predicted_label})

    except Exception as e:
        logging.error(f"Error during inference: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
