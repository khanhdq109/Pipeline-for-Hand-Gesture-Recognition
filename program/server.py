from flask import Flask, request, jsonify
import torch
from torchvision import transforms
import numpy as np
from network.T3D import T3D
import pickle  # For deserializing frames
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the model
model_path = '../models/classify/T3D/t3d-121_0-mp_24-epochs_30frs.pth'
model = T3D(
    block_arch=121,
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
)

# Load model weights
model.load_state_dict(torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
model.eval()

@app.route('/infer', methods=['POST'])
def infer():
    try:
        # Deserialize the received data
        frames_data = pickle.loads(request.data)
        frames_tensor = torch.tensor(frames_data).permute(0, 3, 1, 2).float()  # (T x C x H x W)
        frames_tensor = frames_tensor.unsqueeze(0)  # Add batch dimension
        
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
    app.run(host='0.0.0.0', port=5000)
