import cv2
from torchvision import transforms
import requests
from collections import deque
import numpy as np
import pickle  # For efficient serialization

class GestureRecognizer:
    def __init__(
        self,
        labels,
        resize = (112, 112),
        num_frames = 24,
        drop_frame = 0,
        server_url = "http://188.26.201.29:5000/infer"
    ):
        self.labels = labels
        self.resize = resize
        self.num_frames = num_frames
        self.drop_frame = drop_frame
        self.server_url = server_url
        print("Gesture recognizer initialized.")

    def run(self):
        # Define transformation
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.resize),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Failed to open the default camera.")
        
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = deque(maxlen = self.num_frames)
        frame_count = 0

        print("Starting gesture recognition. Press 'q' to quit.")
        while True:
            frame_count += 1
            
            # Drop frames
            if frame_count % (self.drop_frame + 1) != 0:
                continue
            
            # Read frame
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to capture frame. Exiting...")
                break
            
            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB and apply transformations
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = transform(frame_rgb)
            frames.append(frame_tensor.numpy())
            
            # If enough frames are collected, send to server for prediction
            if len(frames) == self.num_frames:
                try:
                    # Serialize frames using pickle
                    frames_data = pickle.dumps(np.array(frames))
                    
                    # Send frames to server for inference
                    response = requests.post(self.server_url, data = frames_data, headers = {"Content-Type": "application/octet-stream"})
                    if response.status_code == 200:
                        predicted_label = response.json().get("predicted_label")
                        print(f"Prediction: {predicted_label}")
                    else:
                        print(f"Error from server: {response.text}")
                except Exception as e:
                    print(f"Error in communication with server: {e}")
                
                # Display the result
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottom_left_corner = (5, frame_height - 10)
                cv2.putText(
                    frame,
                    f"{predicted_label}",
                    bottom_left_corner,
                    font,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                
            # Display the frame
            cv2.imshow('Frame', frame)
            
            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting gesture recognition...")
                break
            
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
def main():
    labels = [
        'Swiping Left', 'Swiping Right', 'Swiping Down', 'Swiping Up',
        'Pushing Hand Away', 'Pulling Hand In', 'Sliding Two Fingers Left',
        'Sliding Two Fingers Right', 'Sliding Two Fingers Down', 'Sliding Two Fingers Up',
        'Pushing Two Fingers Away', 'Pulling Two Fingers In', 'Rolling Hand Forward',
        'Rolling Hand Backward', 'Turning Hand Clockwise', 'Turning Hand Counterclockwise',
        'Zooming In With Full Hand', 'Zooming Out With Full Hand', 'Zooming In With Two Fingers',
        'Zooming Out With Two Fingers', 'Thumb Up', 'Thumb Down', 'Shaking Hand',
        'Stop Sign', 'Drumming Fingers', 'No gesture', 'Doing other things'
    ]
    
    program = GestureRecognizer(
        labels = labels,
        num_frames = 30,
        drop_frame = 0
    )
    
    program.run()

if __name__ == '__main__':
    main()