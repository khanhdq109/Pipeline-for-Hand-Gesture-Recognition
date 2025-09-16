import time
import cv2
import torch
from torchvision import transforms
from network.T3D import T3D
from collections import Counter, deque
import pyautogui

class GestureRecognizer:
    def __init__(
        self,
        model_path,
        labels,
        model_arch = 't3d', block_arch = 121,
        resize = (112, 112), num_frames = 24,
        no_max_pool = True, n_classes = 27,
        drop_frame = 0, clear_interval = 999,
        demo = False
    ):
        # Initialize params
        self.model_path = model_path
        self.labels = labels
        self.model_arch = model_arch
        self.block_arch = block_arch
        self.resize = resize
        self.num_frames = num_frames
        self.no_max_pool = no_max_pool
        self.n_classes = n_classes
        self.drop_frame = drop_frame
        self.clear_interval = clear_interval
        self.demo = demo

        # Predictions stack
        self.preds = deque(maxlen=20)

        # Select device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def load_model(self):
        model = T3D(
            self.block_arch,
            phi = 0.5,
            growth_rate = 12,
            temporal_expansion = 1,
            transition_t1_size = [1, 3, 6],
            transition_t_size = [1, 3, 4],
            n_input_channels = 3,
            conv1_t_size = 3,
            conv1_t_stride = 1,
            no_max_pool = self.no_max_pool,
            n_classes = self.n_classes,
            dropout = 0.0
        ).to(self.device)

        try:
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print("Model loaded successfully!")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        except RuntimeError as e:
            raise ValueError("Ensure the model architecture matches the weights.") from e

        return model

    def choose_label(self, min_count = 4):
        counts = Counter(self.preds)
        for element, count in counts.items():
            if count > min_count and element not in {25, 26}:
                return element
        return -1

    def perform_action(self, label):
        """Perform computer actions based on the detected gesture label."""
        if label in {"Sliding Two Fingers Down", "Sliding Two Fingers Left", "Swiping Down", "Swiping Left"}:
            for distance in range(500, 0, -100):  # Start fast, then slow
                pyautogui.scroll(distance)  # Positive value scrolls up
                time.sleep(0.0001)  # Small delay between steps
        elif label in {"Sliding Two Fingers Up", "Sliding Two Fingers Right", "Swiping Up", "Swiping Right"}:
            for distance in range(-500, 0, 100):  # Start fast, then slow
                pyautogui.scroll(distance)  # Negative value scrolls down
                time.sleep(0.0001)  # Small delay between stepss
        elif label in {"Zooming In With Full Hand", "Zooming In With Two Fingers"}:
            pyautogui.hotkey("ctrl", "+")  # Zoom in
        elif label in {"Zooming Out With Full Hand", "Zooming Out With Two Fingers"}:
            pyautogui.hotkey("ctrl", "-")  # Zoom out

    def run(self):
        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.resize),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
            )
        ])

        # Load model
        model = self.load_model()
        model.eval()

        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Failed to open the default camera.")

        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = deque(maxlen = self.num_frames)  # Use a deque(circular buffer) to store frames
        frame_count = 0
        rest = 0

        print("Starting gesture recognition. Press 'q' to quit.")
        while True:
            if rest > 0:
                self.preds.clear()
                frames.clear()
                rest -= 1
                continue
            frame_count += 1
            if frame_count % self.clear_interval == 0:
                frames.clear()

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
            frames.append(frame_tensor)

            # If enough frames are collected, make a prediction
            if len(frames) == self.num_frames:
                # Convert deque to list of tensors and stack along the time dimension
                input_frames = torch.stack(list(frames), dim=0).permute(1, 0, 2, 3).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = model(input_frames)
                    _, pred = output.max(1)
                    predicted_label = self.labels[pred.item()]

                # Log prediction
                self.preds.append(pred.item())
                true_pred = self.choose_label(min_count=10)
                if true_pred != -1 and pred.item() not in {25, 26}:
                    print(f"Prediction: {predicted_label} (Class {pred.item()})")
                    self.perform_action(predicted_label)  # Perform action
                    rest = 1
                    time.sleep(1.0)

            # Display the resulting frame
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
        'Swiping Right',
        'Swiping Left',
        'Swiping Down',
        'Swiping Up',
        'Pulling Hand In',
        'Pushing Hand Away',
        'Sliding Two Fingers Right',
        'Sliding Two Fingers Left',
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

    program = GestureRecognizer(
        model_path = '../models/classify/T3D/t3d-121_0-mp_25-epochs_24frs.pth',
        labels = labels,
        num_frames = 30,
        model_arch = 't3d', block_arch = 121,
        drop_frame = 0,
        clear_interval = 500,
        n_classes = 27,
        demo = False
    )

    program.run()


if __name__ == '__main__':
    main()
