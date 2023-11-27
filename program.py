import cv2

import torch
from torchvision import transforms
from network.R3D import R3D
from network.R2Plus1D import R2Plus1D

class GestureRecognizer:
    def __init__(
        self,
        model_path,
        model_arch = 'r3d', block_arch = 18,
        resize = (112, 112), num_frames = 30,
        no_max_pool = True, n_classes = 28,
        drop_frame = 0
    ):
        # Model path
        self.model_path = model_path
        # Model parameters
        self.model_arch = model_arch
        self.block_arch = block_arch
        self.resize = resize
        self.num_frames = num_frames
        self.no_max_pool = no_max_pool
        self.n_classes = n_classes
        # Drop n frames between 2 frames
        self.drop_frame = drop_frame
        
    def load_model(self, device):
        if self.model_arch == 'r3d':
            model = R3D(
                self.block_arch,
                n_input_channels = 3,
                conv1_t_size = 7,
                conv1_t_stride = 1,
                no_max_pool = self.no_max_pool,
                widen_factor = 1,
                n_classes = self.n_classes
            ).to(device)
        elif self.model_arch == 'r2plus1d':
            model = R2Plus1D(
                self.block_arch,
                n_input_channels = 3,
                conv1_t_size = 7,
                conv1_t_stride = 1,
                no_max_pool = self.no_max_pool,
                widen_factor = 1,
                n_classes = self.n_classes
            ).to(device)
        else:
            raise ValueError('Model architecture not supported')
        
        # Load model
        model.load_state_dict(torch.load(self.model_path, map_location = device))
        
        return model
    
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
        
        # Choose device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        # Load model
        model = self.load_model(device)
        
        # Use the default camera as the video source
        cap = cv2.VideoCapture(0)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        count = 0
        frames = []
        while True:
            # Update count
            count += 1
            # Drop frames
            if count % (self.drop_frame + 1) != 0:
                continue
            
            # Read frame
            _, frame = cap.read()
            if frame is None:
                break
            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Apply transformations
            frame_tensor = transform(frame_rgb)
            frames.append(frame_tensor)
            
            # If enough frames are collected, make a prediction
            if len(frames) == self.num_frames:
                # Stack frames along the time dimension
                input_frames = torch.stack(frames, dim = 0) # (T x C x H x W)
                input_frames = input_frames.permute(1, 0, 2, 3) # (C x T x H x W)
                input_frames = input_frames.unsqueeze(0) # Add batch dimension
                input_frames = input_frames.to(device)
                
                with torch.no_grad():
                    # Set model to evaluation mode
                    model.eval()
                    # Forward pass
                    output = model(input_frames)
                    # Get prediction
                    _, pred = output.max(1)
                    # Get the correspoding label name
                    predicted_label = labels[pred.item()]
                       
                print(pred, predicted_label)
                
                # Display the result on the frame at the bottom with red color
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottom_left_corner_of_text = (5, frame_height - 10)  # Adjust the Y-coordinate for the bottom
                font_scale = 0.25
                font_color = (0, 0, 255)  # Red color in BGR format
                font_thickness = 1
                cv2.putText(
                    frame,
                    f'{predicted_label} ({pred.item()})',
                    bottom_left_corner_of_text, font, font_scale, font_color, font_thickness,
                    cv2.LINE_AA
                )
                
                # Remove the first frame
                frames = frames[1:]
                
            # Display the resulting frame
            cv2.imshow('Frame', frame)
            
            # Press Q on keyboard to stop recording
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
        # Release the video capture object
        cap.release()
        # Close all the frames
        cv2.destroyAllWindows()

def main():
    program = GestureRecognizer(
        model_path = '../models/classify/R3D/r3d-18_0-mp_9-epochs.pth',
        model_arch = 'r3d', block_arch = 18,
        drop_frame = 0
    )
    
    program.run()

if __name__ == '__main__':
    labels = [
        'Swiping Left',
        'Swiping Right',
        'Swiping Down',
        'Swiping Up',
        'Pushing Hand Away',
        'Pulling Hand In',
        'Sliding Two Fingers Left',
        'Sliding Two Fingers Right',
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
    
    main()       