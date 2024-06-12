"""
    Evaluate the model with video and real-time camera
"""

import os
import cv2

import torch
from torchvision import transforms
from network.R3D import R3D
from network.R2Plus1D import R2Plus1D

# Convert images to video
def images_to_video(image_folder, fps = 15):
    # Get video name
    video_name = image_folder.split('\\')[-1] + '.mp4'
    video_name = os.path.join('demo\demo_input', video_name)
    
    # Load images
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort()
    
    # Get image size
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    
    # Create video
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
        
    cv2.destroyAllWindows()
    video.release()
    
# Load model
def load_model(
    device, model_path,
    model_arch = 'r3d', block_arch = 18,
    no_max_pool = True, n_classes = 28,
):
    # Define model
    if model_arch == 'r3d':
        model = R3D(
            block_arch,
            n_input_channels = 3,
            conv1_t_size = 7,
            conv1_t_stride = 1,
            no_max_pool = no_max_pool,
            widen_factor = 1,
            n_classes = n_classes
        ).to(device)
    elif model_arch == 'r2plus1d':
        model = R2Plus1D(
            block_arch,
            n_input_channels = 3,
            conv1_t_size = 7,
            conv1_t_stride = 1,
            no_max_pool = no_max_pool,
            widen_factor = 1,
            n_classes = n_classes
        ).to(device)
    else:
        raise ValueError('Invalid model architecture!')
    
    # Load model
    model.load_state_dict(torch.load(model_path, map_location = device))
    
    return model
        
# Predict using existed video
def pred_video(
    video_id,
    video_path, 
    params,
    model = '../models/classify/R3D/r3d-18_0-mp_10-epochs.pth',
    drop_frame = 0, # Drop n frames between 2 frames
):
    # Get parameters
    resize = params['resize'] # (112, 112)
    num_frames = params['num_frames'] # 30
    model_arch = params['model_arch'] # R3D, R(2+1)D
    block_arch = params['block_arch'] # 18
    no_max_pool = params['no_max_pool'] # True
    n_classes = params['n_classes'] # 28
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(resize),
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
    model = load_model(
        device, model,
        model_arch = model_arch,
        block_arch = block_arch,
        no_max_pool = no_max_pool,
        n_classes = n_classes
    )
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set the desired window size (adjust as needed)
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', 800, 600)  # Adjust width and height accordingly
    
    # Define the codec and create VideoWriter object to save the output video
    output_name = str(video_id) + '.mp4'
    out = cv2.VideoWriter(os.path.join('demo\demo_result', output_name), cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_width, frame_height))
    
    count = 0
    frames = []
    while True:
        # Update count
        count += 1
        # Drop frames
        if count % (drop_frame + 1) != 0:
            continue
        
        # Read frame
        _, frame = cap.read()
        if frame is None:
            break
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Apply transformations
        frame_tensor = transform(frame_rgb)
        frames.append(frame_tensor)
        
        # If enough frames are collected, make a prediction
        if len(frames) == num_frames:
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
                # Get the corresponding label name
                predicted_label = labels[pred.item()]
                
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
        
        # Save the frame to the output video
        out.write(frame)
        
        # Press Q on keyboard to stop recording
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
    # Release the video capture object
    cap.release()

    # Release the video writer object
    out.release()
    
    # Destroy all windows
    cv2.destroyAllWindows()
    
def main():
    # Define path
    video_id = 148089
    video_folder = '..\datasets\JESTER-V1\images'
    video_path = os.path.join(video_folder, str(video_id))
    video = os.path.join('demo\demo_input', str(video_id) + '.mp4')
    
    # Check if the video exists
    if not os.path.exists(video):
        images_to_video(video_path, fps = 20)
    
    # Define parameters
    params = {
        # Input parameters
        'resize': (112, 112),
        'num_frames': 30,
        # Model parameters
        'model_arch': 'r3d',
        'block_arch': 18,
        'no_max_pool': True,
        'n_classes': 28,
    }
    
    # Predict video
    pred_video(
        video_id,
        video,
        params
    )
    
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