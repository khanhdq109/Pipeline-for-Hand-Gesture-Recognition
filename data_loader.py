import os
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class JesterV1Dataset(Dataset):
    def __init__(self, 
                 root_dir, annotation_file, 
                 num_frames = 16,
                 transform = None
        ):
        """
        Args:
            root_dir (str): Directory with all the data
            annotation_file (str): Annotation file name
            num_frames (int): Number of frames in each video
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.annotation_file = os.path.join(os.path.join(root_dir, 'annotations'), annotation_file)
        self.num_frames = num_frames
        self.transform = transform
        
        # Load annotation file
        with open(self.annotation_file, 'r') as f:
            self.annotations = f.readlines()
            
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Load video frames and label
        frames, label = self.load_video_data(idx)
        
        if self.transform:
            frames = [self.transform(frame) for frame in frames] # Apply the transform to each frame
            
        print(type(frames))
        print(len(frames))
        print(frames[0].shape)

        return frames, label
    
    def load_video_data(self, idx):
        videos = os.path.join(self.root_dir, 'images')
        # Get video path and label
        video_path = os.path.join(videos, self.annotations[idx].split()[0])
        label = int(self.annotations[idx].split()[1])
        
        # Get list of all frames in the video directory
        total_frames = os.listdir(video_path)
        total_frames.sort()
        
        # Load frames from the video directory
        frames = []
        num_total_frames = len(total_frames)
        start = num_total_frames // 2 - self.num_frames // 2
        end = start + self.num_frames
        for frame_num in range(start, end):
            frame = cv2.imread(os.path.join(video_path, total_frames[frame_num]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        return frames, label
    
def main():
    # Define data transforms
    data_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        ), # --> Subtracts the mean values and then divides by std for each color channel
    ])
    
    # Create dataset instance
    train_dataset = JesterV1Dataset(
        root_dir = '../datasets/JESTER-V1',
        annotation_file = '../datasets/JESTER-V1/annotations/jester-v1-train.txt',
        num_frames = 30,
        transform = data_transforms
    )
    
    # Create a DataLoader instance
    batch_size = 8
    data_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = False
    )

if __name__ == '__main__':
    main()
    