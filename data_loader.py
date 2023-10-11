import os
import cv2

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
            frames = self.transform(frames)

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
    