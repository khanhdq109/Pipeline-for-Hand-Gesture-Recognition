import os
from skimage import io

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import warnings
warnings.filterwarnings('ignore', message = 'The default value of the antialias parameter of all the resizing transforms*')

class JesterV1(Dataset):
    def __init__(self, data_dir, num_frames = 30, transform = None, mode = 'train'):
        """
            data_dir: directory containing the data
            labels_file: file containing the labels
            num_frames: number of frames to consider for each video
            transform: transformations to be applied on the data
        """
        if mode == 'train':
            annotations_file = os.path.join(data_dir, 'annotations\jester-v1-train.txt')
        elif mode == 'val':
            annotations_file = os.path.join(data_dir, 'annotations\jester-v1-validation.txt')
        else:
            raise ValueError('Invalid mode')
        
        self.data_dir = data_dir
        self.annotations = self.load_annotations(annotations_file)
        self.num_frames = num_frames
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        video_path = os.path.join(self.data_dir, 'images')
        video_path = os.path.join(video_path, str(self.annotations[idx][0]))
        # Load frames and label
        frames = self.load_frames(video_path)
        label = self.annotations[idx][1]
        
        # print(type(frames[0]))
        
        # Apply transformations
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        # Stack frames into tensor (C x T x H x W)
        frames = torch.stack(frames, dim = 0) # (T x C x H x W)
        frames = frames.permute(1, 0, 2, 3) # (C x T x H x W)
        
        return frames, label
    
    def load_annotations(self, annotations_file):
        """
            Load file names and labels
        """
        annotations = []
        with open(annotations_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    tmp = line.split()
                    annotations.append((int(tmp[0]), int(tmp[1])))
        
        return annotations
    
    def load_frames(self, video_path):
        """
            Load frames from folder
        """
        frame_paths = os.listdir(video_path)
        frame_paths.sort()
        total_nums_frames = len(frame_paths)
        start = total_nums_frames // 2 - self.num_frames // 2
        end = start + self.num_frames
        
        frames = []
        for i in range(start, end):
            frame_path = os.path.join(video_path, frame_paths[i])
            frame = io.imread(frame_path)
            frames.append(frame)
       
        return frames
    
def main():
    # Define dataset
    data_dir = 'D:\Khanh\Others\Hand_Gesture\datasets\JESTER-V1'
    batch_size = 8
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((96, 96)),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])
    
    # Create an instance of the dataset
    dataset = JesterV1(
        data_dir = data_dir,
        num_frames = 30,
        transform = transform,
        mode = 'train'
    )
    
    # Create a DataLoader
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
    
    # Length of DataLoader
    print("Number of batches:", len(data_loader))
    # Inspect a batch
    for frames, labels in data_loader:
        print("Shape of frames tensor:", frames.shape)
        print("Shape of labels tensor:", labels.shape)
        break
    # Length of Dataset
    print("Number of samples:", dataset.__len__())

if __name__ == '__main__':
    main()
    