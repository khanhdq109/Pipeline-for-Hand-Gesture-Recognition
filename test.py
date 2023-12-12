import torch

frames = torch.randn(1, 3, 5, 112, 112)

last_frame_1 = frames[:, :, -1, :, :]
last_frame_2 = frames[:, :, -1, :, :].unsqueeze(2)

print(frames.shape)
print(last_frame_1.shape)
print(last_frame_2.shape)