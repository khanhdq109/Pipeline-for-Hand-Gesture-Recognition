import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Available GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Running on CPU.")
