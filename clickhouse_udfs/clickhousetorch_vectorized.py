#!/shiba_env/bin/python3

import sys
import os

# Limit multi-threading layout to prevent core fighting
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
from torchvision import datasets, models
from torchvision.transforms import ToTensor, Compose, Resize
from PIL import Image
import numpy as np

# Set internal torch thread execution behavior
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Flatten(),
            torch.nn.Linear(96 * 11 * 18, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),

            torch.nn.Linear(128, 2),
            torch.nn.ReLU(),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

def load_image_tensor(file):
    """Loads a single image from disk and converts it to a Float32 tensor."""
    file = file.rstrip("\n")
    #print("opening file")
    image = Image.open('/images/' + file).convert('RGB')
    #print("pipeline")
    transform_pipeline = Compose([
        Resize((375, 600)),  # Resizes to Height: 375, Width: 600
        ToTensor()
    ])
            
    return transform_pipeline(image).float()

def main():
    device = 'cpu'
    
    # 1. Initialize and load model ONCE at process startup
    model = CNN().to(device)
    model.load_state_dict(torch.load('/models/mymodel', map_location=torch.device(device)))
    model.eval()

    while True:
        line = sys.stdin.readline()
        if not line:
            break  # Server closed stream or terminated
        
        # 2. Extract block size header integer
        num_rows = int(line.strip())
        
        # 3. Read exact quantity of rows allocated to this block
        filenames = [sys.stdin.readline() for _ in range(num_rows)]
        #filenames = [sys.stdin.readline().rstrip('\r\n') for _ in range(num_rows)]
        
        # 4. Process image tensors in memory
        tensors = [load_image_tensor(f) for f in filenames]
        
        # 5. Stack tensors into a single unified 4D batch tensor: [Batch_Size, Channels, Height, Width]
        batch_tensor = torch.stack(tensors).to(device)
        
        # 6. Execute true vectorized inference pass
        with torch.no_grad():
            prediction = model(batch_tensor)
            _, y_preds = torch.max(prediction.data, 1)
        
        # 7. Write predictions back in the exact order received
        for pred in y_preds:
            sys.stdout.write(f"{float(pred)}\n")
            
        sys.stdout.flush()

if __name__ == '__main__':
    main()

