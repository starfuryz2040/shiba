#!/shiba_env/bin/python3

import sys
import os

#os.environ["OMP_NUM_THREADS"] = "4"
#os.environ["MKL_NUM_THREADS"] = "4"
#os.environ["OPENBLAS_NUM_THREADS"] = "4"
#os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
#os.environ["NUMEXPR_NUM_THREADS"] = "4"

import torch
import torch.nn as nn
from torchvision import datasets, models
from torchvision.transforms import ToTensor, Compose, Resize

#torch.set_num_threads(1)
#torch.set_num_interop_threads(1)

import numpy as np
from PIL import Image
from io import BytesIO
import base64
import json

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            # Input = 3 x 32 x 32, Output = 32 x 32 x 32
            # new input 3x600x375 output=
            torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding='same'),
            torch.nn.ReLU(),
            # Input = 32 x 32 x 32, Output = 32 x 16 x 16
            torch.nn.MaxPool2d(kernel_size=2),

            # Input = 32 x 16 x 16, Output = 64 x 16 x 16
            torch.nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding='same'),
            torch.nn.ReLU(),
            # Input = 64 x 16 x 16, Output = 64 x 8 x 8
            torch.nn.MaxPool2d(kernel_size=2),

            # Input = 64 x 8 x 8, Output = 64 x 8 x 8
            torch.nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding='same'),
            torch.nn.ReLU(),
            # Input = 64 x 8 x 8, Output = 64 x 4 x 4
            torch.nn.MaxPool2d(kernel_size=2),

            # Input = 64 x 8 x 8, Output = 64 x 8 x 8
            torch.nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, padding='same'),
            torch.nn.ReLU(),
            # Input = 64 x 8 x 8, Output = 64 x 4 x 4
            torch.nn.MaxPool2d(kernel_size=2),

            # Input = 64 x 8 x 8, Output = 64 x 8 x 8
            torch.nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, padding='same'),
            torch.nn.ReLU(),
            # Input = 64 x 8 x 8, Output = 64 x 4 x 4
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Flatten(),

            torch.nn.Linear(96 * 11 * 18, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),

            # torch.nn.Linear(128, 128),
            # torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(128),

            torch.nn.Linear(128, 2),
            torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(2),
            # torch.nn.Dropout(0.2),

            torch.nn.Softmax(dim=1)

        )

    def forward(self, x):
        return self.model(x)


def predict_tensor(file):

    file = file.rstrip("\n")
    # print("opening file")
    image = Image.open('/images/' + file).convert('RGB')

    transform_pipeline = Compose([
                Resize((375, 600)),  # Height: 375, Width: 600
                ToTensor()
            ])
    tensor = transform_pipeline(image).float()


    device = 'cpu'
    model = CNN().to(device)
    model.load_state_dict(torch.load('/models/mymodel', map_location=torch.device(device)))
    model.eval()  # Set the model to evaluation mode


    with torch.no_grad():  # Disable gradient calculation for inference
        prediction = model(tensor.unsqueeze(0).to(device))
        _, y_pred = torch.max(prediction.data, 1)

    return float(y_pred)

if __name__ == '__main__':
    for line in sys.stdin:
        print(predict_tensor(line))
        sys.stdout.flush()