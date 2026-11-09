#!/shiba_env/bin/python3

import sys
import os

import torch
import torch.nn as nn
from torchvision import datasets, models
from torchvision.transforms import ToTensor

import numpy as np
from PIL import Image
from io import BytesIO
import base64
import json

with open('/runlogs/test.log', 'a', encoding='utf-8') as file:
    file.write('in udf.\n')


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
    with open('/runlogs/test.log', 'a', encoding='utf-8') as log_file:
        log_file.write('load image.\n')

    file = file.rstrip("\n")
    try:
        image = Image.open('/images/' + file + '.PNG')
        with open('/runlogs/test.log', 'a', encoding='utf-8') as log_file:
            log_file.write('load image complete PNG.\n')
        # print(f'{file}.png found')
    except:
        try:
            image = Image.open('/images/' + file + '.JPEG')
            with open('/runlogs/test.log', 'a', encoding='utf-8') as log_file:
                log_file.write('load image complete. JPEG\n')
            # print(f'{file}.jpg found')
        except Exception as e:
            with open('/runlogs/test.log', 'a', encoding='utf-8') as log_file:
                log_file.write(str(e))
                return e


    # img_data = base64.b64decode(base64image)
    # image = Image.open(BytesIO(img_data))
    tensor = ToTensor()(image).float()
    with open('/runlogs/test.log', 'a', encoding='utf-8') as log_file:
        log_file.write('tensor loaded.\n')

    device = 'cpu'
    model = CNN().to(device)
    model.load_state_dict(torch.load('/models/mymodel', map_location=torch.device(device)))
    model.eval()  # Set the model to evaluation mode
    with open('/runlogs/test.log', 'a', encoding='utf-8') as log_file:
        log_file.write('model loaded\n')

    with torch.no_grad():  # Disable gradient calculation for inference
        prediction = model(tensor.unsqueeze(0).to(device))
        _, y_pred = torch.max(prediction.data, 1)

    with open('/runlogs/test.log', 'a', encoding='utf-8') as log_file:
        log_file.write('prediction done.\n')

    with open('/runlogs/test.log', 'a', encoding='utf-8') as log_file:
        log_file.write('y_pred.\n')
    return float(y_pred)

if __name__ == '__main__':
    for line in sys.stdin:
        print(predict_tensor(line))
        sys.stdout.flush()