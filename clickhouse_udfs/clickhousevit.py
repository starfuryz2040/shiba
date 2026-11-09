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
from torchvision.transforms import ToTensor, Normalize, Resize, Compose

#torch.set_num_threads(1)
#torch.set_num_interop_threads(1)

import numpy as np
from PIL import Image
from io import BytesIO
import base64
import json




def predict_tensor(file):
    file = file.rstrip("\n")
    # print("opening file")
    image = Image.open('/images/' + file).convert('RGB')

    transform_pipeline = Compose([
        Resize((224, 224)),  # Explicitly resize all varying image dimensions
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform_pipeline(image).float()

    # img_data = base64.b64decode(base64image)
    # image = Image.open(BytesIO(img_data))
    #tensor = ToTensor()(image).float()

    device = 'cpu'
    vit_model = torch.jit.load('/models/vit_tiny_compiled.pt', map_location=torch.device(device))
    vit_model.eval()  # Set the model to evaluation mode


    with torch.no_grad():  # Disable gradient calculation for inference
        prediction = vit_model(tensor.unsqueeze(0).to(device))
        _, y_pred = torch.max(prediction.data, 1)

    return float(y_pred)

if __name__ == '__main__':
    for line in sys.stdin:
        print(predict_tensor(line))
        sys.stdout.flush()