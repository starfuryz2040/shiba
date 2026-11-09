import torch
import evadb
from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import setup, forward
from typing import Optional, List
import pandas as pd
import io
from torchvision.io import read_image
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Compose, Resize
import numpy as np
from PIL import Image
from io import BytesIO
import base64

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            #Input = 3 x 32 x 32, Output = 32 x 32 x 32
            #new input 3x600x375 output=
            torch.nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 3, padding = 'same'),
            torch.nn.ReLU(),
            #Input = 32 x 32 x 32, Output = 32 x 16 x 16
            torch.nn.MaxPool2d(kernel_size=2),

            #Input = 32 x 16 x 16, Output = 64 x 16 x 16
            torch.nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size = 3, padding = 'same'),
            torch.nn.ReLU(),
            #Input = 64 x 16 x 16, Output = 64 x 8 x 8
            torch.nn.MaxPool2d(kernel_size=2),

            #Input = 64 x 8 x 8, Output = 64 x 8 x 8
            torch.nn.Conv2d(in_channels = 12, out_channels = 24, kernel_size = 3, padding = 'same'),
            torch.nn.ReLU(),
            #Input = 64 x 8 x 8, Output = 64 x 4 x 4
            torch.nn.MaxPool2d(kernel_size=2),

            #Input = 64 x 8 x 8, Output = 64 x 8 x 8
            torch.nn.Conv2d(in_channels = 24, out_channels = 48, kernel_size = 3, padding = 'same'),
            torch.nn.ReLU(),
            #Input = 64 x 8 x 8, Output = 64 x 4 x 4
            torch.nn.MaxPool2d(kernel_size=2),

            #Input = 64 x 8 x 8, Output = 64 x 8 x 8
            torch.nn.Conv2d(in_channels = 48, out_channels = 96, kernel_size = 3, padding = 'same'),
            torch.nn.ReLU(),
            #Input = 64 x 8 x 8, Output = 64 x 4 x 4
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Flatten(),

            torch.nn.Linear(96*11*18, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),




            #torch.nn.Linear(128, 128),
            #torch.nn.ReLU(),
            #torch.nn.BatchNorm1d(128),



            torch.nn.Linear(128, 2),
            torch.nn.ReLU(),
            #torch.nn.BatchNorm1d(2),
            #torch.nn.Dropout(0.2),


            torch.nn.Softmax(dim=1)

        )

    def forward(self, x):



        return self.model(x)

class predict_with_torch_optimized(AbstractFunction):

    @property
    def name(self) -> str:
        return "custom_torch_model_2"

    @setup(cacheable=True, function_type='object_detection', batchable=True)
    def setup(self):
        # In the setup method, you initialize your model and load pre-trained weights

        self.device = 'cpu'
        self.model = CNN().to(self.device)
        self.model.load_state_dict(torch.load('/models/mymodel', map_location=torch.device(self.device)))
        self.model.eval()  # Set the model to evaluation mode

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["imageData"],
                column_types=[NdArrayType.STR],
                column_shapes=[(None,)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["predictions"],
                column_types=[
                    NdArrayType.STR
                ],
                column_shapes=[(None,)],
            )
        ],
    )
    def forward(self, imagedata):
        #print(type(imagedata))
        #print(imagedata.columns)
        series = imagedata['imagedata']
        base64image_decode = series.apply(lambda x: base64.b64decode(x))
        image_pil = base64image_decode.apply(lambda x: Image.open(BytesIO(x)).convert('RGB'))
        transform_pipeline = Compose([
            Resize((375, 600)),  # Resizes to Height: 375, Width: 600
            ToTensor()
        ])

        tensor_series = image_pil.apply(lambda x: transform_pipeline(x).float())
        input_tensor = tensor_series.tolist()

        with torch.no_grad():  # Disable gradient calculation for inference
            predictions = self.model(torch.stack(input_tensor).to(self.device))
            _, y_pred = torch.max(predictions.data, 1)
            #return y_pred
        #df = pd.Series(y_pred.squeeze().tolist(), name="Predictions")
        out = pd.DataFrame(y_pred.squeeze().tolist(), columns=['predictions'])
        return out

    """
        def forward(self, imagedata):
        #print(type(imagedata))
        print(imagedata.columns)
        repeat = imagedata['imagedata']
        #base64image_decode = imagedata.apply(lambda x: base64.b64decode(x))
        #image_pil = base64image_decode.apply(lambda x: Image.open(BytesIO(x)))
        #tensor_series = image_pil.apply(lambda x: ToTensor()(x).float())
        #input_tensor = tensor_series.tolist()

        #with torch.no_grad():  # Disable gradient calculation for inference
        #    predictions = self.model(torch.stack(input_tensor).to(self.device))
        #    _, y_pred = torch.max(predictions.data, 1)
        # return y_pred
        #df = pd.Series(y_pred.squeeze().tolist(), name="Predictions")
        print(imagedata['imagedata'].nunique())
        return imagedata.rename(columns={"imagedata": "predictions"})"""