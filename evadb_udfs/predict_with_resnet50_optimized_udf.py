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
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
import numpy as np
from PIL import Image
from io import BytesIO
import base64



class predict_with_resnet50_optimized(AbstractFunction):

    @property
    def name(self) -> str:
        return "predict_with_resnet50_optimized"

    @setup(cacheable=True, function_type='object_detection', batchable=True)
    def setup(self):
        # In the setup method, you initialize your model and load pre-trained weights

        self.device = 'cpu'
        self.resnet50_model = torch.jit.load('/models/resnet50_compiled.pt', map_location=torch.device(self.device))
        self.resnet50_model.eval()  # Set the model to evaluation mode

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
            Resize((224, 224)),  # Explicitly resize all varying image dimensions
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        tensor_series = image_pil.apply(lambda x: transform_pipeline(x).float())
        input_tensor = tensor_series.tolist()

        with torch.no_grad():  # Disable gradient calculation for inference
            predictions = self.resnet50_model(torch.stack(input_tensor).to(self.device))
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