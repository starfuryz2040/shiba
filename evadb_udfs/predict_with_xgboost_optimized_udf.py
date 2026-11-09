import torch
import evadb
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

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
from torchvision.transforms import ToTensor
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import xgboost as xgb


class predict_with_xgboost_optimized_2(AbstractFunction):

    @property
    def name(self) -> str:
        return "predict_with_xgboost_optimized_2"

    @setup(cacheable=True, function_type='object_detection', batchable=True)
    def setup(self):
        # In the setup method, you initialize your model and load pre-trained weights

        self.xgboostmodel = xgb.XGBClassifier(enable_categorical=True)
        self.xgboostmodel.load_model('/models/xgboost_model.model')

        xgb.config_context(nthread=4)


    @forward(
        input_signatures=[],
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
    def forward(self, frame):
        #print(frame)
        #print(type(frame))
        """
        data = [
            {
                'gender': gender,
                'ethnicity': ethnicity,
                'class': driver_class,
                'eye_color': eye_color,
                'hair_color': hair_color,
                'is_donor': is_donor,
                'is_veteran': is_veteran,
                'height_inches': height_inches,
                'weight_lbs': weight_lbs
            }
        ]
        df = pd.DataFrame(data)
        """
        frame = frame.rename(
            columns={'driver_hair_color': 'hair_color', 'driver_eye_color': 'eye_color', 'driver_weight': 'weight_lbs'})
        df = frame[["gender", "ethnicity", "class", "eye_color", "hair_color", "is_donor", "is_veteran", "height_inches", "weight_lbs"]]

        df['gender'] = df['gender'].astype('category')
        df['ethnicity'] = df['ethnicity'].astype('category')
        df['class'] = df['class'].astype('category')
        df['eye_color'] = df['eye_color'].astype('category')
        df['hair_color'] = df['hair_color'].astype('category')
        df['is_donor'] = df['is_donor'].astype('category')
        df['is_veteran'] = df['is_veteran'].astype('category')
        df['height_inches'] = df['height_inches'].astype('int64')
        df['weight_lbs'] = df['weight_lbs'].astype('int64')

        # feature_np = np.array(feature_list, dtype=object)

        y_pred = self.xgboostmodel.predict(df)

        out = pd.DataFrame(y_pred.tolist(), columns=['predictions'])
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