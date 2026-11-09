#%%
#run spark.stop before running this notebook. some of the configs won't take if an active spark sessions is grabbed or running. 
#spark.stop()
#%%
import psutil
import time
import threading
import datetime
import json
import xgboost as xgb
import matplotlib.pyplot as plt
import duckdb
import pandas as pd
import pandas as pd

import pyarrow as pa


import os

from torchgen.api.cpp import return_type
from torchvision.io import read_image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, models
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
from torchvision.utils import make_grid

import numpy as np
from PIL import Image
import pyarrow
from io import BytesIO
import base64

import xgboost as xgb
import pandas as pd

import pyarrow as pa
from duckdb.typing import *

from dbms_benchmark_parent_classes.dbms_parent import DMBSParent

#%%
def predict_tensor(base64image):
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

    img_data = base64.b64decode(base64image)
    image = Image.open(BytesIO(img_data)).convert('RGB')
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


def predict_tensor_resnet50(base64image):
    device = 'cpu'
    resnet50_model = torch.jit.load('/models/resnet50_compiled.pt', map_location=torch.device(device))
    resnet50_model.eval()  # Set the model to evaluation mode

    img_data = base64.b64decode(base64image)
    image = Image.open(BytesIO(img_data)).convert('RGB')

    transform_pipeline = Compose([
        Resize((224, 224)),  # Explicitly resize all varying image dimensions
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor = transform_pipeline(image).float()

    with torch.no_grad():  # Disable gradient calculation for inference
        prediction = resnet50_model(tensor.unsqueeze(0).to(device))
        _, y_pred = torch.max(prediction.data, 1)

    return float(y_pred)


def predict_tensor_vit(base64image):
    device = 'cpu'
    vit_model = torch.jit.load('/models/vit_tiny_compiled.pt', map_location=torch.device(device))
    vit_model.eval()

    img_data = base64.b64decode(base64image)
    image = Image.open(BytesIO(img_data)).convert('RGB')

    transform_pipeline = Compose([
        Resize((224, 224)),  # Explicitly resize all varying image dimensions
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor = transform_pipeline(image).float()

    with torch.no_grad():  # Disable gradient calculation for inference
        prediction = vit_model(tensor.unsqueeze(0).to(device))
        _, y_pred = torch.max(prediction.data, 1)

    return float(y_pred)




def predict_xgboost(gender, ethnicity, driver_class, eye_color, hair_color, is_donor, is_veteran, height_inches,
                    weight_lbs):
    xgboostmodel = xgb.XGBClassifier(enable_categorical=True)
    xgboostmodel.load_model('/models/xgboost_model.model')

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

    y_pred = xgboostmodel.predict(df)
    # print(y_pred[0])

    return float(y_pred[0])





class DuckDB_Naive(DMBSParent):
    def __init__(self, target_table, run_num):
        super().__init__()
        self.target_table = target_table
        self.run_num = run_num
        self.system = "duckdb"
        self.udf_type = "naive"
        self.join_df = "toll_audit_no_dupe"


    def query_1(self):
        """
        query 1: simple CNN inference in projection

        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:
            qcon.create_function(
            name='predict_with_torch_naive',
            function=predict_tensor,
            parameters=['STRING'],
            return_type='FLOAT'
            )
            result = qcon.sql(f'select predict_with_torch_naive(imagedata) from {self.target_table}')
            result.execute()

    def query_2(self):
        """
        query 2: CNN inference in CTE, followed by aggregation

        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:
            qcon.create_function(
            name='predict_with_torch_naive',
            function=predict_tensor,
            parameters=['STRING'],
            return_type='FLOAT'
            )
            result = qcon.sql(f"""with cte as (select gender, predict_with_torch_naive(imagedata) as my_predictions from {self.target_table}) select gender, my_predictions, count(my_predictions) as count_predictions from cte group by gender, my_predictions""")
            result.execute()


    def query_3(self):
        """
        query 3: simple XGBOOST inference in projection

        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:
            qcon.create_function(
            name='predict_with_xgboost_naive',
            function=predict_xgboost,
            parameters=['VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'INTEGER', 'INTEGER'],
            return_type='FLOAT'
                )
            result = qcon.sql(f'select predict_with_xgboost_naive(gender, ethnicity, class, eye_color, hair_color, is_donor, is_veteran, height_inches, weight_lbs) from {self.target_table}')
            result.execute()

    def query_4(self):
        """
        query 4: xgboost inference in CTE, followed by aggregation



        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:
            qcon.create_function(
            name='predict_with_xgboost_naive',
            function=predict_xgboost,
            parameters=['VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'INTEGER', 'INTEGER'],
            return_type='FLOAT'
                )
            result = qcon.sql(f'with cte as (select gender, predict_with_xgboost_naive(gender, ethnicity, class, eye_color, hair_color, is_donor, is_veteran, height_inches, weight_lbs) as my_predictions from {self.target_table}) select gender, my_predictions, count(my_predictions) as count_predictions from cte group by gender, my_predictions')
            result.execute()

    def query_5(self):
        """
        query 5: CNN inference in CTE, followed by aggregation, followed by structured predicate

        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:
            qcon.create_function(
            name='predict_with_torch_naive',
            function=predict_tensor,
            parameters=['STRING'],
            return_type='FLOAT'
            )

            result = qcon.sql(f"""with cte as (select gender, weight_lbs, predict_with_torch_naive(imagedata) as my_predictions from {self.target_table}) select gender, my_predictions, count(my_predictions) as count_predictions from cte where cast(weight_lbs AS INTEGER) > 150 group by gender, my_predictions;""")
            result.execute()

    def query_6(self):
        """

        query 6: Join to synthetic toll audit data, return all columns, find max of both CNN and XGBOOST with features intermixed
        between both tables in projection.

        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:
            qcon.create_function(
            name='predict_with_torch_naive',
            function=predict_tensor,
            parameters=['STRING'],
            return_type='FLOAT'
            )
            qcon.create_function(
                name='predict_with_xgboost_naive',
                function=predict_xgboost,
                parameters=['VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'INTEGER',
                            'VARCHAR'],
                return_type='FLOAT'
            )
            result = qcon.sql(f"""
        select lst.*, jd.*, greatest(
                predict_with_torch_naive(lst.imageData)
            ,predict_with_xgboost_naive(lst.gender, lst.ethnicity, lst.class, jd.driver_eye_color, jd.driver_hair_color, lst.is_donor, lst.is_veteran, lst.height_inches, jd.driver_weight)  
                            ) from {self.target_table} as lst 
        inner join {self.join_df} as jd on lst.license_number = jd.license_number 
        """)
            result.execute()


    def query_7(self):
        """
        query 7: Join to synthetic toll audit data, return all columns, CNN predict in projection

        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:
            qcon.create_function(
            name='predict_with_torch_naive',
            function=predict_tensor,
            parameters=['STRING'],
            return_type='FLOAT'
            )

            result = qcon.sql(f"""
        select lst.*, jd.*, predict_with_torch_naive(lst.imageData) as my_predictions
            from {self.target_table} as lst 
        inner join {self.join_df} as jd on lst.license_number = jd.license_number 
        """)
            result.execute()


    def query_8(self):
        """

        query 8: Join to synthetic toll audit data, return all columns, XGBOOST predict in projection

        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:

            qcon.create_function(
                name='predict_with_xgboost_naive',
                function=predict_xgboost,
                parameters=['VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'INTEGER',
                            'VARCHAR'],
                return_type='FLOAT'
            )
            result = qcon.sql(f"""
        select lst.*, jd.*, predict_with_xgboost_naive(lst.gender, lst.ethnicity, lst.class, jd.driver_eye_color, jd.driver_hair_color, lst.is_donor, lst.is_veteran, lst.height_inches, jd.driver_weight) as my_predictions
            from {self.target_table} as lst 
        inner join {self.join_df} as jd on lst.license_number = jd.license_number 
        """)
            result.execute()




    def query_9(self):
        """

        query 9: Join to synthetic toll audit data, return all columns, CNN predict in predicate

        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:
            qcon.create_function(
            name='predict_with_torch_naive',
            function=predict_tensor,
            parameters=['STRING'],
            return_type='FLOAT'
            )
            result = qcon.sql(f"""
        select lst.*, jd.* 
            from {self.target_table} as lst 
        inner join {self.join_df} as jd on lst.license_number = jd.license_number 
            where cast(predict_with_torch_naive(lst.imageData) as integer)==0
        """)
            result.execute()




    def query_10(self):
        """

        query 10: Join to synthetic toll audit data, return all columns, XGBOOST predict in predicate

        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:

            qcon.create_function(
                name='predict_with_xgboost_naive',
                function=predict_xgboost,
                parameters=['VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'INTEGER',
                            'VARCHAR'],
                return_type='FLOAT'
            )
            result = qcon.sql(f"""
        select lst.*, jd.*
            from {self.target_table} as lst 
        inner join {self.join_df} as jd on lst.license_number = jd.license_number 
            where cast(predict_with_xgboost_naive(lst.gender, lst.ethnicity, lst.class, jd.driver_eye_color, jd.driver_hair_color, lst.is_donor, lst.is_veteran, lst.height_inches, jd.driver_weight) as integer)==0
        """)
            result.execute()

    def query_11(self):
        """

        query 11: Join to synthetic toll audit data, CTE and aggregate using structured data, CNN in HAVING clause as predicate

        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:
            qcon.create_function(
            name='predict_with_torch_naive',
            function=predict_tensor,
            parameters=['STRING'],
            return_type='FLOAT'
            )
            result = qcon.sql(f"""with cte as (select lst.imageData, lst.gender, lst.eye_color, jd.driver_hair_color from {self.target_table} as lst
        inner join {self.join_df} as jd on lst.license_number = jd.license_number)
                     select imageData, gender, eye_color, count(driver_hair_color) as count_hair from cte group by imageData, gender, eye_color
                       having cast(predict_with_torch_naive(imageData) as integer)==0 """)
            result.execute()



    def query_12(self):
        """
        query 12: Join to synthetic toll audit data, CTE and aggregate using structured data, XGBOOST in HAVING clause as predicate

        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:

            qcon.create_function(
                name='predict_with_xgboost_naive',
                function=predict_xgboost,
                parameters=['VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'INTEGER',
                            'VARCHAR'],
                return_type='FLOAT'
            )
            result = qcon.sql(f"""with cte as (select lst.class, lst.gender, lst.ethnicity, jd.driver_eye_color, jd.driver_hair_color, lst.is_donor, lst.is_veteran, lst.height_inches, jd.driver_weight
                                                from {self.target_table} as lst
                                       inner join {self.join_df} as jd
                                   on lst.license_number = jd.license_number)
        select *, count(class) as count_class
        from cte
        group by gender, ethnicity, class, driver_eye_color, driver_hair_color, is_donor, is_veteran, height_inches, driver_weight
        having cast(predict_with_xgboost_naive(gender, ethnicity, class, driver_eye_color, driver_hair_color, is_donor, is_veteran, height_inches, driver_weight) as integer)==0 """)
            result.execute()



    def query_13(self):
        """

        query 13: Join to synthetic toll audit data, select all, use CNN in order by


        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:
            qcon.create_function(
            name='predict_with_torch_naive',
            function=predict_tensor,
            parameters=['STRING'],
            return_type='FLOAT'
            )

            result = qcon.sql(f"""
                                select lst.*, jd.*
                                from {self.target_table} as lst 
           inner join {self.join_df} as jd
                                on lst.license_number = jd.license_number
                                    order by predict_with_torch_naive(lst.imageData) desc
                                """)
            result.execute()



    def query_14(self):
        """

        query 14: Join to synthetic toll audit data, select all, use XGBOOST in order by

        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:

            qcon.create_function(
                name='predict_with_xgboost_naive',
                function=predict_xgboost,
                parameters=['VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'INTEGER',
                            'VARCHAR'],
                return_type='FLOAT'
            )
            result = qcon.sql(f"""
                                select lst.*,
                                       jd.*
                                from {self.target_table} as lst 
           inner join {self.join_df} as jd
                                on lst.license_number = jd.license_number
                                    order by predict_with_xgboost_naive(lst.gender, lst.ethnicity, lst.class, jd.driver_eye_color,
                                                           jd.driver_hair_color, lst.is_donor, lst.is_veteran,
                                                           lst.height_inches, jd.driver_weight) desc
                                """)
            result.execute()



    def query_15(self):
        """
        query 1: simple CNN inference in projection

        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:
            qcon.create_function(
                name='predict_with_torch_resnet',
                function=predict_tensor_resnet50,
                parameters=['STRING'],
                return_type='FLOAT'
            )
            result = qcon.sql(f"""SELECT lst.id,
                                      jd.toll_id,
                                      predict_with_torch_resnet(lst.imageData) AS score
                               FROM {self.target_table} AS lst inner join {self.join_df} AS jd
                               ON lst.license_number = jd.license_number
                               ORDER BY score DESC LIMIT 100;""")
            result.execute()


    def query_16(self):
        """
        query 1: simple CNN inference in projection

        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:
            qcon.create_function(
                name='predict_with_torch_vit',
                function=predict_tensor_vit,
                parameters=['STRING'],
                return_type='FLOAT'
            )
            result = qcon.sql(f"""SELECT lst.id,
                                              jd.toll_id,
                                              predict_with_torch_vit(lst.imageData) AS score
                                       FROM {self.target_table} AS lst inner join {self.join_df} AS jd
                                       ON lst.license_number = jd.license_number
                                       ORDER BY score DESC LIMIT 100;""")
            result.execute()

    def query_17(self):
        """
        query 1: simple CNN inference in projection

        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:
            qcon.create_function(
                name='predict_with_xgboost_naive',
                function=predict_xgboost,
                parameters=['VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'INTEGER',
                            'VARCHAR'],
                return_type='FLOAT'
            )

            qcon.create_function(
                name='predict_with_torch_resnet',
                function=predict_tensor_resnet50,
                parameters=['STRING'],
                return_type='FLOAT'
            )
            result = qcon.sql(f"""
                            select lst.*,
                                   jd.*,
                                   greatest(
                                           predict_with_xgboost_naive(lst.gender, lst.ethnicity, lst.class,
                                                               jd.driver_eye_color, jd.driver_hair_color, lst.is_donor,
                                                               lst.is_veteran, lst.height_inches, jd.driver_weight),
                                           predict_with_torch_resnet(lst.imageData)
                                   )
                            from {self.target_table} as lst 
        inner join {self.join_df} as jd
                            on lst.license_number = jd.license_number
                            ;""")
            result.execute()

    def query_18(self):
        """
        query 1: simple CNN inference in projection

        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:
            qcon.create_function(
                name='predict_with_xgboost_naive',
                function=predict_xgboost,
                parameters=['VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'INTEGER',
                            'VARCHAR'],
                return_type='FLOAT'
            )
            qcon.create_function(
                name='predict_with_torch_vit',
                function=predict_tensor_vit,
                parameters=['STRING'],
                return_type='FLOAT'
            )
            result = qcon.sql(f"""
                                select lst.*,
                                       jd.*,
                                       greatest(
                                               predict_with_xgboost_naive(lst.gender, lst.ethnicity, lst.class,
                                                                   jd.driver_eye_color, jd.driver_hair_color, lst.is_donor,
                                                                   lst.is_veteran, lst.height_inches, jd.driver_weight),
                                               predict_with_torch_vit(lst.imageData)
                                       )
                                from {self.target_table} as lst 
            inner join {self.join_df} as jd
                                on lst.license_number = jd.license_number
                                ;""")
            result.execute()

    def query_20(self):
        """
        query 1: simple CNN inference in projection

        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:
            qcon.create_function(
                name='predict_with_torch_vit',
                function=predict_tensor_vit,
                parameters=['STRING'],
                return_type='FLOAT'
            )
            result = qcon.sql(f"""select gender, predict_with_torch_vit(imageData) as my_predictions from {self.target_table} where predict_with_torch_vit(imageData)::integer = 1;""")
            result.execute()


    def query_19(self):
        """
        query 1: simple CNN inference in projection

        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:
            qcon.create_function(
                name='predict_with_torch_resnet',
                function=predict_tensor_resnet50,
                parameters=['STRING'],
                return_type='FLOAT'
            )
            result = qcon.sql(f"""select gender, predict_with_torch_resnet(imageData) as my_predictions from {self.target_table} where predict_with_torch_resnet(imageData)::integer = 1;""")
            result.execute()


    #%%
    def run(self):
        for i in range(20,21):
            self.log = []

            query = i
            print(f'{i} monitoring start')
            self.update_global(True)
            monitor_thread = threading.Thread(target=self.monitor_system_resources, args=(1,query))
            monitor_thread.start()

            # Run the function to be traced
            print(f'{i} start')
            if i == 1:
                self.query_1()
            if i == 2:
                self.query_2()
            if i == 3:
                self.query_3()
            if i == 4:
                self.query_4()
            if i == 5:
                self.query_5()
            if i == 6:
                self.query_6()
            if i == 7:
                self.query_7()
            if i == 8:
                self.query_8()
            if i == 9:
                self.query_9()
            if i == 10:
                self.query_10()
            if i == 11:
                self.query_11()
            if i == 12:
                self.query_12()
            if i == 13:
                self.query_13()
            if i == 14:
                self.query_14()
            if i == 15:
                self.query_15()
            if i == 16:
                self.query_16()
            if i == 17:
                self.query_17()
            if i == 18:
                self.query_18()
            if i == 19:
                self.query_19()
            if i == 20:
                self.query_20()
            print(f'{i} stop')

            # Stop the monitoring thread
            self.update_global(False)
            monitor_thread.join() # Wait for the monitoring thread to finish
            print(f'{i} monitoring end')

            with open(f"/runlogs/{self.target_table}_{self.run_num}_{self.system}_{self.udf_type}_{query}.json", "w") as outfile:
                json.dump(self.log[:], outfile, indent=4)

