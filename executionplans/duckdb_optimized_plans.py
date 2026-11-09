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

import duckdb

import os

from torchgen.api.cpp import return_type
from torchvision.io import read_image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, models
from torchvision.transforms import ToTensor
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



def v_predict_tensor(base64image):
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

    # base64image_decode = base64image.apply(lambda x: base64.b64decode(x))
    # image_pil = base64image_decode.apply(lambda x: Image.open(BytesIO(x)))
    # tensor_series = image_pil.apply(lambda x: ToTensor()(x).float())

    input_data = pd.Series(base64image)

    tensor_series = input_data.apply(lambda x: ToTensor()(Image.open(BytesIO(base64.b64decode(x)))).float())
    input_tensor = tensor_series.tolist()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNN().to(device)
    model.load_state_dict(torch.load('/models/mymodel', map_location=torch.device(device)))
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation for inference
        predictions = model(torch.stack(input_tensor).to(device))
        _, y_pred = torch.max(predictions.data, 1)
    # return y_pred
    return pa.array(y_pred.squeeze().tolist())


def predict_xgboost_vector(gender, ethnicity, driver_class, eye_color, hair_color, is_donor, is_veteran, height_inches,
                           weight_lbs):
    xgboostmodel = xgb.XGBClassifier(enable_categorical=True)
    xgboostmodel.load_model('/models/xgboost_model.model')

    gender_ps = pd.Series(gender, dtype="category")
    ethnicity_ps = pd.Series(ethnicity, dtype="category")
    class_ps = pd.Series(driver_class, dtype="category")
    eye_color_ps = pd.Series(eye_color, dtype="category")
    hair_color_ps = pd.Series(hair_color, dtype="category")
    is_donor_ps = pd.Series(is_donor, dtype="category")
    is_veteran_ps = pd.Series(is_veteran, dtype="category")
    height_inches_ps = pd.Series(height_inches, dtype="int64")
    weight_lbs_ps = pd.Series(weight_lbs).astype("int64")

    df = pd.DataFrame({
        'gender': gender_ps,
        'ethnicity': ethnicity_ps,
        'class': class_ps,
        'eye_color': eye_color_ps,
        'hair_color': hair_color_ps,
        'is_donor': is_donor_ps,
        'is_veteran': is_veteran_ps,
        'height_inches': height_inches_ps,
        'weight_lbs': weight_lbs_ps
    })

    y_pred = xgboostmodel.predict(df)

    return pa.array(y_pred)


class DuckDB_Optimized(DMBSParent):
    def __init__(self, target_table, run_num):
        super().__init__()
        self.target_table = target_table
        self.run_num = run_num
        self.system = "duckdb"
        self.udf_type = "optimized"
        self.join_df = "toll_audit"


    def query_1(self):
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:
            qcon.create_function(name='predict_with_torch_optimized',
                                 function=v_predict_tensor,
                                 parameters=['STRING'],
                                 return_type='FLOAT',
                                 type='arrow')
            result = qcon.sql(f"""
            SET enable_profiling = 'html';
            SET profiling_output = '/querystatslogs/duckdb_optimized/query1_{self.target_table}.html';
            SET profiling_mode = 'detailed';
            
            select predict_with_torch_optimized(imagedata) from {self.target_table};
            
            RESET profiling_output;
            """)
            print(result)


    def query_2(self):
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:
            qcon.create_function(name='predict_with_torch_optimized',
                                 function=v_predict_tensor,
                                 parameters=['STRING'],
                                 return_type='FLOAT',
                                 type='arrow')
            result = qcon.sql(f"""
            SET enable_profiling = 'html';
            SET profiling_output = '/querystatslogs/duckdb_optimized/query2_{self.target_table}.html';
            SET profiling_mode = 'detailed';
            
            with cte as (select gender, predict_with_torch_optimized(imagedata) as my_predictions from {self.target_table}) select gender, my_predictions, count(my_predictions) as count_predictions from cte group by gender, my_predictions;
            
            RESET profiling_output;
            """)
            print(result)

    def query_3(self):
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:
            qcon.create_function(name='predict_with_xgboost_optimized',
                                 function=predict_xgboost_vector,
                                 parameters=['VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'INTEGER', 'VARCHAR'],
                                 return_type='FLOAT',
                                 type='arrow' )
            result = qcon.sql(
                f"""
            SET enable_profiling = 'html';
            SET profiling_output = '/querystatslogs/duckdb_optimized/query3_{self.target_table}.html';
            SET profiling_mode = 'detailed';
            
            select predict_with_xgboost_optimized(gender, ethnicity, class, eye_color, hair_color, is_donor, is_veteran, height_inches, weight_lbs) from {self.target_table};
            
            RESET profiling_output;
            """)
            print(result)


    def query_4(self):
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:
            qcon.create_function(name='predict_with_xgboost_optimized',
                                 function=predict_xgboost_vector,
                                 parameters=['VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR',
                                             'VARCHAR', 'INTEGER', 'VARCHAR'],
                                 return_type='FLOAT',
                                 type='arrow')
            result = qcon.sql(
                f"""
            SET enable_profiling = 'html';
            SET profiling_output = '/querystatslogs/duckdb_optimized/query4_{self.target_table}.html';
            SET profiling_mode = 'detailed';
            
            with cte as (select gender, predict_with_xgboost_optimized(gender, ethnicity, class, eye_color, hair_color, is_donor, is_veteran, height_inches, weight_lbs) as my_predictions from {self.target_table}) select gender, my_predictions, count(my_predictions) as count_predictions from cte group by gender, my_predictions;
            
            RESET profiling_output;
            """)
            print(result)

    def query_5(self):
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:
            qcon.create_function(name='predict_with_torch_optimized',
                                 function=v_predict_tensor,
                                 parameters=['STRING'],
                                 return_type='FLOAT',
                                 type='arrow')
            result = qcon.sql(
                f"""
            SET enable_profiling = 'html';
            SET profiling_output = '/querystatslogs/duckdb_optimized/query5_{self.target_table}.html';
            SET profiling_mode = 'detailed';
            
            with cte as (select gender, weight_lbs, predict_with_torch_optimized(imagedata) as my_predictions
                                from {self.target_table}) select gender, my_predictions, count(my_predictions) as count_predictions from cte where cast(weight_lbs AS INTEGER) > 150 group by gender, my_predictions;
            
            RESET profiling_output;
            """)
            print(result)

    def query_6(self):
        """

        query 6: Join to synthetic toll audit data, return all columns, find max of both CNN and XGBOOST with features intermixed
        between both tables in projection.

        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:
            qcon.create_function(
            name='predict_with_torch_optimized',
            function=v_predict_tensor,
            parameters=['STRING'],
            return_type='FLOAT',
                type='arrow')

            qcon.create_function(
                name='predict_with_xgboost_optimized',
                function=predict_xgboost_vector,
                parameters=['VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'INTEGER',
                            'VARCHAR'],
                return_type='FLOAT',
                type='arrow'
            )
            result = qcon.sql(f"""
            SET enable_profiling = 'html';
            SET profiling_output = '/querystatslogs/duckdb_optimized/query6_{self.target_table}.html';
            SET profiling_mode = 'detailed';
            
        select lst.*, jd.*, greatest(
                predict_with_torch_optimized(lst.imageData)
            ,predict_with_xgboost_optimized(lst.gender, lst.ethnicity, lst.class, jd.driver_eye_color, jd.driver_hair_color, lst.is_donor, lst.is_veteran, lst.height_inches, jd.driver_weight)  
                            ) from {self.target_table} as lst 
        inner join {self.join_df} as jd on lst.license_number = jd.license_number;
        
        RESET profiling_output; 
        """)
            print(result)


    def query_7(self):
        """
        query 7: Join to synthetic toll audit data, return all columns, CNN predict in projection

        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:
            qcon.create_function(
            name='predict_with_torch_optimized',
            function=v_predict_tensor,
            parameters=['STRING'],
            return_type='FLOAT',
                type='arrow'
            )

            result = qcon.sql(f"""
            SET enable_profiling = 'html';
            SET profiling_output = '/querystatslogs/duckdb_optimized/query7_{self.target_table}.html';
            SET profiling_mode = 'detailed';
            
        select lst.*, jd.*, predict_with_torch_optimized(lst.imageData) as my_predictions
            from {self.target_table} as lst 
        inner join {self.join_df} as jd on lst.license_number = jd.license_number;
        
        RESET profiling_output; 
        """)
            print(result)


    def query_8(self):
        """

        query 8: Join to synthetic toll audit data, return all columns, XGBOOST predict in projection

        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:

            qcon.create_function(
                name='predict_with_xgboost_optimized',
                function=predict_xgboost_vector,
                parameters=['VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'INTEGER',
                            'VARCHAR'],
                return_type='FLOAT',
                type='arrow'
            )
            result = qcon.sql(f"""
            SET enable_profiling = 'html';
            SET profiling_output = '/querystatslogs/duckdb_optimized/query8_{self.target_table}.html';
            SET profiling_mode = 'detailed';
            
        select lst.*, jd.*, predict_with_xgboost_optimized(lst.gender, lst.ethnicity, lst.class, jd.driver_eye_color, jd.driver_hair_color, lst.is_donor, lst.is_veteran, lst.height_inches, jd.driver_weight) as my_predictions
            from {self.target_table} as lst 
        inner join {self.join_df} as jd on lst.license_number = jd.license_number;
        
        RESET profiling_output; 
        """)
            print(result)




    def query_9(self):
        """

        query 9: Join to synthetic toll audit data, return all columns, CNN predict in predicate

        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:
            qcon.create_function(
            name='predict_with_torch_optimized',
            function=v_predict_tensor,
            parameters=['STRING'],
            return_type='FLOAT',
                type='arrow'
            )
            result = qcon.sql(f"""
            SET enable_profiling = 'html';
            SET profiling_output = '/querystatslogs/duckdb_optimized/query9_{self.target_table}.html';
            SET profiling_mode = 'detailed';
            
        select lst.*, jd.* 
            from {self.target_table} as lst 
        inner join {self.join_df} as jd on lst.license_number = jd.license_number 
            where cast(predict_with_torch_optimized(lst.imageData) as integer)==0;
            
            RESET profiling_output;
        """)
            print(result)




    def query_10(self):
        """

        query 10: Join to synthetic toll audit data, return all columns, XGBOOST predict in predicate

        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:

            qcon.create_function(
                name='predict_with_xgboost_optimized',
                function=predict_xgboost_vector,
                parameters=['VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'INTEGER',
                            'VARCHAR'],
                return_type='FLOAT',
                type='arrow'
            )
            result = qcon.sql(f"""
            SET enable_profiling = 'html';
            SET profiling_output = '/querystatslogs/duckdb_optimized/query10_{self.target_table}.html';
            SET profiling_mode = 'detailed';
            
        select lst.*, jd.*
            from {self.target_table} as lst 
        inner join {self.join_df} as jd on lst.license_number = jd.license_number 
            where cast(predict_with_xgboost_optimized(lst.gender, lst.ethnicity, lst.class, jd.driver_eye_color, jd.driver_hair_color, lst.is_donor, lst.is_veteran, lst.height_inches, jd.driver_weight) as integer)==0;
            
            RESET profiling_output;
        """)
            print(result)

    def query_11(self):
        """

        query 11: Join to synthetic toll audit data, CTE and aggregate using structured data, CNN in HAVING clause as predicate

        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:
            qcon.create_function(
            name='predict_with_torch_optimized',
            function=v_predict_tensor,
            parameters=['STRING'],
            return_type='FLOAT',
                type='arrow'
            )
            result = qcon.sql(f"""
            SET enable_profiling = 'html';
            SET profiling_output = '/querystatslogs/duckdb_optimized/query11_{self.target_table}.html';
            SET profiling_mode = 'detailed';
            
            with cte as (select lst.imageData, lst.gender, lst.eye_color, jd.driver_hair_color from {self.target_table} as lst
        inner join {self.join_df} as jd on lst.license_number = jd.license_number)
                     select imageData, gender, eye_color, count(driver_hair_color) as count_hair from cte group by imageData, gender, eye_color
                       having cast(predict_with_torch_optimized(imageData) as integer)==0;

            RESET profiling_output;
            """)
            print(result)



    def query_12(self):
        """
        query 12: Join to synthetic toll audit data, CTE and aggregate using structured data, XGBOOST in HAVING clause as predicate

        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:

            qcon.create_function(
                name='predict_with_xgboost_optimized',
                function=predict_xgboost_vector,
                parameters=['VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'INTEGER',
                            'VARCHAR'],
                return_type='FLOAT',
                type='arrow'
            )
            result = qcon.sql(f"""
            SET enable_profiling = 'html';
            SET profiling_output = '/querystatslogs/duckdb_optimized/query12_{self.target_table}.html';
            SET profiling_mode = 'detailed';
            
            with cte as (select lst.class, lst.gender, lst.ethnicity, jd.driver_eye_color, jd.driver_hair_color, lst.is_donor, lst.is_veteran, lst.height_inches, jd.driver_weight
                                                from {self.target_table} as lst
                                       inner join {self.join_df} as jd
                                   on lst.license_number = jd.license_number)
            select *, count(class) as count_class
            from cte
            group by gender, ethnicity, class, driver_eye_color, driver_hair_color, is_donor, is_veteran, height_inches, driver_weight
            having cast(predict_with_xgboost_optimized(gender, ethnicity, class, driver_eye_color, driver_hair_color, is_donor, is_veteran, height_inches, driver_weight) as integer)==0;
            
            RESET profiling_output;""")

            print(result)



    def query_13(self):
        """

        query 13: Join to synthetic toll audit data, select all, use CNN in order by


        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:
            qcon.create_function(
            name='predict_with_torch_optimized',
            function=v_predict_tensor,
            parameters=['STRING'],
            return_type='FLOAT',
                type='arrow'
            )

            result = qcon.sql(f"""
            SET enable_profiling = 'html';
            SET profiling_output = '/querystatslogs/duckdb_optimized/query13_{self.target_table}.html';
            SET profiling_mode = 'detailed';
            
            
                                select lst.*, jd.*
                                from {self.target_table} as lst 
           inner join {self.join_df} as jd
                                on lst.license_number = jd.license_number
                                    order by predict_with_torch_optimized(lst.imageData) desc;
                                    
                                    RESET profiling_output;
                                """)
            print(result)



    def query_14(self):
        """

        query 14: Join to synthetic toll audit data, select all, use XGBOOST in order by

        """
        with duckdb.connect("../duckdatabase/idnet.db") as qcon:

            qcon.create_function(
                name='predict_with_xgboost_optimized',
                function=predict_xgboost_vector,
                parameters=['VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'INTEGER',
                            'VARCHAR'],
                return_type='FLOAT',
                type='arrow'
            )
            result = qcon.sql(f"""
            SET enable_profiling = 'html';
            SET profiling_output = '/querystatslogs/duckdb_optimized/query14_{self.target_table}.html';
            SET profiling_mode = 'detailed';
            
                                select lst.*,
                                       jd.*
                                from {self.target_table} as lst 
           inner join {self.join_df} as jd
                                on lst.license_number = jd.license_number
                                    order by predict_with_xgboost_optimized(lst.gender, lst.ethnicity, lst.class, jd.driver_eye_color,
                                                           jd.driver_hair_color, lst.is_donor, lst.is_veteran,
                                                           lst.height_inches, jd.driver_weight) desc;
                                                           
                                                           RESET profiling_output;
                                """)
            print(result)




    #%%
    def run(self):
        for i in range(1,15):
            self.log = []

            query = i
            #print(f'{i} monitoring start')
            #self.update_global(True)
            #monitor_thread = threading.Thread(target=self.monitor_system_resources, args=(1,query))
            #monitor_thread.start()

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
            print(f'{i} stop')

            # Stop the monitoring thread
            #self.update_global(False)
            #monitor_thread.join() # Wait for the monitoring thread to finish
            #print(f'{i} monitoring end')

            #with open(f"/runlogs/{self.target_table}_{self.run_num}_{self.system}_{self.udf_type}_{query}.json", "w") as outfile:
            #    json.dump(self.log[:], outfile, indent=4)

