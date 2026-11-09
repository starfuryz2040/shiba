import psutil
import time
import threading
import datetime
import json
import xgboost as xgb
import matplotlib.pyplot as plt

import pyspark
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
from torchvision.io import read_image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, models
#from torchvision.transforms import ToTensor
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

from torchvision.utils import make_grid

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pyarrow
from io import BytesIO
import base64



from pyspark.sql.functions import udf, pandas_udf
from pyspark.sql.types import FloatType

import math
from statistics import NormalDist

from dbms_benchmark_parent_classes.dbms_parent import DMBSParent

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



class SparkDB_Optimized(DMBSParent):
    def __init__(self, target_table, run_num):
        super().__init__()
        self.target_table = target_table
        self.run_num = run_num
        self.system = "spark"
        self.udf_type = "optimized"

    def setup(self):

        self.spark = pyspark.sql.SparkSession.builder.master("local[10]").appName("spark_optimized").config(
            "spark.executor.memory", "15g").config("spark.driver.memory", "15g").getOrCreate()

        self.spark.conf.set("spark.sql.execution.pythonUDF.arrow.enabled", True)
        self.spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 5)

        device = 'cpu'
        model = CNN().to(device)
        model.load_state_dict(torch.load('../mymodel', map_location=torch.device(device)))
        model.eval()  # Set the model to evaluation mode

        with open('../resnet50_compiled.pt', 'rb') as f:
            resnet50_bytes = f.read()

        bc_resnet50_bytes = self.spark.sparkContext.broadcast(resnet50_bytes)

        with open('../vit_tiny_compiled.pt', 'rb') as f:
            vit_bytes = f.read()

        bc_vit_bytes = self.spark.sparkContext.broadcast(vit_bytes)

        bc_model = self.spark.sparkContext.broadcast(model)

        @pandas_udf(returnType=FloatType())
        def v_predict_tensor_resnet50(base64image):

            resnet50_model = torch.jit.load(BytesIO(bc_resnet50_bytes.value), map_location=torch.device(device))
            resnet50_model.eval()  # Set the model to evaluation mode

            #print(f"batch size: {len(base64image)}")
            start_time = time.time()
            base64image_decode = base64image.apply(base64.b64decode)
            image_pil = base64image_decode.apply(lambda x: Image.open(BytesIO(x)).convert('RGB'))

            transform_pipeline = Compose([
                Resize((224, 224)),  # Explicitly resize all varying image dimensions
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            tensor_series = image_pil.apply(lambda x: transform_pipeline(x).float())

            # tensor_series = base64image.apply(lambda x: ToTensor()(Image.open(BytesIO(base64.b64decode(x)))).float())
            input_tensor = tensor_series.tolist()

            processing_time = time.time()
            processing_duration = processing_time - start_time
            # print(f"processing: {processing_duration}")

            with torch.no_grad():  # Disable gradient calculation for inference
                predictions = resnet50_model(torch.stack(input_tensor).to(device))
                _, y_pred = torch.max(predictions.data, 1)
            # return y_pred
            val = pd.Series(y_pred.squeeze().tolist())

            torch_time = time.time()
            torch_duration = torch_time - processing_time
            # print(f"torch_duration: {torch_duration}")

            return val

        self.spark.udf.register("v_predict_resnet50", v_predict_tensor_resnet50)

        @pandas_udf(returnType=FloatType())
        def v_predict_tensor_vit(base64image):
            # print(f"batch size: {len(base64image)}")

            vit_model = torch.jit.load(BytesIO(bc_vit_bytes.value), map_location=torch.device(device))
            vit_model.eval()

            start_time = time.time()
            base64image_decode = base64image.apply(base64.b64decode)
            image_pil = base64image_decode.apply(lambda x: Image.open(BytesIO(x)).convert('RGB'))

            transform_pipeline = Compose([
                Resize((224, 224)),  # Explicitly resize all varying image dimensions
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            tensor_series = image_pil.apply(lambda x: transform_pipeline(x).float())

            # tensor_series = base64image.apply(lambda x: ToTensor()(Image.open(BytesIO(base64.b64decode(x)))).float())
            input_tensor = tensor_series.tolist()

            processing_time = time.time()
            processing_duration = processing_time - start_time
            # print(f"processing: {processing_duration}")

            with torch.no_grad():  # Disable gradient calculation for inference
                predictions = vit_model(torch.stack(input_tensor).to(device))
                _, y_pred = torch.max(predictions.data, 1)
            # return y_pred
            val = pd.Series(y_pred.squeeze().tolist())

            torch_time = time.time()
            torch_duration = torch_time - processing_time
            # print(f"torch_duration: {torch_duration}")

            return val

        self.spark.udf.register("v_predict_vit_udf", v_predict_tensor_vit)

        @pandas_udf(returnType=FloatType())
        def v_predict_tensor(base64image):
            # print(f"batch size: {len(base64image)}")

            start_time = time.time()
            base64image_decode = base64image.apply(base64.b64decode)
            image_pil = base64image_decode.apply(lambda x: Image.open(BytesIO(x)).convert('RGB'))

            transform_pipeline = Compose([
                Resize((375, 600)),  # Resizes to Height: 375, Width: 600
                ToTensor()
            ])

            tensor_series = image_pil.apply(lambda x: transform_pipeline(x).float())

            # tensor_series = base64image.apply(lambda x: ToTensor()(Image.open(BytesIO(base64.b64decode(x)))).float())
            input_tensor = tensor_series.tolist()

            processing_time = time.time()
            processing_duration = processing_time - start_time
            # print(f"processing: {processing_duration}")

            with torch.no_grad():  # Disable gradient calculation for inference
                predictions = model(torch.stack(input_tensor).to(device))
                _, y_pred = torch.max(predictions.data, 1)
            # return y_pred
            val = pd.Series(y_pred.squeeze().tolist())

            torch_time = time.time()
            torch_duration = torch_time - processing_time
            # print(f"torch_duration: {torch_duration}")

            return val

        self.spark.udf.register("v_predict_udf", v_predict_tensor)

        xgboostmodel = xgb.XGBClassifier(enable_categorical=True)

        xgboostmodel.load_model('../xgboost_model.model')

        def predict_xgboost_vector(gender, ethnicity, driver_class, eye_color, hair_color, is_donor, is_veteran,
                                   height_inches, weight_lbs):

            df = pd.DataFrame({
                'gender': pd.Series(gender.tolist(), dtype="category"),
                'ethnicity': pd.Series(ethnicity.tolist(), dtype="category"),
                'class': pd.Series(driver_class.tolist(), dtype="category"),
                'eye_color': pd.Series(eye_color.tolist(), dtype="category"),
                'hair_color': pd.Series(hair_color.tolist(), dtype="category"),
                'is_donor': pd.Series(is_donor.tolist(), dtype="category"),
                'is_veteran': pd.Series(is_veteran.tolist(), dtype="category"),
                'height_inches': pd.Series(height_inches.tolist(), dtype="int64"),
                'weight_lbs': pd.Series(weight_lbs.tolist(), dtype="int64")
            })

            xgb.config_context(nthread=2)

            y_pred = xgboostmodel.predict(df)

            return pd.Series(y_pred)

        predict_xgboost_vector_udf = pandas_udf(predict_xgboost_vector, FloatType())
        self.spark.udf.register("predict_xgboost_vector_udf", predict_xgboost_vector_udf)

        if self.target_table == 'idnet_no_dupe':
            df = self.spark.read.parquet("../silver/idnet_no_dupe.parquet")

            self.labeled_training_set = df

        if self.target_table == 'idnet10k_no_dupe':
            df = self.spark.read.parquet("../silver/idnet10k_no_dupe.parquet")

            self.labeled_training_set = df

        if self.target_table == 'idnet100k_no_dupe':
            df = self.spark.read.parquet("../silver/idnet100k_no_dupe.parquet")

            self.labeled_training_set = df

        self.join_df = self.spark.read.parquet("../silver/alt_docs_x5_no_dupe.parquet")

        # self.labeled_training_set.cache()
        # self.labeled_training_set.count()

    def close(self):
        self.spark.stop()

    def query_1(self):
        """
        query 1: simple CNN inference in projection

        """
        self.setup()
        self.spark.sql("select gender, v_predict_udf(imageData) as my_predictions from {labeled_training_set}",
                  labeled_training_set=self.labeled_training_set).write.format("noop").mode("overwrite").save()
        self.close()


    def query_2(self):
        """
        query 2: CNN inference in CTE, followed by aggregation

        """
        self.setup()
        self.spark.sql("""with cte as (select gender, v_predict_udf(imageData) as my_predictions from {labeled_training_set})
                     select gender, my_predictions, count(my_predictions) as count_predictions from cte group by gender, my_predictions"""
                  , labeled_training_set=self.labeled_training_set).write.format("noop").mode("overwrite").save()
        self.close()

    # %%
    def query_3(self):
        """
        query 3: simple XGBOOST inference in projection

        """
        self.setup()
        self.spark.sql("""select gender, predict_xgboost_vector_udf(gender, ethnicity, class, eye_color, hair_color, is_donor, is_veteran, height_inches, weight_lbs) 
            as my_predictions from {labeled_training_set}"""
                  , labeled_training_set=self.labeled_training_set).write.format("noop").mode("overwrite").save()
        self.close()

    # %%
    def query_4(self):
        """
        query 4: xgboost inference in CTE, followed by aggregation

        """
        self.setup()
        self.spark.sql("""with cte as (select gender, 
                     predict_xgboost_vector_udf(gender, ethnicity, class, eye_color, hair_color, is_donor, is_veteran, height_inches, weight_lbs) as my_predictions
                      from {labeled_training_set}) select gender, my_predictions, count(my_predictions) as count_predictions from cte 
                     group by gender, my_predictions"""
                  , labeled_training_set=self.labeled_training_set).write.format("noop").mode("overwrite").save()
        self.close()

    # %%
    def query_5(self):
        """
        query 5: CNN inference in CTE, followed by aggregation, followed by structured predicate

        """
        self.setup()
        self.spark.sql("""with cte as (select gender, weight_lbs, v_predict_udf(imageData) as my_predictions from {labeled_training_set}) 
                     select gender, my_predictions, count(my_predictions) as count_predictions from cte where weight_lbs > 150 
                     group by gender, my_predictions"""
                  , labeled_training_set=self.labeled_training_set).write.format("noop").mode("overwrite").save()
        self.close()

    def query_6(self):
        """

        query 6: Join to synthetic toll audit data, return all columns, find max of both CNN and XGBOOST with features intermixed
        between both tables in projection.

        """
        self.setup()
        result = self.spark.sql("""
        select lst.*, jd.*, greatest(
                v_predict_udf(lst.imageData)
            ,predict_xgboost_vector_udf(lst.gender, lst.ethnicity, lst.class, jd.driver_eye_color, jd.driver_hair_color, lst.is_donor, lst.is_veteran, lst.height_inches, jd.driver_weight)  
                            ) from {labeled_training_set} as lst 
        inner join {join_df} as jd on lst.license_number = jd.license_number 
        """, join_df=self.join_df, labeled_training_set=self.labeled_training_set).write.format("noop").mode("overwrite").save()
        self.close()

    def query_7(self):
        """
        query 7: Join to synthetic toll audit data, return all columns, CNN predict in projection

        """

        self.setup()
        result = self.spark.sql("""
        select lst.*, jd.*, v_predict_udf(lst.imageData) as my_predictions
            from {labeled_training_set} as lst 
        inner join {join_df} as jd on lst.license_number = jd.license_number 
        """, join_df=self.join_df, labeled_training_set=self.labeled_training_set).write.format("noop").mode("overwrite").save()
        self.close()

    def query_8(self):
        """

        query 8: Join to synthetic toll audit data, return all columns, XGBOOST predict in projection

        """
        self.setup()
        result = self.spark.sql("""
        select lst.*, jd.*, predict_xgboost_vector_udf(lst.gender, lst.ethnicity, lst.class, jd.driver_eye_color, jd.driver_hair_color, lst.is_donor, lst.is_veteran, lst.height_inches, jd.driver_weight) as my_predictions
            from {labeled_training_set} as lst 
        inner join {join_df} as jd on lst.license_number = jd.license_number 
        """, join_df=self.join_df, labeled_training_set=self.labeled_training_set).write.format("noop").mode("overwrite").save()
        self.close()

    def query_9(self):
        """

        query 9: Join to synthetic toll audit data, return all columns, CNN predict in predicate

        """
        self.setup()
        result = self.spark.sql("""
        select lst.*, jd.* 
            from {labeled_training_set} as lst 
        inner join {join_df} as jd on lst.license_number = jd.license_number 
            where int(v_predict_udf(lst.imageData))==0
        """, join_df=self.join_df, labeled_training_set=self.labeled_training_set).write.format("noop").mode("overwrite").save()
        self.close()

    def query_10(self):
        """

        query 10: Join to synthetic toll audit data, return all columns, XGBOOST predict in predicate

        """
        self.setup()
        result = self.spark.sql("""
        select lst.*, jd.*
            from {labeled_training_set} as lst 
        inner join {join_df} as jd on lst.license_number = jd.license_number 
            where int(predict_xgboost_vector_udf(lst.gender, lst.ethnicity, lst.class, jd.driver_eye_color, jd.driver_hair_color, lst.is_donor, lst.is_veteran, lst.height_inches, jd.driver_weight))==0
        """, join_df=self.join_df, labeled_training_set=self.labeled_training_set).write.format("noop").mode("overwrite").save()
        self.close()

    def query_11(self):
        """

        query 11: Join to synthetic toll audit data, CTE and aggregate using structured data, CNN in HAVING clause as predicate

        """
        self.setup()
        result = self.spark.sql("""with cte as (select lst.imageData, lst.gender, lst.eye_color, jd.driver_hair_color from {labeled_training_set} as lst
        inner join {join_df} as jd on lst.license_number = jd.license_number)
                     select imageData, gender, eye_color, count(driver_hair_color) as count_hair from cte group by imageData, gender, eye_color
                       having int(v_predict_udf(imageData))==0 """
                  , join_df=self.join_df, labeled_training_set=self.labeled_training_set).write.format("noop").mode("overwrite").save()
        self.close()

    def query_12(self):
        """
        query 12: Join to synthetic toll audit data, CTE and aggregate using structured data, XGBOOST in HAVING clause as predicate

        """
        self.setup()
        result = self.spark.sql("""with cte as (select lst.class, lst.gender, lst.ethnicity, lst.class, jd.driver_eye_color, jd.driver_hair_color, lst.is_donor, lst.is_veteran, lst.height_inches, jd.driver_weight
                                                from {labeled_training_set} as lst
                                       inner join {join_df} as jd
                                   on lst.license_number = jd.license_number)
        select *, count(class) as count_class
        from cte
        group by gender, ethnicity, class, driver_eye_color, driver_hair_color, is_donor, is_veteran, height_inches, driver_weight
        having int(predict_xgboost_vector_udf(gender, ethnicity, class, driver_eye_color, driver_hair_color, is_donor, is_veteran, height_inches, driver_weight))==0 """
                                , join_df=self.join_df, labeled_training_set=self.labeled_training_set).write.format("noop").mode("overwrite").save()
        self.close()

    def query_13(self):
        """

        query 13: Join to synthetic toll audit data, select all, use CNN in order by


        """
        self.setup()
        result = self.spark.sql("""
                                select lst.*, jd.*
                                from {labeled_training_set} as lst 
           inner join {join_df} as jd
                                on lst.license_number = jd.license_number
                                    order by v_predict_udf(lst.imageData) desc
                                """, join_df=self.join_df, labeled_training_set=self.labeled_training_set).write.format("noop").mode("overwrite").save()
        self.close()

    def query_14(self):
        """

        query 14: Join to synthetic toll audit data, select all, use XGBOOST in order by

        """
        self.setup()
        result = self.spark.sql("""
                                select lst.*,
                                       jd.*
                                from {labeled_training_set} as lst 
           inner join {join_df} as jd
                                on lst.license_number = jd.license_number
                                    order by predict_xgboost_vector_udf(lst.gender, lst.ethnicity, lst.class, jd.driver_eye_color,
                                                           jd.driver_hair_color, lst.is_donor, lst.is_veteran,
                                                           lst.height_inches, jd.driver_weight) desc
                                """, join_df=self.join_df, labeled_training_set=self.labeled_training_set).write.format("noop").mode("overwrite").save()
        self.close()

    def query_15(self):
        """
        query 1: simple CNN inference in projection

        """
        self.setup()

        df = self.spark.sql("""SELECT lst.id,
                                      jd.toll_id,
                                      v_predict_resnet50(lst.imageData) AS score
                               FROM {labeled_training_set} AS lst inner join {join_df} AS jd
                               ON lst.license_number = jd.license_number
                               ORDER BY score DESC LIMIT 100;""",
                            join_df=self.join_df, labeled_training_set=self.labeled_training_set).write.format(
            "noop").mode("overwrite").save()

        self.close()

    def query_16(self):
        """
        query 1: simple CNN inference in projection

        """
        self.setup()
        df = self.spark.sql("""SELECT lst.id,
                                      jd.toll_id,
                                      v_predict_vit_udf(lst.imageData) AS score
                               FROM {labeled_training_set} AS lst inner join {join_df} AS jd
                               ON lst.license_number = jd.license_number
                               ORDER BY score DESC LIMIT 100;""",
                            join_df=self.join_df, labeled_training_set=self.labeled_training_set).write.format(
            "noop").mode("overwrite").save()

        self.close()

    def query_17(self):
        """
        query 1: simple CNN inference in projection

        """
        self.setup()

        df = self.spark.sql("""
                            select lst.*,
                                   jd.*,
                                   greatest(
                                           predict_xgboost_vector_udf(lst.gender, lst.ethnicity, lst.class,
                                                               jd.driver_eye_color, jd.driver_hair_color, lst.is_donor,
                                                               lst.is_veteran, lst.height_inches, jd.driver_weight),
                                           v_predict_resnet50(lst.imageData)
                                   )
                            from {labeled_training_set} as lst 
        inner join {join_df} as jd
                            on lst.license_number = jd.license_number
                            """,
                            join_df=self.join_df, labeled_training_set=self.labeled_training_set).write.format(
            "noop").mode("overwrite").save()

        self.close()

    def query_18(self):
        """
        query 1: simple CNN inference in projection

        """
        self.setup()
        df = self.spark.sql("""
                            select lst.*,
                                   jd.*,
                                   greatest(
                                           predict_xgboost_vector_udf(lst.gender, lst.ethnicity, lst.class,
                                                               jd.driver_eye_color, jd.driver_hair_color, lst.is_donor,
                                                               lst.is_veteran, lst.height_inches, jd.driver_weight),
                                           v_predict_vit_udf(lst.imageData)
                                   )
                            from {labeled_training_set} as lst 
        inner join {join_df} as jd
                            on lst.license_number = jd.license_number
                            """,
                            join_df=self.join_df, labeled_training_set=self.labeled_training_set).write.format(
            "noop").mode("overwrite").save()


        self.close()

    def query_19(self):
        """
        query 1: simple CNN inference in projection

        """
        self.setup()
        self.spark.sql(
            "select gender, v_predict_resnet50(imageData) as my_predictions from {labeled_training_set} where int(v_predict_resnet50(imageData))==1",
            labeled_training_set=self.labeled_training_set).write.format("noop").mode("overwrite").save()

        self.close()

    def query_20(self):
        """
        query 1: simple CNN inference in projection

        """
        self.setup()
        self.spark.sql(
            "select gender, v_predict_vit_udf(imageData) as my_predictions from {labeled_training_set} where int(v_predict_vit_udf(imageData))==1",
            labeled_training_set=self.labeled_training_set).write.format("noop").mode("overwrite").save()

        self.close()






    #%%
    def run(self):
        for i in range(15,21):
            self.log = []

            query = i
            print(f'{i} monitoring start')
            # Start the monitoring thread
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

        self.spark.stop()

