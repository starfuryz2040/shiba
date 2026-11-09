

import psutil
import time
import threading
import datetime
import json
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import pandas as pd
import pandas as pd
from sqlalchemy import create_engine, text
import csv

import pandas as pd
import pyarrow
import pyspark
import urllib.parse
import psycopg2

from dbms_benchmark_parent_classes.dbms_parent import DMBSParent


class PostgresDB_Optimized(DMBSParent):
    def __init__(self, target_table, run_num):
        super().__init__()
        self.target_table = target_table
        self.run_num = run_num
        self.system = "postgres"
        self.udf_type = "optimized"
        self.join_df = "toll_audit_no_dupe"
        self.password = urllib.parse.quote_plus("postgres")
        self.db_connection_str = f'postgresql+psycopg2://postgres:{self.password}@localhost:5434/postgres'

    def query_1(self):
        """
        query 1: simple CNN inference in projection

        """

        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            query = text(f"""select gender, predict_with_torch_optimized("imageData") as my_predictions
                            from {self.target_table}""")
            result = conn.execute(query).fetchall()


    def query_2(self):
        """
        query 2: CNN inference in CTE, followed by aggregation

        """
        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            query = text(
                f"""with cte as (select gender, predict_with_torch_optimized("imageData") as my_predictions from {self.target_table})
                   select gender, my_predictions, count(my_predictions) as count_predictions from cte group by gender, my_predictions;""")
            result = conn.execute(query).fetchall()

    # %%
    def query_3(self):
        """
        query 3: simple XGBOOST inference in projection

        """
        db_connection = create_engine(self.db_connection_str)


        with db_connection.connect() as conn:
            query = text(f"""select gender,
                                   predict_with_xgboost_optimized(gender, ethnicity, class, eye_color, hair_color,
                                                              is_donor, is_veteran, height_inches,
                                                              weight_lbs) as my_predictions
                            from {self.target_table};""")
            result = conn.execute(query).fetchall()


    # %%
    def query_4(self):
        """
        query 4: xgboost inference in CTE, followed by aggregation



        """
        db_connection = create_engine(self.db_connection_str)


        with db_connection.connect() as conn:
            query = text(f"""with cte as (select gender,
                                                predict_with_xgboost_optimized(gender, ethnicity, class, eye_color,
                                                                           hair_color, is_donor, is_veteran,
                                                                           height_inches,
                                                                           weight_lbs) as my_predictions
                                         from {self.target_table})
                            select gender, my_predictions, count(my_predictions) as count_predictions
                            from cte
                            group by gender, my_predictions;""")
            result = conn.execute(query).fetchall()


    def query_5(self):
        """
        query 5: CNN inference in CTE, followed by aggregation, followed by structured predicate

        """
        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            query = text(
                f"""with cte as (select gender, weight_lbs, predict_with_torch_optimized("imageData") as my_predictions
                                from {self.target_table}) select gender, my_predictions, count(my_predictions) as count_predictions from cte where weight_lbs > 150 group by gender, my_predictions;""")
            result = conn.execute(query).fetchall()
    def query_6(self):
        """

        query 6: Join to synthetic toll audit data, return all columns, find max of both CNN and XGBOOST with features intermixed
        between both tables in projection.

        """

        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            #query = text("SET max_parallel_workers_per_gather=8")
            #conn.execute(query)

            query = text(
                f"""
        select lst.*, jd.*, greatest(
                predict_with_torch_optimized(lst."imageData")
            ,predict_with_xgboost_optimized(lst.gender, lst.ethnicity, lst.class, jd.driver_eye_color, jd.driver_hair_color, lst.is_donor, lst.is_veteran, lst.height_inches, jd.driver_weight)  
                            ) from {self.target_table} as lst 
        inner join {self.join_df} as jd on lst.license_number = jd.license_number 
        """)
            result = conn.execute(query).fetchall()
            #print(len(result))
            #for x in result:
            #    print(x)


    def query_7(self):
        """
        query 7: Join to synthetic toll audit data, return all columns, CNN predict in projection

        """
        db_connection = create_engine(self.db_connection_str)



        with db_connection.connect() as conn:
            #query = text("SET max_parallel_workers_per_gather=8")
            #conn.execute(query)

            query = text(
                f"""
        select lst.*, jd.*, predict_with_torch_optimized(lst."imageData") as my_predictions
            from {self.target_table} as lst 
        inner join {self.join_df} as jd on lst.license_number = jd.license_number 
        """)
            result = conn.execute(query).fetchall()
            #print(len(result))


    def query_8(self):
        """

        query 8: Join to synthetic toll audit data, return all columns, XGBOOST predict in projection

        """
        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            query = text(
                f"""
        select lst.*, jd.*, predict_with_xgboost_optimized(lst.gender, lst.ethnicity, lst.class, jd.driver_eye_color, jd.driver_hair_color, lst.is_donor, lst.is_veteran, lst.height_inches, jd.driver_weight) as my_predictions
            from {self.target_table} as lst 
        inner join {self.join_df} as jd on lst.license_number = jd.license_number 
        """)
            result = conn.execute(query).fetchall()
            #print(len(result))


    def query_9(self):
        """

        query 9: Join to synthetic toll audit data, return all columns, CNN predict in predicate

        """
        db_connection = create_engine(self. db_connection_str)

        with db_connection.connect() as conn:
            query = text(
                f"""
        select lst.*, jd.* 
            from {self.target_table} as lst 
        inner join {self.join_df} as jd on lst.license_number = jd.license_number 
            where predict_with_torch_optimized(lst."imageData")::numeric::integer = 0
        """)
            result = conn.execute(query).fetchall()
            #print(len(result))

    def query_10(self):
        """

        query 10: Join to synthetic toll audit data, return all columns, XGBOOST predict in predicate

        """
        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            query = text(
                f"""
        select lst.*, jd.*
            from {self.target_table} as lst 
        inner join {self.join_df} as jd on lst.license_number = jd.license_number 
            where predict_with_xgboost_optimized(lst.gender, lst.ethnicity, lst.class, jd.driver_eye_color, jd.driver_hair_color, lst.is_donor, lst.is_veteran, lst.height_inches, jd.driver_weight)::numeric::integer = 0
        """)
            result = conn.execute(query).fetchall()
            #print(len(result))

    def query_11(self):
        """

        query 11: Join to synthetic toll audit data, CTE and aggregate using structured data, CNN in HAVING clause as predicate

        """

        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            query = text(
                f"""with cte as (select lst."imageData", lst.gender, lst.eye_color, jd.driver_hair_color from {self.target_table} as lst
        inner join {self.join_df} as jd on lst.license_number = jd.license_number)
                     select "imageData", gender, eye_color, count(driver_hair_color) as count_hair from cte group by "imageData", gender, eye_color
                       having predict_with_torch_optimized("imageData")::numeric::integer = 0 """)
            result = conn.execute(query).fetchall()
            #print(len(result))

    def query_12(self):
        """
        query 12: Join to synthetic toll audit data, CTE and aggregate using structured data, XGBOOST in HAVING clause as predicate

        """
        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            query = text(
                f"""with cte as (select lst.class, lst.gender, lst.ethnicity, jd.driver_eye_color, jd.driver_hair_color, lst.is_donor, lst.is_veteran, lst.height_inches, jd.driver_weight
                                                from {self.target_table} as lst
                                       inner join {self.join_df} as jd
                                   on lst.license_number = jd.license_number)
        select *, count(class) as count_class
        from cte
        group by gender, ethnicity, class, driver_eye_color, driver_hair_color, is_donor, is_veteran, height_inches, driver_weight
        having predict_with_xgboost_optimized(gender, ethnicity, class, driver_eye_color, driver_hair_color, is_donor, is_veteran, height_inches, driver_weight)::numeric::integer = 0 """)
            result = conn.execute(query).fetchall()
            #print(len(result))


    def query_13(self):
        """

        query 13: Join to synthetic toll audit data, select all, use CNN in order by


        """
        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            query = text(
                f"""
                                select lst.*, jd.*
                                from {self.target_table} as lst 
           inner join {self.join_df} as jd
                                on lst.license_number = jd.license_number
                                    order by predict_with_torch_optimized(lst."imageData") desc
                                """)
            result = conn.execute(query).fetchall()
            #print(len(result))


    def query_14(self):
        """

        query 14: Join to synthetic toll audit data, select all, use XGBOOST in order by

        """
        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            query = text(
                f"""
                                select lst.*,
                                       jd.*
                                from {self.target_table} as lst 
           inner join {self.join_df} as jd
                                on lst.license_number = jd.license_number
                                    order by predict_with_xgboost_optimized(lst.gender, lst.ethnicity, lst.class, jd.driver_eye_color,
                                                           jd.driver_hair_color, lst.is_donor, lst.is_veteran,
                                                           lst.height_inches, jd.driver_weight) desc
                                """)
            result = conn.execute(query).fetchall()
            #print(len(result))



    def query_15(self):
        """
        query 1: simple CNN inference in projection

        """
        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            query = text(
                f"""SELECT lst.id,
                                      jd.toll_id,
                                      predict_with_resnet50_optimized(lst."imageData") AS score
                               FROM {self.target_table} AS lst inner join {self.join_df} AS jd
                               ON lst.license_number = jd.license_number
                               ORDER BY score DESC LIMIT 100;""")
            result = conn.execute(query).fetchall()


    def query_16(self):
        """
        query 1: simple CNN inference in projection

        """
        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            query = text(
                f"""SELECT lst.id,
                                              jd.toll_id,
                                              predict_with_vit_optimized(lst."imageData") AS score
                                       FROM {self.target_table} AS lst inner join {self.join_df} AS jd
                                       ON lst.license_number = jd.license_number
                                       ORDER BY score DESC LIMIT 100;""")
            result = conn.execute(query).fetchall()

    def query_17(self):
        """
        query 1: simple CNN inference in projection

        """
        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            query = text(
                f"""
                            select lst.*,
                                   jd.*,
                                   greatest(
                                           predict_with_xgboost_optimized(lst.gender, lst.ethnicity, lst.class,
                                                               jd.driver_eye_color, jd.driver_hair_color, lst.is_donor,
                                                               lst.is_veteran, lst.height_inches, jd.driver_weight),
                                           predict_with_resnet50_optimized(lst."imageData")
                                   )
                            from {self.target_table} as lst 
        inner join {self.join_df} as jd
                            on lst.license_number = jd.license_number
                            ;""")
            result = conn.execute(query).fetchall()

    def query_18(self):
        """
        query 1: simple CNN inference in projection

        """
        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            query = text(
                f"""
                            select lst.*,
                                   jd.*,
                                   greatest(
                                           predict_with_xgboost_optimized(lst.gender, lst.ethnicity, lst.class,
                                                               jd.driver_eye_color, jd.driver_hair_color, lst.is_donor,
                                                               lst.is_veteran, lst.height_inches, jd.driver_weight),
                                           predict_with_vit_optimized(lst."imageData")
                                   )
                            from {self.target_table} as lst 
        inner join {self.join_df} as jd
                            on lst.license_number = jd.license_number
                            ;""")
            result = conn.execute(query).fetchall()

    def query_19(self):
        """
        query 1: simple CNN inference in projection

        """
        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            query = text(
                f"""select gender, predict_with_resnet50_optimized("imageData") as my_predictions from {self.target_table} where predict_with_resnet50_optimized("imageData")::numeric::integer = 1;""")
            result = conn.execute(query).fetchall()


    def query_20(self):
        """
        query 1: simple CNN inference in projection

        """
        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            query = text(
                f"""select gender, predict_with_vit_optimized("imageData") as my_predictions from {self.target_table} where predict_with_vit_optimized("imageData")::numeric::integer = 1;""")
            result = conn.execute(query).fetchall()


    #%%
    def run(self):
        for i in range(15,21):
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

