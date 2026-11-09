

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
        self.join_df = "toll_audit"
        self.password = urllib.parse.quote_plus("postgres")
        self.db_connection_str = f'postgresql+psycopg2://postgres:{self.password}@localhost:5434/postgres'

    def query_1(self):
        """
        query 1: simple CNN inference in projection

        """

        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            query_prep = text(f""" select pg_stat_reset();
                                                            set track_io_timing = on;
                                                        """)
            conn.execute(query_prep)

            query = text(f"""explain (analyze, buffers) select gender, predict_with_torch_optimized(imagedata) as my_predictions
                            from {self.target_table}""")
            df = pd.read_sql(query, conn)

            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query1_{self.target_table}.txt", index=False)

        with db_connection.connect() as conn2:
            post_query = text(
                f"""select funcname, calls, total_time from pg_stat_user_functions;""")
            # time.sleep(10)
            df = pd.read_sql(post_query, conn2)
            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query1_{self.target_table}_functions.txt", index=False)


    def query_2(self):
        """
        query 2: CNN inference in CTE, followed by aggregation

        """
        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            query_prep = text(f""" select pg_stat_reset();
                                                            set track_io_timing = on;
                                                        """)
            conn.execute(query_prep)

            query = text(
                f"""explain (analyze, buffers) with cte as (select gender, predict_with_torch_optimized(imagedata) as my_predictions from {self.target_table})
                   select gender, my_predictions, count(my_predictions) as count_predictions from cte group by gender, my_predictions;""")
            df = pd.read_sql(query, conn)

            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query2_{self.target_table}.txt", index=False)

        with db_connection.connect() as conn2:
            post_query = text(
                f"""select funcname, calls, total_time from pg_stat_user_functions;""")
            # time.sleep(10)
            df = pd.read_sql(post_query, conn2)
            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query2_{self.target_table}_functions.txt", index=False)

    # %%
    def query_3(self):
        """
        query 3: simple XGBOOST inference in projection

        """
        db_connection = create_engine(self.db_connection_str)


        with db_connection.connect() as conn:
            query_prep = text(f""" select pg_stat_reset();
                                                            set track_io_timing = on;
                                                        """)
            conn.execute(query_prep)

            query = text(f"""explain (analyze, buffers) select gender,
                                   predict_with_xgboost_optimized(gender, ethnicity, class, eye_color, hair_color,
                                                              is_donor, is_veteran, height_inches,
                                                              weight_lbs) as my_predictions
                            from {self.target_table};""")
            df = pd.read_sql(query, conn)

            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query3_{self.target_table}.txt", index=False)

        with db_connection.connect() as conn2:
            post_query = text(
                f"""select funcname, calls, total_time from pg_stat_user_functions;""")
            # time.sleep(10)
            df = pd.read_sql(post_query, conn2)
            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query3_{self.target_table}_functions.txt", index=False)


    # %%
    def query_4(self):
        """
        query 4: xgboost inference in CTE, followed by aggregation



        """
        db_connection = create_engine(self.db_connection_str)


        with db_connection.connect() as conn:
            query_prep = text(f""" select pg_stat_reset();
                                                            set track_io_timing = on;
                                                        """)
            conn.execute(query_prep)

            query = text(f"""explain (analyze, buffers) with cte as (select gender,
                                                predict_with_xgboost_optimized(gender, ethnicity, class, eye_color,
                                                                           hair_color, is_donor, is_veteran,
                                                                           height_inches,
                                                                           weight_lbs) as my_predictions
                                         from {self.target_table})
                            select gender, my_predictions, count(my_predictions) as count_predictions
                            from cte
                            group by gender, my_predictions;""")
            df = pd.read_sql(query, conn)

            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query4_{self.target_table}.txt", index=False)

        with db_connection.connect() as conn2:
            post_query = text(
                f"""select funcname, calls, total_time from pg_stat_user_functions;""")
            # time.sleep(10)
            df = pd.read_sql(post_query, conn2)
            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query4_{self.target_table}_functions.txt", index=False)


    def query_5(self):
        """
        query 5: CNN inference in CTE, followed by aggregation, followed by structured predicate

        """
        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            query_prep = text(f""" select pg_stat_reset();
                                                            set track_io_timing = on;
                                                        """)
            conn.execute(query_prep)

            query = text(
                f"""explain (analyze, buffers) with cte as (select gender, weight_lbs, predict_with_torch_optimized(imagedata) as my_predictions
                                from {self.target_table}) select gender, my_predictions, count(my_predictions) as count_predictions from cte where weight_lbs > 150 group by gender, my_predictions;""")
            df = pd.read_sql(query, conn)

            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query5_{self.target_table}.txt", index=False)

        with db_connection.connect() as conn2:
            post_query = text(
                f"""select funcname, calls, total_time from pg_stat_user_functions;""")
            # time.sleep(10)
            df = pd.read_sql(post_query, conn2)
            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query5_{self.target_table}_functions.txt", index=False)

    def query_6(self):
        """

        query 6: Join to synthetic toll audit data, return all columns, find max of both CNN and XGBOOST with features intermixed
        between both tables in projection.

        """

        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            query_prep = text(f""" select pg_stat_reset();
                                                            set track_io_timing = on;
                                                        """)
            conn.execute(query_prep)

            query = text(
                f"""
        explain (analyze, buffers) select lst.*, jd.*, greatest(
                predict_with_torch_optimized(lst.imageData)
            ,predict_with_xgboost_optimized(lst.gender, lst.ethnicity, lst.class, jd.driver_eye_color, jd.driver_hair_color, lst.is_donor, lst.is_veteran, lst.height_inches, jd.driver_weight)  
                            ) from {self.target_table} as lst 
        inner join {self.join_df} as jd on lst.license_number = jd.license_number 
        """)
            df = pd.read_sql(query, conn)

            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query6_{self.target_table}.txt", index=False)

        with db_connection.connect() as conn2:
            post_query = text(
                f"""select funcname, calls, total_time from pg_stat_user_functions;""")
            # time.sleep(10)
            df = pd.read_sql(post_query, conn2)
            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query6_{self.target_table}_functions.txt", index=False)


    def query_7(self):
        """
        query 7: Join to synthetic toll audit data, return all columns, CNN predict in projection

        """
        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            query_prep = text(f""" select pg_stat_reset();
                                                            set track_io_timing = on;
                                                        """)
            conn.execute(query_prep)

            query = text(
                f"""
        explain (analyze, buffers) select lst.*, jd.*, predict_with_torch_optimized(lst.imageData) as my_predictions
            from {self.target_table} as lst 
        inner join {self.join_df} as jd on lst.license_number = jd.license_number 
        """)
            df = pd.read_sql(query, conn)

            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query7_{self.target_table}.txt", index=False)

        with db_connection.connect() as conn2:
            post_query = text(
                f"""select funcname, calls, total_time from pg_stat_user_functions;""")
            # time.sleep(10)
            df = pd.read_sql(post_query, conn2)
            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query7_{self.target_table}_functions.txt", index=False)


    def query_8(self):
        """

        query 8: Join to synthetic toll audit data, return all columns, XGBOOST predict in projection

        """
        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            query_prep = text(f""" select pg_stat_reset();
                                                            set track_io_timing = on;
                                                        """)
            conn.execute(query_prep)

            query = text(
                f"""
        explain (analyze, buffers) select lst.*, jd.*, predict_with_xgboost_optimized(lst.gender, lst.ethnicity, lst.class, jd.driver_eye_color, jd.driver_hair_color, lst.is_donor, lst.is_veteran, lst.height_inches, jd.driver_weight) as my_predictions
            from {self.target_table} as lst 
        inner join {self.join_df} as jd on lst.license_number = jd.license_number 
        """)
            df = pd.read_sql(query, conn)

            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query8_{self.target_table}.txt", index=False)

        with db_connection.connect() as conn2:
            post_query = text(
                f"""select funcname, calls, total_time from pg_stat_user_functions;""")
            # time.sleep(10)
            df = pd.read_sql(post_query, conn2)
            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query8_{self.target_table}_functions.txt", index=False)


    def query_9(self):
        """

        query 9: Join to synthetic toll audit data, return all columns, CNN predict in predicate

        """
        db_connection = create_engine(self. db_connection_str)

        with db_connection.connect() as conn:
            query_prep = text(f""" select pg_stat_reset();
                                                            set track_io_timing = on;
                                                        """)
            conn.execute(query_prep)

            query = text(
                f"""
        explain (analyze, buffers) select lst.*, jd.* 
            from {self.target_table} as lst 
        inner join {self.join_df} as jd on lst.license_number = jd.license_number 
            where predict_with_torch_optimized(lst.imageData)::numeric::integer = 0
        """)
            df = pd.read_sql(query, conn)

            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query9_{self.target_table}.txt", index=False)

        with db_connection.connect() as conn2:
            post_query = text(
                f"""select funcname, calls, total_time from pg_stat_user_functions;""")
            # time.sleep(10)
            df = pd.read_sql(post_query, conn2)
            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query9_{self.target_table}_functions.txt", index=False)

    def query_10(self):
        """

        query 10: Join to synthetic toll audit data, return all columns, XGBOOST predict in predicate

        """
        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            query_prep = text(f""" select pg_stat_reset();
                                                            set track_io_timing = on;
                                                        """)
            conn.execute(query_prep)

            query = text(
                f"""
        explain (analyze, buffers) select lst.*, jd.*
            from {self.target_table} as lst 
        inner join {self.join_df} as jd on lst.license_number = jd.license_number 
            where predict_with_xgboost_optimized(lst.gender, lst.ethnicity, lst.class, jd.driver_eye_color, jd.driver_hair_color, lst.is_donor, lst.is_veteran, lst.height_inches, jd.driver_weight)::numeric::integer = 0
        """)
            df = pd.read_sql(query, conn)

            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query10_{self.target_table}.txt", index=False)

        with db_connection.connect() as conn2:
            post_query = text(
                f"""select funcname, calls, total_time from pg_stat_user_functions;""")
            # time.sleep(10)
            df = pd.read_sql(post_query, conn2)
            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query10_{self.target_table}_functions.txt", index=False)

    def query_11(self):
        """

        query 11: Join to synthetic toll audit data, CTE and aggregate using structured data, CNN in HAVING clause as predicate

        """

        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            query_prep = text(f""" select pg_stat_reset();
                                                            set track_io_timing = on;
                                                        """)
            conn.execute(query_prep)

            query = text(
                f"""explain (analyze, buffers) with cte as (select lst.imageData, lst.gender, lst.eye_color, jd.driver_hair_color from {self.target_table} as lst
        inner join {self.join_df} as jd on lst.license_number = jd.license_number)
                     select imageData, gender, eye_color, count(driver_hair_color) as count_hair from cte group by imageData, gender, eye_color
                       having predict_with_torch_optimized(imageData)::numeric::integer = 0 """)
            df = pd.read_sql(query, conn)

            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query11_{self.target_table}.txt", index=False)

        with db_connection.connect() as conn2:
            post_query = text(
                f"""select funcname, calls, total_time from pg_stat_user_functions;""")
            # time.sleep(10)
            df = pd.read_sql(post_query, conn2)
            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query11_{self.target_table}_functions.txt", index=False)

    def query_12(self):
        """
        query 12: Join to synthetic toll audit data, CTE and aggregate using structured data, XGBOOST in HAVING clause as predicate

        """
        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            query_prep = text(f""" select pg_stat_reset();
                                                            set track_io_timing = on;
                                                        """)
            conn.execute(query_prep)

            query = text(
                f"""explain (analyze, buffers) with cte as (select lst.class, lst.gender, lst.ethnicity, jd.driver_eye_color, jd.driver_hair_color, lst.is_donor, lst.is_veteran, lst.height_inches, jd.driver_weight
                                                from {self.target_table} as lst
                                       inner join {self.join_df} as jd
                                   on lst.license_number = jd.license_number)
        select *, count(class) as count_class
        from cte
        group by gender, ethnicity, class, driver_eye_color, driver_hair_color, is_donor, is_veteran, height_inches, driver_weight
        having predict_with_xgboost_optimized(gender, ethnicity, class, driver_eye_color, driver_hair_color, is_donor, is_veteran, height_inches, driver_weight)::numeric::integer = 0 """)
            df = pd.read_sql(query, conn)

            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query12_{self.target_table}.txt", index=False)

        with db_connection.connect() as conn2:
            post_query = text(
                f"""select funcname, calls, total_time from pg_stat_user_functions;""")
            # time.sleep(10)
            df = pd.read_sql(post_query, conn2)
            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query12_{self.target_table}_functions.txt", index=False)


    def query_13(self):
        """

        query 13: Join to synthetic toll audit data, select all, use CNN in order by


        """
        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            query_prep = text(f""" select pg_stat_reset();
                                                            set track_io_timing = on;
                                                        """)
            conn.execute(query_prep)

            query = text(
                f"""
                                explain (analyze, buffers)  select lst.*, jd.*
                                from {self.target_table} as lst 
           inner join {self.join_df} as jd
                                on lst.license_number = jd.license_number
                                    order by predict_with_torch_optimized(lst.imageData) desc
                                """)
            df = pd.read_sql(query, conn)

            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query13_{self.target_table}.txt", index=False)

        with db_connection.connect() as conn2:
            post_query = text(
                f"""select funcname, calls, total_time from pg_stat_user_functions;""")
            # time.sleep(10)
            df = pd.read_sql(post_query, conn2)
            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query13_{self.target_table}_functions.txt", index=False)


    def query_14(self):
        """

        query 14: Join to synthetic toll audit data, select all, use XGBOOST in order by

        """
        db_connection = create_engine(self.db_connection_str)

        with db_connection.connect() as conn:
            query_prep = text(f""" select pg_stat_reset();
                                                            set track_io_timing = on;
                                                        """)
            conn.execute(query_prep)

            query = text(
                f"""
                                explain (analyze, buffers)  select lst.*,
                                       jd.*
                                from {self.target_table} as lst 
           inner join {self.join_df} as jd
                                on lst.license_number = jd.license_number
                                    order by predict_with_xgboost_optimized(lst.gender, lst.ethnicity, lst.class, jd.driver_eye_color,
                                                           jd.driver_hair_color, lst.is_donor, lst.is_veteran,
                                                           lst.height_inches, jd.driver_weight) desc
                                """)
            df = pd.read_sql(query, conn)

            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query14_{self.target_table}.txt", index=False)

        with db_connection.connect() as conn2:
            post_query = text(
                f"""select funcname, calls, total_time from pg_stat_user_functions;""")
            # time.sleep(10)
            df = pd.read_sql(post_query, conn2)
            df.to_csv(f"/querystatslogs/postgres_{self.udf_type}/query14_{self.target_table}_functions.txt", index=False)






    #%%
    def run(self):
        for i in range(1,15):
            self.log = []

            query = i


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




