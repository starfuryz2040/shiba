import psutil
import time
import threading
import datetime
import json
import xgboost as xgb
import matplotlib.pyplot as plt
import evadb
import pandas as pd


from dbms_benchmark_parent_classes.dbms_parent import DMBSParent

import logging
logger = logging.getLogger(__name__)



class EvaDB_Naive(DMBSParent):
    def __init__(self, target_table, run_num):
        super().__init__()
        self.target_table = target_table
        self.run_num = run_num
        self.system = "evadb"
        self.udf_type = "naive"
        self.join_df = "toll_audit"


    def query_1(self):
        cursor = evadb.connect('../evadb_data').cursor()



        query = (f"""
                 explain select gender, predict_with_torch_naive(imagedata)
                 from postgres_data.{self.target_table}
                 """
                 )
        #print(cursor.query(query).df())
        result = cursor.query(query).df()
        result.to_csv(f"{self.target_table}_{self.udf_type}_query1.csv")

        cursor.close

    def query_2(self):
        pass

    def query_3(self):
        cursor = evadb.connect('../evadb_data').cursor()
        query = (f"""
                 explain select gender,
                        predict_with_xgboost_naive(gender, ethnicity, class, eye_color, hair_color, is_donor,
                                                   is_veteran, height_inches, weight_lbs)
                 from postgres_data.{self.target_table}
                 """
                 )
        result = cursor.query(query).df()
        print(result)
        cursor.close

    def query_4(self):
        pass

    def query_5(self):
        pass

    def query_6(self):
        """

        query 6: Join to synthetic toll audit data, return all columns, find max of both CNN and XGBOOST with features intermixed
        between both tables in projection.

        """
        cursor = evadb.connect('../evadb_data').cursor()
        query = (f"""
        explain select *, greatest(predict_with_torch_naive(lst.imagedata), predict_with_xgboost_naive(lst.gender, lst.ethnicity, lst.class, jd.driver_eye_color, jd.driver_hair_color, lst.is_donor, lst.is_veteran, lst.height_inches, jd.driver_weight)  
                            ) from postgres_data.{self.target_table} as lst 
        join postgres_data.{self.join_df} as jd on lst.license_number = jd.license_number 
        """
                 )
        result = cursor.query(query).df()
        print(result)
        cursor.close


    def query_7(self):
        """
        query 7: Join to synthetic toll audit data, return all columns, CNN predict in projection

        """
        cursor = evadb.connect('../evadb_data').cursor()
        query = (f"""
        select *, predict_with_torch_naive(lst.imageData) as my_predictions
            from postgres_data.{self.target_table} as lst 
        join postgres_data.{self.join_df} as jd on lst.license_number = jd.license_number 
        """
                 )
        result = cursor.query(query).df()
        cursor.close


    def query_8(self):
        """

        query 8: Join to synthetic toll audit data, return all columns, XGBOOST predict in projection

        """
        cursor = evadb.connect('../evadb_data').cursor()
        query = (f"""
        select *, predict_with_xgboost_naive(lst.gender, lst.ethnicity, lst.class, jd.driver_eye_color, jd.driver_hair_color, lst.is_donor, lst.is_veteran, lst.height_inches, jd.driver_weight) as my_predictions
            from postgres_data.{self.target_table} as lst 
        join postgres_data.{self.join_df} as jd on lst.license_number = jd.license_number 
        """
                 )
        result = cursor.query(query).df()
        cursor.close



    def query_9(self):
        """

        query 9: Join to synthetic toll audit data, return all columns, CNN predict in predicate

        """
        cursor = evadb.connect('../evadb_data').cursor()
        query = (f"""
        select * 
            from postgres_data.{self.target_table} as lst 
        join postgres_data.{self.join_df} as jd on lst.license_number = jd.license_number 
            where predict_with_torch_naive(lst.imagedata) < 1
        """
                 )
        result = cursor.query(query).df()
        cursor.close


    def query_10(self):
        """

        query 10: Join to synthetic toll audit data, return all columns, XGBOOST predict in predicate


        """

        cursor = evadb.connect('../evadb_data').cursor()
        query = (f"""
        select *
            from postgres_data.{self.target_table} as lst 
        join postgres_data.{self.join_df} as jd on lst.license_number = jd.license_number 
        where predict_with_xgboost_naive(gender, ethnicity, class, driver_eye_color, driver_hair_color,  is_donor, is_veteran, height_inches, driver_weight) < 1

        """
                 )
        result = cursor.query(query).df()
        cursor.close





    def query_11(self):
        """

        query 11: Join to synthetic toll audit data, CTE and aggregate using structured data, CNN in HAVING clause as predicate

        """
        pass

    def query_12(self):
        """
        query 12: Join to synthetic toll audit data, CTE and aggregate using structured data, XGBOOST in HAVING clause as predicate

        """
        pass

    def query_13(self):
        """

        query 13: Join to synthetic toll audit data, select all, use CNN in order by


        """
        pass




    def query_14(self):
        """

        query 14: Join to synthetic toll audit data, select all, use XGBOOST in order by

        """

        pass

    #%%
    def run(self):
        for i in range(1,15):
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
            print(f'{i} stop')

            # Stop the monitoring thread
            self.update_global(False)
            monitor_thread.join() # Wait for the monitoring thread to finish
            print(f'{i} monitoring end')

            with open(f"/runlogs/{self.target_table}_{self.run_num}_{self.system}_{self.udf_type}_{query}.json", "w") as outfile:
                json.dump(self.log[:], outfile, indent=4)

