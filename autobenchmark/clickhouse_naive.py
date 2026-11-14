import clickhouse_connect

from clickhouse_connect.driver.tools import insert_file
import pyarrow
import pandas as pd

import matplotlib.pyplot as plt

import psutil
import time
import threading
import datetime
import json

from dbms_benchmark_parent_classes.dbms_parent import DMBSParent


class ClickHouseDB_Naive(DMBSParent):
    def __init__(self, target_table, run_num):
        super().__init__()
        self.target_table = target_table
        self.run_num = run_num
        self.system = "clickhouse"
        self.udf_type = "naive"
        self.join_df = "toll_audit_no_dupe"




    def query_1(self):
        client = clickhouse_connect.get_client(host='localhost', username='default', password='', send_receive_timeout=14400)
        result = client.query(f'select gender, torch_function_python(name) as my_predictions from {self.target_table}').result_rows
        df = pd.DataFrame(result)
        df
        client.close()

    def query_2(self):
        client = clickhouse_connect.get_client(host='localhost', username='default', password='', send_receive_timeout=14400)
        result = client.query(
            f'with cte as (select gender, torch_function_python(name) as my_predictions from {self.target_table}) select gender, my_predictions, count(my_predictions) as count_predictions from cte group by gender, my_predictions').result_rows
        df = pd.DataFrame(result)
        df
        client.close()

    # %%
    def query_3(self):
        client = clickhouse_connect.get_client(host='localhost', username='default', password='', send_receive_timeout=14400)
        result = client.query(
            f"select gender, xgboost_function_python(gender, ifNull(ethnicity, 'None') as ethnicity, class, eye_color, ifNull(hair_color,'None') as hair_color, is_donor, is_veteran, height_inches, weight_lbs) from {self.target_table}").result_rows
        df = pd.DataFrame(result)
        df
        client.close()

    # %%
    def query_4(self):
        client = clickhouse_connect.get_client(host='localhost', username='default', password='', send_receive_timeout=14400)

        result = client.query(
            f"with cte as (select gender, xgboost_function_python(gender, ifNull(ethnicity, 'None') as ethnicity, class, eye_color, ifNull(hair_color,'None') as hair_color, is_donor, is_veteran, height_inches, weight_lbs) as my_predictions from {self.target_table}) select gender, my_predictions, count(my_predictions) as count_predictions from cte group by gender, my_predictions").result_rows
        df = pd.DataFrame(result)
        df
        client.close()

    # %%
    def query_5(self):
        client = clickhouse_connect.get_client(host='localhost', username='default', password='', send_receive_timeout=14400)

        result = client.query(
            f'with cte as (select gender, weight_lbs, torch_function_python(name) as my_predictions from {self.target_table}) select gender, my_predictions, count(my_predictions) as count_predictions from cte where (weight_lbs > 150) group by gender, my_predictions').result_rows
        df = pd.DataFrame(result)
        df
        client.close()

    def query_6(self):
        """

        query 6: Join to synthetic toll audit data, return all columns, find max of both CNN and XGBOOST with features intermixed
        between both tables in projection.

        """
        client = clickhouse_connect.get_client(host='localhost', username='default', password='',
                                               send_receive_timeout=14400)

        result = client.query(
            f"""
        select lst.*, jd.*, greatest(
                torch_function_python(lst.name)
            ,xgboost_function_python(lst.gender, ifNull(lst.ethnicity,'None') as ethnicity, lst.class, jd.driver_eye_color, ifNull(jd.driver_hair_color,'None') as driver_hair_color, lst.is_donor, lst.is_veteran, lst.height_inches, jd.driver_weight)  
                            ) from {self.target_table} as lst 
        inner join {self.join_df} as jd on lst.license_number = jd.license_number 
        """).result_rows
        df = pd.DataFrame(result)
        df
        client.close()




    def query_7(self):
        """
        query 7: Join to synthetic toll audit data, return all columns, CNN predict in projection

        """

        client = clickhouse_connect.get_client(host='localhost', username='default', password='',
                                               send_receive_timeout=14400)

        result = client.query(
            f"""
        select lst.*, jd.*, torch_function_python(lst.name) as my_predictions
            from {self.target_table} as lst 
        inner join {self.join_df} as jd on lst.license_number = jd.license_number 
        """).result_rows
        df = pd.DataFrame(result)
        df
        client.close()





    def query_8(self):
        """

        query 8: Join to synthetic toll audit data, return all columns, XGBOOST predict in projection

        """
        client = clickhouse_connect.get_client(host='localhost', username='default', password='',
                                               send_receive_timeout=14400)

        result = client.query(
            f"""
        select lst.*, jd.*, xgboost_function_python(lst.gender, ifNull(lst.ethnicity, 'None') as ethnicity, lst.class, jd.driver_eye_color, ifNull(driver_hair_color,'None') as driver_hair_color, lst.is_donor, lst.is_veteran, lst.height_inches, jd.driver_weight) as my_predictions
            from {self.target_table} as lst 
        inner join {self.join_df} as jd on lst.license_number = jd.license_number 
        """).result_rows
        df = pd.DataFrame(result)
        df
        client.close()



    def query_9(self):
        """

        query 9: Join to synthetic toll audit data, return all columns, CNN predict in predicate

        """

        client = clickhouse_connect.get_client(host='localhost', username='default', password='',
                                               send_receive_timeout=14400)

        result = client.query(
            f"""
        select lst.*, jd.* 
            from {self.target_table} as lst 
        inner join {self.join_df} as jd on lst.license_number = jd.license_number 
            where torch_function_python(lst.name)::numeric::integer = 0
        """).result_rows
        df = pd.DataFrame(result)
        df
        client.close()




    def query_10(self):
        """

        query 10: Join to synthetic toll audit data, return all columns, XGBOOST predict in predicate

        """
        client = clickhouse_connect.get_client(host='localhost', username='default', password='',
                                               send_receive_timeout=14400)

        result = client.query(
            f"""
        select lst.*, jd.*
            from {self.target_table} as lst 
        inner join {self.join_df} as jd on lst.license_number = jd.license_number 
            where xgboost_function_python(lst.gender, ifNull(lst.ethnicity, 'None') as ethnicity, lst.class, jd.driver_eye_color, ifNull(driver_hair_color,'None') as driver_hair_color, lst.is_donor, lst.is_veteran, lst.height_inches, jd.driver_weight)::numeric::integer = 0
        """).result_rows
        df = pd.DataFrame(result)
        df
        client.close()




    def query_11(self):
        """

        query 11: Join to synthetic toll audit data, CTE and aggregate using structured data, CNN in HAVING clause as predicate

        """

        client = clickhouse_connect.get_client(host='localhost', username='default', password='',
                                               send_receive_timeout=14400)

        result = client.query(
            f"""with cte as (select lst.name, lst.imageData, lst.gender, lst.eye_color, jd.driver_hair_color from {self.target_table} as lst
        inner join {self.join_df} as jd on lst.license_number = jd.license_number)
                     select name, imageData, gender, eye_color, count(driver_hair_color) as count_hair from cte group by name, imageData, gender, eye_color
                       having torch_function_python(name)::numeric::integer = 0 """).result_rows
        df = pd.DataFrame(result)
        df
        client.close()



    def query_12(self):
        """
        query 12: Join to synthetic toll audit data, CTE and aggregate using structured data, XGBOOST in HAVING clause as predicate

        """
        client = clickhouse_connect.get_client(host='localhost', username='default', password='',
                                               send_receive_timeout=14400)

        result = client.query(
            f"""with cte as (select lst.class, lst.gender, lst.ethnicity, jd.driver_eye_color, jd.driver_hair_color, lst.is_donor, lst.is_veteran, lst.height_inches, jd.driver_weight
                                                from {self.target_table} as lst
                                       inner join {self.join_df} as jd
                                   on lst.license_number = jd.license_number)
        select gender, ethnicity, class, driver_eye_color, driver_hair_color, is_donor, is_veteran, height_inches, driver_weight, count(class) as count_class
        from cte
        group by gender, ethnicity, class, driver_eye_color, driver_hair_color, is_donor, is_veteran, height_inches, driver_weight
        having xgboost_function_python(gender, ifNull(ethnicity, 'None') as ethnicity, class, driver_eye_color, ifNull(driver_hair_color,'None') as driver_hair_color, is_donor, is_veteran, height_inches, driver_weight)::numeric::integer = 0 """).result_rows
        df = pd.DataFrame(result)
        df
        client.close()




    def query_13(self):
        """

        query 13: Join to synthetic toll audit data, select all, use CNN in order by


        """
        client = clickhouse_connect.get_client(host='localhost', username='default', password='',
                                               send_receive_timeout=14400)

        result = client.query(
            f"""
                                select lst.*, jd.*
                                from {self.target_table} as lst 
           inner join {self.join_df} as jd
                                on lst.license_number = jd.license_number
                                    order by torch_function_python(lst.name) desc
                                """).result_rows
        df = pd.DataFrame(result)
        df
        client.close()





    def query_14(self):
        """

        query 14: Join to synthetic toll audit data, select all, use XGBOOST in order by

        """
        client = clickhouse_connect.get_client(host='localhost', username='default', password='',
                                               send_receive_timeout=14400)

        result = client.query(
            f"""
                                select lst.*,
                                       jd.*
                                from {self.target_table} as lst 
           inner join {self.join_df} as jd
                                on lst.license_number = jd.license_number
                                    order by xgboost_function_python(lst.gender, ifNull(lst.ethnicity, 'None') as ethnicity, lst.class, jd.driver_eye_color,
                                                           ifNull(driver_hair_color,'None') as driver_hair_color, lst.is_donor, lst.is_veteran,
                                                           lst.height_inches, jd.driver_weight) desc
                                """).result_rows
        df = pd.DataFrame(result)
        df
        client.close()


    
    def query_15(self):
        """
        query 1: simple CNN inference in projection

        """
        client = clickhouse_connect.get_client(host='localhost', username='default', password='', send_receive_timeout=14400)
        result = client.query(
                f"""SELECT lst.id,
                                      jd.toll_id,
                                      resnet50_function_python(lst.name) AS score
                               FROM {self.target_table} AS lst inner join {self.join_df} AS jd
                               ON lst.license_number = jd.license_number
                               ORDER BY score DESC LIMIT 100;""").result_rows
        df = pd.DataFrame(result)
        df
        client.close()


    def query_16(self):
        """
        query 1: simple CNN inference in projection

        """
        client = clickhouse_connect.get_client(host='localhost', username='default', password='', send_receive_timeout=14400)
        result = client.query(
                f"""SELECT lst.id,
                                              jd.toll_id,
                                              vit_function_python(lst.name) AS score
                                       FROM {self.target_table} AS lst inner join {self.join_df} AS jd
                                       ON lst.license_number = jd.license_number
                                       ORDER BY score DESC LIMIT 100;""").result_rows
        df = pd.DataFrame(result)
        df
        client.close()

    def query_17(self):
        """
        query 1: simple CNN inference in projection

        """
        client = clickhouse_connect.get_client(host='localhost', username='default', password='', send_receive_timeout=14400)
        result = client.query(
                f"""
                            select lst.*,
                                   jd.*,
                                   greatest(
                                           xgboost_function_python(lst.gender, lst.ethnicity, lst.class,
                                                               jd.driver_eye_color, jd.driver_hair_color, lst.is_donor,
                                                               lst.is_veteran, lst.height_inches, jd.driver_weight),
                                           resnet50_function_python(lst.name)
                                   )
                            from {self.target_table} as lst 
        inner join {self.join_df} as jd
                            on lst.license_number = jd.license_number
                            ;""").result_rows
        df = pd.DataFrame(result)
        df
        client.close()

    def query_18(self):
        """
        query 1: simple CNN inference in projection

        """
        client = clickhouse_connect.get_client(host='localhost', username='default', password='', send_receive_timeout=14400)
        result = client.query(
                f"""
                            select lst.*,
                                   jd.*,
                                   greatest(
                                           xgboost_function_python(lst.gender, lst.ethnicity, lst.class,
                                                               jd.driver_eye_color, jd.driver_hair_color, lst.is_donor,
                                                               lst.is_veteran, lst.height_inches, jd.driver_weight),
                                           vit_function_python(lst.name)
                                   )
                            from {self.target_table} as lst 
        inner join {self.join_df} as jd
                            on lst.license_number = jd.license_number
                            ;""").result_rows
        df = pd.DataFrame(result)
        df
        client.close()

    def query_19(self):
        """
        query 1: simple CNN inference in projection

        """
        client = clickhouse_connect.get_client(host='localhost', username='default', password='', send_receive_timeout=14400)
        result = client.query(
                f"""select gender, resnet50_function_python(name) as my_predictions from {self.target_table} where resnet50_function_python(name)::Int8 == 1;""").result_rows
        df = pd.DataFrame(result)
        df
        client.close()


    def query_20(self):
        """
        query 1: simple CNN inference in projection

        """
        client = clickhouse_connect.get_client(host='localhost', username='default', password='', send_receive_timeout=14400)
        result = client.query(
                f"""select gender, vit_function_python(name) as my_predictions from {self.target_table} where vit_function_python(name)::Int8 = 1;""").result_rows
        df = pd.DataFrame(result)
        df
        client.close()





    #%%
    def run(self):
        for i in range(1,21):
        #for i in [1]:
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

