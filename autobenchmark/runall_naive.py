from duckdb_naive import DuckDB_Naive
from evadb_naive import EvaDB_Naive
from clickhouse_naive import ClickHouseDB_Naive
from postgres_naive import PostgresDB_Naive
from spark_naive import SparkDB_Naive
import os


#for initial testing, consider only running idnet_no_dupe
tables = ['idnet_no_dupe', 'idnet10k_no_dupe', 'idnet100k_no_dupe']

#for initial testing, consider using Z or test
runs = [1,2,3,4,5,6,7,8,9,10]


for table in tables:
    for run in runs:

        try:
            DuckDB_Naive(table, run).run()
        except Exception as e:
            print(e)
        
        try:
            EvaDB_Naive(table, run).run()
        except Exception as e:
            print(e)
        
        try:
            PostgresDB_Naive(table, run).run()
        except Exception as e:
            print(e)

       
        try:
            SparkDB_Naive(table, run).run()
        except Exception as e:
            print(e)

        
        try:
            ClickHouseDB_Naive(table, run).run()
        except Exception as e:
            print(e)








