#from duckdb_naive import DuckDB_Naive
#from duckdb_optimized import DuckDB_Optimized
#from evadb_naive import EvaDB_Naive
from evadb_optimized import EvaDB_Optimized
#from clickhouse_naive import ClickHouseDB_Naive
#from clickhouse_optimized import ClickHouseDB_Optimized
#from clickhouse_optimized_vectorized import ClickHouseDB_Optimized_Vectorized
#from postgres_naive import PostgresDB_Naive
from postgres_optimized import PostgresDB_Optimized
#from spark_naive import SparkDB_Naive
#from spark_optimized import SparkDB_Optimized
import os



tables = ['idnet100k_no_dupe']

#runs = [200, 201, 202, 203, 204, 204, 205, 206, 207, 208, 209, 210]
runs = [1002, 1003, 1004]
#32 1 run all queries all systems
#33 HT disabled
#34 thread limit






for table in tables:
    for run in runs:
        """
        try:
            DuckDB_Naive(table, run).run()
        except Exception as e:
            print(e)

        try:
            DuckDB_Optimized(table, run).run()
        except Exception as e:
            print(e)
        
        try:
            EvaDB_Naive(table, run).run()
        except Exception as e:
            print(e)
        """
        try:
            EvaDB_Optimized(table, run).run()
        except Exception as e:
            print(e)

        """
        try:
            PostgresDB_Naive(table, run).run()
        except Exception as e:
            print(e)
        
        try:
            PostgresDB_Optimized(table, run).run()
        except Exception as e:
            print(e)
       
        try:
            SparkDB_Naive(table, run).run()
        except Exception as e:
            print(e)

        try:
            SparkDB_Optimized(table, run).run()
        except Exception as e:
            print(e)



        try:
            ClickHouseDB_Naive(table, run).run()
        except Exception as e:
            print(e)


        try:
            ClickHouseDB_Optimized(table, run).run()
        except Exception as e:
            print(e)

        try:
            ClickHouseDB_Optimized_Vectorized(table, run).run()
        except Exception as e:
            print(e)
        """








