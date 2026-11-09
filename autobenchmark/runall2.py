#from duckdb_naive import DuckDB_Naive
from duckdb_optimized import DuckDB_Optimized
#from evadb_naive import EvaDB_Naive
from evadb_optimized import EvaDB_Optimized
from clickhouse_naive import ClickHouseDB_Naive
#from postgres_naive import PostgresDB_Naive
from postgres_optimized import PostgresDB_Optimized
#from spark_naive import SparkDB_Naive
from spark_optimized import SparkDB_Optimized


tables = ['idnet', 'idnet10k']

runs = [45,46,47,48,49,50]
#32 1 run all queries all systems
#33 HT disabled
#34 thread limit



for table in tables:
    for run in runs:


        try:
            SparkDB_Optimized(table, run).run()
        except Exception as e:
            print(e)
        

        







