from postgres_optimized import PostgresDB_Optimized
import os

#for initial testing, consider only running idnet_no_dupe
tables = ['idnet_no_dupe', 'idnet10k_no_dupe', 'idnet100k_no_dupe']

#for initial testing, consider using Z or test
runs = [1,2,3,4,5,6,7,8,9,10]

for table in tables:
    for run in runs:
        try:
            PostgresDB_Optimized(table, run).run()
        except Exception as e:
            print(e)
       







