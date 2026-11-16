# SHIBA: Sql Hybrid Inference Benchmark Analytics

Benchmark framework to analyze and compare sql inference of hybrid/complex data and models. 

## Description

We developed a framework to compare multiple popular SQL
databases and analytic processors in sql inference query execution with complex DS/ML models. Databases were populated with a subset
of the IDNET dataset. This multimodal and complex data allows for realistic queries and models as a basis of the 
comparison. If interest persists, this will be developed further.

## Getting Started

DuckDb, Evadb, and Pyspark Python dependencies will need to be installed in the project venv. Postgres python dependencies 
will need to be installed at the system level. Clickhouse dependencies
can be installed at the system level, but the top line of the UDFS can be set to a path where a venv exists. That path
is currently set to /shiba_env/bin/python3. Please use the requirements.txt file to install the dependencies.

Any IDE that supports py and ipynb files. Pycharm is recommended. A database management tool such as Datagrip will
simplify loading of Postgres UDFs, and general troubleshooting. 

### Dependencies

please install the following systems
* OS - Ubuntu 24.04 LTS
* Clickhouse 25
* Postgres 18
* postgres plpython3-18
* Python 3.12
* openjdk-21-jdk (with env vars set for home and path)
* https://pytorch.org/ for pytorch installation. version 2.9.1 is the target. CPU is the default option in SHIBA. GPU infernence requires updating the device=cpu line in the udfs. 


* Python dependencies
  * evadb 0.3.9
  * duckdb 1.4.1
  * xgboost 3.0.5
  * pandas 1.3.5
  * numpy 2.1.2
  * pyspark 4.0.1
* Download the following files and unzip to ./silver
  * https://huggingface.co/datasets/starfuryz2040/shiba/blob/main/alt_docs_x5_no_dupe.parquet.tar.xz
  * https://huggingface.co/datasets/starfuryz2040/shiba/blob/main/idnet_no_dupe.parquet.tar.xz
  * https://huggingface.co/datasets/starfuryz2040/shiba/blob/main/idnet10k_no_dupe.parquet.tar.xz
  * https://huggingface.co/datasets/starfuryz2040/shiba/blob/main/idnet100k_no_dupe.parquet.tar.xz

### Installing Shiba

* Copy/pull this entire project as is to target machine
* Create 3 folders sub-bulleted below in root(/) of the system. give rw(777) permission to all.
  * runlogs
    * leave empty. SHIBA logs will flow here.
  * models
    * https://huggingface.co/datasets/starfuryz2040/shiba/blob/main/models.tar.xz
    * unzip to this folder AND to the project folder (where this readme lives)
  * images (needed for clickhouse only)
    * https://huggingface.co/datasets/starfuryz2040/shiba/blob/main/images.tar.xz
* use the data_loader scripts in ./data_loaders to install all data needed in each system
  * update user/pass as needed
  * for all systems, use the _no_dupe version of the data loaders
* clickhouse needs two folders created as well. 
  * var/lib/clickhouse/user_scripts
    * copy all py files in our clickhouse_udfs folder there and mirror folder access to parent
    * may need to be set to runnable in permissions
  * /etc/clickhouse-server
    * copy all xml files from clickhouse_udfs to that folder
* Postgres needs special loading for the models
  * go to ./postgres_udfs
  * copy the contents of each txt file into a console connected to the postgres instance and run
* Evadb needs special loading of the models
  * go to ./evadb_udfs
  * copy every py file to the ./evadb_data/functions folder
  * run the evadb_udfs.ipynb to load all the models to evadb


### Executing program

* find the run_*.py files in the autobenchmark folder
  * the runall_naive.py will run all dbs with the naive udf
  * the "optimized" variants will only run optimized version of one db at a time to avoid settings leakage
* results appear in /runlogs
* if you want to use our analysis script, use ./analysis/run_log_analysis.ipynb
* results for the vm and hw based runs are in the same folder as csv files



