# SHIBA: Sql Hybrid Inference Benchmark Analytics

Benchmark framework to analyze and compare in sql inference of hybrid/complex data and models. 

## Description

We developed a framework to compare multiple popular SQL
database and analytic processors in sql inference query execution. Databases were populate with a subset
of the IDENET dataset. This multimodal and complex data allows for realistic queries and models as a basis of the 
comparison. This code base as of now isn't production ready. If interest persists, this will be developed further.

## Getting Started

DuckDb, Evadb, and Pysark Python dependencies will need to be installed in the venv the rest of the 
code base is using. Postgres python dependencies will need to be installed at the system level. Clickhouse dependencies
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
* https://pytorch.org/ for pytorch installation. version 2.9.1 is the target. 


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

* After installing the above, create 3 folders in root. give rw permission to all.
  * runlogs
  * models
    * https://huggingface.co/datasets/starfuryz2040/shiba/blob/main/models.tar.xz
  * images (needed for clickhouse only)
    * https://huggingface.co/datasets/starfuryz2040/shiba/blob/main/images.tar.xz
* clickhouse needs two folders created as well. 
  * var/lib/clickhouse/user_scripts
    * copy all py files in our clickhouse_udfs folder there and mirror folder access to parent
    * may need to be set to runnable in permissions
  * /etc/clickhouse-server
    * copy all xml files from clickhouse_udfs to that folder
* copy this entire project as is to target machine
* use the data_loader scripts in ./data_loaders to install all data needed in each system
  * update user/pass as needed
  * for all systems, use the _no_dupe version of the data loaders
  * evadb should use the alt_loaders. If the smaller tables dont have them, copy and update the bigger table py files
    * evadb only supports CSV files which i will provide, but the 100k table loading method is strictly superior
    * use the partials of the old tables to create the link in eva. then just use direct to postgres loading. 
* runall.py in ./autobenchmark folder does exactly as it says
  * each .py in ./autobenchmark will need its user/pass/connection string updates for your systems


### Executing program

* find the runall.py, execute it
* results appear in /runlogs
* if you want to use our analysis script, use ./analysis/run_log_analysis.ipynb
* results for the vm and hw based runs are in the same folder as csv files



