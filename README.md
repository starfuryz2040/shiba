# SHIBA: Sql Hybrid Inference Benchmark Analytics

Benchmark framework to analyze and compare in sql inference of hybrid/complex data and models. 

## Description

We developed a framework to compare multiple popular SQL
database and analytic processors in sql inference query execution. Databases were populate with a subset
of the IDENET dataset. This multimodal and complex data allows for realistic queries and models as a basis of the 
comparison. This code base as of now isn't production ready. If interest persists, this will be developed further.

## Getting Started

Dependencies that are relatively straight forward. They must be installed both in your venv as well as system level (sudo pip). 
postgres and clickhouse do not have documented ways to use their own venv (though some workaround exist). 

The development environment was pycharm and dataspell. Pycharm support both py and ipynb files. Dataspell connects to 
almost every database, which is handy for troubleshooting.

### Dependencies

please install the following
* OS - Ubuntu 22.04.5 LTS
* Python 3.10
* openjdk-21-jdk (with env vars set for home and path)
* Python dependencies
  * evadb 0.3.9
  * duckdb 1.4.0
  * xgboost 3.0.5
  * torch 2.10
  * pandas 2.3.2
  * numpy 2.1.2
  * pyspark 4.1.1
* postgres 14
* postgres plpython3-14
* clickhouse 25
* Download the following files and unzip to ./silver
  * https://huggingface.co/datasets/starfuryz2040/shiba/blob/main/alt_docs_x5_no_dupe.parquet.tar.xz
  * https://huggingface.co/datasets/starfuryz2040/shiba/blob/main/idnet_no_dupe.parquet.tar.xz
  * https://huggingface.co/datasets/starfuryz2040/shiba/blob/main/idnet10k_no_dupe.parquet.tar.xz
  * https://huggingface.co/datasets/starfuryz2040/shiba/blob/main/idnet100k_no_dupe.parquet.tar.xz

### Installing

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
* use the data_loader scripts to install all data needed in each system
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



