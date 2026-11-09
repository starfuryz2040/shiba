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
  * https://www.mediafire.com/file/5v90pk2ow7tyo1k/alt_docs_x5.parquet.7z/file
  * https://www.mediafire.com/file/zcipirm9y258vyd/labeled_training_set_no_tensor.parquet.7z/file
  * https://www.mediafire.com/file/mdvkblmmcdj5dj4/labeled_training_set_no_tensor_10k.parquet.7z/file

### Installing

* After installing the above, create 3 folders in root. give rw permission to all.
  * runlogs
  * models
    * copy mymodel and xgboost_model.model to this directory
    * used by postgres and clickhouse
  * images (needed for clickhouse only)
    * https://www.mediafire.com/file/m3d6yktd1rmo9wc/images.7z/file
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



