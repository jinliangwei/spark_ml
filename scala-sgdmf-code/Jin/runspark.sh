#!/usr/bin/env bash

numexecutor="30"

# inputdata_path rank dsgd_block_number
app_program="/users/jinkyuk/gittedproject/ml-single-scala/newsgdmf/target/scala-2.10/sgdmf_2.10-1.0.jar"

app_params="/users/jinkyuk/gittedproject/ml-single-scala/sgdmf/data/100k.csv 100 ${numexecutor} ~/disk1/rdd/test3 2"

master_url="spark://aa4.stradsplainaaa2.biglearning.nome.nx:7077" 

spark_submit_path="/users/jinkyuk/buildspark/sparkhadoop/spark-1.6.1/bin/spark-submit"

cmd0=" $spark_submit_path \
    --master ${master_url} \
    --conf spark.local.dir=/l0/rdd \
    --conf spark.eventLog.enabled=true \
    --conf spark.eventLog.dir=/var/log/spark \
    --executor-memory 2g \
    --total-executor-cores 30 \
    --num-executors 30 \
    --executor-cores 1 \
      ${app_program} ${app_params} "
echo "@@@@@@@@@@@@@@@@@@@@ small 100k rating " 

eval $cmd0  # Use this to run locally (on one machine)
