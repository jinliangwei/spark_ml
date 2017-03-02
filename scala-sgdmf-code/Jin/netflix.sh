#!/usr/bin/env bash

# change application cod to persist MEMORY_AND_DISK_SER, not DISK_ONLY

numexecutor="8"
ememory="28g"
numparts=32


master_url="spark://h5.spark9.biglearning.nome.nx:7077"

# inputdata_path rank dsgd_block_number
app_program="/users/jinkyuk/gittedproject/ml-single-scala/newsgdmf/target/scala-2.10/sgdmf_2.10-1.0.jar"

app_params="/l0/netflix-wholeshuffle/shuffle_whole.dat 500 ${numparts} /l0/rdd/test3 50"

spark_submit_path="/users/jinkyuk/buildspark/sparkhadoop/spark-1.6.1/bin/spark-submit"

cmd0=" $spark_submit_path \
    --master ${master_url} \
    --conf spark.local.dir=/l0/rdd \
    --conf spark.eventLog.enabled=true \
    --conf spark.eventLog.dir=/l0/log/spark \
    --executor-memory ${ememory} \
    --total-executor-cores ${numexecutor} \
    --num-executors ${numexecutor} \
    --executor-cores 1 \
      ${app_program} ${app_params} "

echo "@@@@@@@@@@@@@@@@@@@@ Netflix 2mach  - ${numexecutor} executors, ${ememory} mem per executor" 

eval $cmd0  # Use this to run locally (on one machine)
