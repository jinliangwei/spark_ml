#!/usr/bin/env bash

numexecutor="4"

DIR="$(cd "$(dirname "$0")" && pwd)"

scriptname=`basename "$0"`

/users/haoranw2/Tools/ml_hrw/scripts/init_env/init.sh $project $num
data_dir="/proj/BigLearning/haoranw2/ml-20m"
data_file="ratings_small.csv"

#jar_file="/users/jinkyuk/gittedproject/ml-single-scala/newsgdmf/target/scala-2.10/sgdmf_2.10-1.0.jar"
jar_file=$DIR/target/scala-2.10/sgdmf_2.10-1.0.jar
app_params="200 4 2"

echo "Start"
/users/haoranw2/Tools/spark-my-fork/bin/spark-submit \
    --master local[1] \
    --deploy-mode client \
    --executor-memory 2G \
    --executor-cores 1 \
    --driver-memory 1G \
    --num-executors ${numexecutor} \
    ${jar_file} \
    ${data_dir}/$data_file \
    ${app_params}

echo "Done"
