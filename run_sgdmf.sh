#!/bin/bash
data_dir="/proj/BigLearning/jinlianw/ml-latest"
rm -rf ${data_dir}/ratings_p.csv"."*
/users/jinlianw/spark-1.6.0/bin/spark-submit \
    --master spark://h0.jwhadoopspark2n5.biglearning.nome.nx:7077 \
    --deploy-mode client \
    --num-executors 48 \
    --executor-memory 2G \
    --executor-cores 1 \
    --driver-memory 1G \
    sgdmf.py \
    file://${data_dir}/ratings_p.csv \
    500 \
    48 \
    1
