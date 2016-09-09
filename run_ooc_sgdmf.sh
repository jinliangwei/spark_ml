#!/bin/bash
data_dir="/proj/BigLearning/jinlianw/ml-10M100K"
data=${data_dir}/ratings.csv

./ooc_sgd_mf.py ${data} 100 1 tmp
