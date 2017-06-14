#!/usr/bin/env bash

BENCHMARK_DIR=/root/benchmark

read -r -d '' USAGE << EOS
Usage: mlbench DATASET METHOD PARTITIONS [SPARK_OPTIONS]

  DATASET is the dataset directory (e.g., /data/.libsvm).
  METHOD "Regression" or "Classification".
  PARTITIONS number of partitions.
  SPARK_OPTIONS are passed on to spark-submit.
EOS

if [[ $# -lt 3 ]]; then
  echo "$USAGE"
  exit 1
fi
 
DATASET=$1
shift
METHOD=$1
shift
PARTITIONS=$1
shift

#/bin/bash
echo "********************************************************************************************************Dividing the dataset************************************************************"
${SPARK_HOME}/bin/spark-submit --class "PrepareData" "$@" ${BENCHMARK_DIR}/target/scala-2.11/bench-spark-assembly-1.0.jar \
       -d $DATASET -w "/data/" -m $METHOD -p $PARTITIONS 
echo "********************************************************************************************************Running the benchmark***********************************************************"
if [ $METHOD = "Regression" ]; then
	${SPARK_HOME}/bin/spark-submit --class "RUN" "$@" ${BENCHMARK_DIR}/target/scala-2.11/bench-spark-assembly-1.0.jar \
        	-w "/data/" L1_Lasso_GD L1_Lasso_SGD Mllib_Lasso_SGD Mllib_Lasso_GD
else
	${SPARK_HOME}/bin/spark-submit --class "RUN" "$@" ${BENCHMARK_DIR}/target/scala-2.11/bench-spark-assembly-1.0.jar \
		-w "/data/" L2_SVM_SGD L2_SVM_GD L2_LR_SGD L2_LR_GD L1_LR_SGD L1_LR_GD Mllib_L1_LR_SGD Mllib_L2_LR_SGD Mllib_L2_SVM_SGD Mllib_L2_SVM_GD 
fi
echo "********************************************************************************************************Running the evaluation***********************************************************"
${SPARK_HOME}/bin/spark-submit --class "Evaluate" "$@" ${BENCHMARK_DIR}/target/scala-2.11/bench-spark-assembly-1.0.jar -d "/data/"
 
bash
