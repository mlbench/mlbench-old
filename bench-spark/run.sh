#!/usr/bin/env bash

BENCHMARK_DIR=/root/benchmark

read -r -d '' USAGE << EOS
Usage: mlbench DATASET METHOD PARTITIONS MIN_NUMBER_OF_ITERATIONS STEP MAX_NUMBER_OF_ITERATIONS [SPARK_OPTIONS]

  DATASET is the dataset directory (e.g., /data/.libsvm).
  METHOD "Regression" or "Classification".
  PARTITIONS number of partitions.
  MIN_NUMBER_OF_ITERATIONS minimum number of iterations (if you only want to run the benchmark once with the set parameters the value is not important)
  MAX_NUMBER_OF_ITERATIONS minimum number of iterations (if you only want to run the benchmark once with the set parameters the value is not important)
  STEP the steps for number of iterations is going to run (if you only want to run the benchmark once with the set parameter, set it to 0)
  SPARK_OPTIONS are passed on to spark-submit.
EOS

if [[ $# -lt 6 ]]; then
  echo "$USAGE"
  exit 1
fi
 
DATASET=$1
shift
METHOD=$1
shift
PARTITIONS=$1
shift
MIN_NUMBER_OF_ITERATIONS=$1
shift
STEP=$1
shift
MAX_NUMBER_OF_ITERATIONS=$1
shift

mkdir /data/results
#/bin/bash
echo "********************************************************************************************************Dividing the dataset************************************************************"
${SPARK_HOME}/bin/spark-submit --class "PrepareData" "$@" ${BENCHMARK_DIR}/target/scala-2.11/bench-spark-assembly-1.0.jar \
       -d "/data/$DATASET" -w "/data/results/" -m $METHOD -p $PARTITIONS 
echo "********************************************************************************************************Running the benchmark***********************************************************"
if [ $STEP -eq 0 ]; then
	if [ $METHOD = "Regression" ]; then
		${SPARK_HOME}/bin/spark-submit --class "RUN" "$@" ${BENCHMARK_DIR}/target/scala-2.11/bench-spark-assembly-1.0.jar \
        		-w "/data/results/" L1_Lasso_GD L1_Lasso_SGD Mllib_Lasso_SGD Mllib_Lasso_GD
	else
		${SPARK_HOME}/bin/spark-submit --class "RUN" "$@" ${BENCHMARK_DIR}/target/scala-2.11/bench-spark-assembly-1.0.jar \
			-w "/data/results/" L2_SVM_SGD L2_SVM_GD L2_LR_SGD L2_LR_GD L1_LR_SGD L1_LR_GD Mllib_L1_LR_SGD Mllib_L2_LR_SGD Mllib_L2_SVM_SGD Mllib_L2_SVM_GD 
	fi
	echo "********************************************************************************************************Running the evaluation***********************************************************"
	${SPARK_HOME}/bin/spark-submit --class "Evaluate" "$@" ${BENCHMARK_DIR}/target/scala-2.11/bench-spark-assembly-1.0.jar -d "/data/results/"
else
	for i in $(eval echo "{$MIN_NUMBER_OF_ITERATIONS..$MAX_NUMBER_OF_ITERATIONS..$STEP}")
		do
			echo "****************************************************iterations: "$i" *******************************************************************************************************"
			sed -i -e"s/iterations:.*/iterations: $i/" /data/parameters.yaml
			if [ $METHOD = "Regression" ]; then
		                ${SPARK_HOME}/bin/spark-submit --class "RUN" "$@" ${BENCHMARK_DIR}/target/scala-2.11/bench-spark-assembly-1.0.jar \
                		        -w "/data/results/" L1_Lasso_GD L1_Lasso_SGD Mllib_Lasso_SGD Mllib_Lasso_GD
        		else
                		${SPARK_HOME}/bin/spark-submit --class "RUN" "$@" ${BENCHMARK_DIR}/target/scala-2.11/bench-spark-assembly-1.0.jar \
                        		-w "/data/results/" L2_SVM_SGD L2_SVM_GD L2_LR_SGD L2_LR_GD L1_LR_SGD L1_LR_GD Mllib_L1_LR_SGD Mllib_L2_LR_SGD Mllib_L2_SVM_SGD Mllib_L2_SVM_GD
        		fi
			echo "**************************************************************************************************Running the evaluation***********************************************************"
        		${SPARK_HOME}/bin/spark-submit --class "Evaluate" "$@" ${BENCHMARK_DIR}/target/scala-2.11/bench-spark-assembly-1.0.jar -d "/data/results/"

	
		done

fi
 
bash
