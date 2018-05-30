#!/bin/bash


spark-submit --class "PrepareData" target/scala-2.10/bench-spark-assembly-1.0.jar -d ../datasets/breast-cancer_scale.libsvm -m Regression

spark-submit --class "RUN" target/scala-2.10/bench-spark-assembly-1.0.jar L1_Lasso_SGD Mllib_Lasso_SGD L1_Lasso_ProxCocoa Elastic_ProxCocoa

spark-submit --class "Evaluate" target/scala-2.10/bench-spark-assembly-1.0.jar

