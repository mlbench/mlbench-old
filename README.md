# Distributed Machine Learning Benchmark
A public and reproducible comparison of distributed machine learning systems. Benchmark for large scale classification and regression running on different software frameworks.

Here is the [design doc](https://docs.google.com/document/d/1jM4zXRDezEJmIKwoDOKNlGvuNNJk5_FxcBrn1mfYp0E/edit#)

Towards a more precise definition of the benchmark task, work in progress

# Task specifications:

## Tasks D.. (Dummy tasks for measuring raw system performance)
### Task D-S (Dummy communication rounds)
i.e. reduce all operation, synchronous
(this task is to eliminate obvious issues with e.g. flink, spark), 
varying d=10^{0,2,3,6,9,...}

## Tasks L2.. (L2-Regularized Linear Models)
### Task L2-SVM
L2-Regularized SVM
benchmarks:
- L2-SVM-GD
- L2-SVM-SGD
- L2-SVM-QN (quasi-newton, L-BFGS)
- L2-SVM-COCOA

### Task L2-LR (Logistic Regression)
L2-Regularized Logistic Regression
benchmarks:
- L2-LR-GD
- L2-LR-SGD
- L2-LR-QN (quasi-newton, L-BFGS)

## Tasks L1.. (L1-Regularized Linear Models)
### Task L1-Lasso
Lasso
benchmarks:
- L1-Lasso-GD (proximal GD)
- L1-Lasso-SGD (proximal SGD)
- L1-Lasso-QN (quasi-newton, OWL-QN)
- L1-Lasso-proxCOCOA

### Task L1-LR
Logistic Regression (L1 reg)
benchmarks:
- L2-LR-GD (proximal GD)
- L2-LR-SGD (proximal SGD)
- L2-LR-QN (quasi-newton, L-BFGS)


## Future Task Ideas
- D-A: asynchronous communication rounds
- Ridge Regression (would include linear systems solvers)
- Elastic Net Regularizers (L1+L2, used for Least-Squares and LR)

## Algorithm Descriptions
- GD (Gradient Descent)
this is the batch-variant, i.e. using all datapoints per iteration
- SGD
TODO:
 - describe the updates mathematically
 - add links to reference implementations
 - specify stepsizes and all potential other algo parameters for each dataset (see below)

## List of Datasets
### Classification:
- DC1: ionosphere
- DC2: rcv
- DC3: epsilon
- DC4: criteo

### Regression:
- DR1: ionosphere
- DR2: rcv
- DR3: epsilon
- DR4: criteo
