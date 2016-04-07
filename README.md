# Distributed Machine Learning Benchmark
A public and reproducible comparison of distributed machine learning systems. Benchmark for large scale classification and regression running on different software frameworks.

Here is the [design doc](https://docs.google.com/document/d/1jM4zXRDezEJmIKwoDOKNlGvuNNJk5_FxcBrn1mfYp0E/edit#)

Towards a more precise definition of the benchmark task

# Task 0: dummy communication rounds,
i.e. reduce all operation
(this task is to eliminate obvious issues with e.g. flink, spark), 
varying d=10^{0,2,3,6,9,...}

# Task 1: L2-Regularized Linear Models
## Task 1a: SVM
## Task 1b: LR
Logistic Regression (L2 reg)
# Task 2: L1-Regularized Linear Models
## Task 2a: Lasso
## Task 2b: LR
Logistic Regression (L1 reg)
