# Distributed Machine Learning Benchmark, Apace Spark parts


## Running the Benchmark
###### Clone the repository: 
```bash
git clone https://github.com/mlbench/mlbench
```
###### Add submodule projects:
```bash
git submodule init
git submodule update
```

## How to run
**Spark Benchmark and Reference Implementations**: 
First create a fat JAR of the project with all of its dependencies by running the following command in /bench-spark folder:
```bash
sbt assembly
```
It is important to first create a "working directory" where all results will be gathered. The default working directory is ../results relative to your project folder.
First step is to prepare the data, second is to run desired optimization algorithms on data and finally evaluate the performance and compare them.
To prepare the dataset use "PrepareData" which accepts the following arguments:
```bash
-d, --dataset  <arg>      absolute address of the libsvm dataset. This must be
                            provided.
  -w, --dir  <arg>          working directory where results are stored. Default is
                            "../results".
  -m, --method  <arg>       Method can be either "Regression" or "Classification".
                            This must be provided
  -p, --partitions  <arg>   Number of spark partitions to be used. Optional.
      --help                Show help message
```
This command will split the original data to train and test data and will save them in working directory for later use.
To run different optimization algorithms use "RUN" which needs the following arguments:
```bash
-w, --dir  <arg>   working directory where resultsare stored. Default is
                     "../results".
      --help         Show help message

 trailing arguments:
  optimizers (required)   List of optimizers to be used. At least one is required
```
The trailing arguments must be one of the tasks mentioned below in [Tasks Specifications](#task-specifications). An output file "res.out" will be saved in the working directory containing weight vectors of each different optimization method.
Finally to get the corresposing objective values run the script "Evaluate":
```bash
 -d, --dir  <arg>   absolute address of the working directory. This must be
                     provided.
      --help         Show help message
```

###### Example: 
Go to /bench-spark folder and run:
```bash
mkdir ../results

$your-spark-folder/bin/spark-submit --class "PrepareData" 
target/scala-2.10/bench-spark-assembly-1.0.jar 
-d "/path/to/your/dataset/breast-cancer_scale.libsvm" 
-m Regression

$your-spark-folder/bin/spark-submit --class "RUN" 
target/scala-2.10/bench-spark-assembly-1.0.jar 
L1_Lasso_GD L1_Lasso_SGD L1_Lasso_ProxCocoa Elastic_ProxCocoa

$your-spark-folder/bin/spark-submit --class "Evaluate" 
target/scala-2.10/bench-spark-assembly-1.0.jar
```

The output res.out:
```bash
L1_Lasso_GD: DenseVector(-2.672158927525372, 0.2835398557585042, 0.2616505245785392, 0.1062669616202757, 0.0010041643706443565, -3.247334399853071E-4, 0.414111116641944, 0.004824409829823558, 0.00742871210731205, -0.7643289526295685) elapsed: 1281ms lambda: 0.1
L1_Lasso_SGD: DenseVector(-2.112996906376575, 0.10904730089760592, 0.0789677010341028, 7.846417009157897E-4, 0.007214770187810594, -0.002870040147401511, 0.45213296371762246, -4.0625328938762354E-4, 0.007843171211156274, -0.9378429139210855) elapsed: 621ms lambda: 0.1
L1_Lasso_ProxCocoa: DenseVector(-3.0703759321596684, 0.2063079519043623, 0.328673676903811, 0.027289089761645777, 0.0, 0.0, 0.41197113667299534, 0.0, 0.0, -0.3722738846697741) elapsed: 1683ms lambda: 0.1
Elastic_ProxCocoa: DenseVector(-2.1647682768571683, 0.35959693942313214, 0.16053636571576063, 0.07100394273617437, 0.0, 0.0, 0.32159948533563326, 0.0, 0.0, -0.8746298720599254) elapsed: 1436ms lambda: 0.1 alpha: 0.5
```

# Benchmark Task Specifications:
Here we list the task descriptions. We start with the algorithms and their reference implementations.
TODO: here we'll soon add the benchmark dimensions, together with the specific datasets chosen.

## Tasks D.. (Dummy tasks for measuring raw system performance)
### Task D-S (Dummy communication rounds)
i.e. reduce all operation, synchronous
(this task is to eliminate obvious issues with e.g. flink, spark), 
varying dimensionality, i.e. number of features d=10^{0,2,3,6,9,...}

## Tasks L2.. (L2-Regularized Linear Models)
### Task L2-SVM
L2-Regularized SVM
benchmarks:
- [L2-SVM-GD](https://github.com/mlbench/mlbench/tree/master/bench-spark/src/main/scala/Classification.scala#L55)
- [L2-SVM-SGD](https://github.com/mlbench/mlbench/tree/master/bench-spark/src/main/scala/Classification.scala#L48)
- L2-SVM-QN (quasi-newton, L-BFGS)
- [L2-SVM-COCOA] (https://github.com/mlbench/mlbench/tree/master/bench-spark/src/main/scala/Classification.scala#L90)

### Task L2-LR (Logistic Regression)
L2-Regularized Logistic Regression
benchmarks:
- [L2-LR-GD](https://github.com/mlbench/mlbench/tree/master/bench-spark/src/main/scala/Classification.scala#L69)
- [L2-LR-SGD](https://github.com/mlbench/mlbench/tree/master/bench-spark/src/main/scala/Classification.scala#L62)
- L2-LR-QN (quasi-newton, L-BFGS)

## Tasks L1.. (L1-Regularized Linear Models)
### Task L1-Lasso
Lasso
benchmarks:
- [L1-Lasso-GD (proximal GD)](https://github.com/mlbench/mlbench/tree/master/bench-spark/src/main/scala/Regression.scala#L48)
- [L1-Lasso-SGD (proximal SGD)](https://github.com/mlbench/mlbench/tree/master/bench-spark/src/main/scala/Regression.scala#L41)
- L1-Lasso-QN (quasi-newton, OWL-QN)
- [L1-Lasso-proxCOCOA] (https://github.com/mlbench/mlbench/blob/71191ba1f04e362fde6fcb247b1629265dc2d5af/bench-spark/src/main/scala/Regression.scala#L62)

### Task L1-LR
Logistic Regression (L1 reg)
benchmarks:
- [L1-LR-GD (proximal GD)](https://github.com/mlbench/mlbench/tree/master/bench-spark/src/main/scala/Classification.scala#L84)
- [L1-LR-SGD (proximal SGD)](https://github.com/mlbench/mlbench/tree/master/bench-spark/src/main/scala/Classification.scala#L77)
- L1-LR-QN (quasi-newton, L-BFGS)

## Tasks Elastic-Net
### Task L1-L2 Least-Squares
- [Elastic_ProxCOCOA] (https://github.com/mlbench/mlbench/tree/master/bench-spark/src/main/scala/Regression.scala#L55)

## Future Benchmark Task Ideas
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
- dummy small ones for testing: e.g. ionosphere
- medium size: e.g. epsilon, webspam
- large: e.g. criteo 
