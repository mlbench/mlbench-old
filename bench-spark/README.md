# Bench-Spark #

This benchmark uses Apache Spark and runs GD and SGD algorithms
in-memory on different datasets. The metrics of interest is the
time in seconds of training and the loss function objective.

Stochastic Gradient Descent (SGD) is an efficient approach for learning linear classifiers under convex loss functions such as (linear) Support Vector Machines and Logistic Regression.
Although SGD is an old machine learning method, it has been considered just recently in the context of large-scale learning. 

SGD has been successfully applied to different large-scale and sparse machine learning problems. It is necessary to take a deeper look at this algorithm. 


### Building the image
To build the Docker image, you should download or clone the bench-spark directory. Set your work directory to the bench-spark. After that, you need to build the Spark image. 

    $ docker build -t spark21 ./Docker-source/spark/2.1.0/


By this command, you build a spark image with the name of spark21 which is the base of the main docker image. To be able to run the benchmark, we need to build the main image by the following command:

    $ docker build -t mlbench .

### Datasets

You can download the dataset that you want in libsvm format. You must put it under the dataset directory. [Here][dataset-path] you can find a set of datasets.


### Running the Benchmark

The benchmark runs the SGD and GD algorithms with different optimizers and regularizers on Spark through the spark-submit script
distributed with Spark.
Before running the benchmark you can set the algorithm parameters in the ./dataset/parameters.yml file. 

The running command takes 6 arguments: the dataset to use for training and test, the method which is "Regression" or "Classification", the number of data partitions, and 3 arguments that show the number of iterations. Any remaining arguments are passed to spark-submit.

To run the benchmark, we should use the following command:

    $ docker run --rm -v ./dataset:/data/ mlbench \
            {NAME_OF_DATASET} {METHOD} {NUMBER_OF_PARTITIONS} {MINIMUM_NUMBER_OF_ITERATIONS} {STEP} {MAXIMUM_NUMBER_OF_ITERATIONS} [SPARK_ARGUMENTS]


This command mounts the ./dataset directory and saves the results there. Here is the list of parameters in details:

  - NAME\_OF\_DATASET: the name of the dataset file that you will run the benchmark on.
  - METHOD: "Regression" or "Classification", here is the list of optimizers that will run for each of them:
      - Regression: GD and SGD with Lasso optimizer and L1 regularizer implemented in Scala and the MLlib version.
      - Classification: GD and SGD with SVM or Logistic Regression optimizer and L1 or L2 regularizer implemented in Scala and the MLlib version.
  - NUMBER\_OF\_PARTITIONS: the number of data partitions which the dataset is partitioned into.
  - {MINIMUM\_NUMBER\_OF\_ITERATIONS} {STEP} {MAXIMUM\_NUMBER\_OF\_ITERATIONS}: In this benchmark, we support running the algorithm in different numbers of iterations to produce the results for drawing some graphs showing the loss function of the method versus time. If you do not want to use this option you can enter 0 0 0 for these three arguments. In this case, the benchmark is only run by the number of iterations you set in the ./dataset/parameters.yaml.

Here is an example of the running command:

   $ docker run --rm -v ./dataset:/data/ mlbench \
               gisette_scale Classification 4 10 10 100


### Tweaking the Benchmark

Any arguments after the two mandatory ones are passed to spark-submit and can
be used to tweak execution. For example, to ensure that Spark has enough memory
allocated to be able to execute the benchmark in-memory, supply it with
--driver-memory and --executor-memory arguments:

    $ docker run --rm -v ./dataset:/data/ mlbench \
        gisette_scale Classification 4 10 10 100 \
        --driver-memory 4g --executor-memory 4g


### Multi-node deployment

This section explains how to run the benchmark using multiple Spark workers
(each running in a Docker container) that can be spread across multiple nodes
in a cluster.

First, you need to have your ./dataset directory on every physical node where Spark workers will
be running. 

The next step is to start Spark master and Spark workers. They should all run within the same
Docker network, which we call spark-net here. The workers get access to the
datasets with -v ./dataset/:/data/.

    $ docker run -dP --net spark-net --hostname spark-master --name spark-master spark21 master
    $ docker run -dP --net spark-net -v ./dataset:/data --name spark-worker-01 spark21 worker \
        spark://spark-master:7077
    $ docker run -dP --net spark-net -v ./dataset:/data --name spark-worker-02 spark21 worker \
        spark://spark-master:7077
    $ ...

Finally, run the benchmark as the client to the Spark master:

    $ docker run --rm -v ./dataset:/data/ mlbench \
        gisette_scale Classification 4 10 10 100 \ 
        --master spark://spark-master:7077

[dataset-path]: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

