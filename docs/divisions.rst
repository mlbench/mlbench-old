======================
Benchmarking Divisions
======================

There are two divisions of benchmarking, the closed one which is restrictive to allow fair comparisons of hardware and implementation,
and the open divisions, which allows users to run their own models and code while still providing a reasonably fair comparison.


Closed Division
---------------

The Closed Division encompasses several subcategories to compare different dimensions of distributed machine learning.

The two basic metrics for comparison are `Accuracy after Time` and `Time to Accuracy`

Variable dimensions in this category include:

- Hardware
  - GPU
  - CPU(s)
  - Memory
- Scalability
  - Number of workers
- Network
  -Bandwidth

Accuracy after Time
~~~~~~~~~~~~~~~~~~~

The system has a certain amount ot time for training (2 hours) and at the end, the accuracy of the final model is evaluated.
The higher the better

Time to Accuracy
~~~~~~~~~~~~~~~~
A certain Accuracy, e.g. 97% is defined for a task and the training time of the system until that accuracy is reached is measured.
The shorter the better



Open Division
-------------
The Open Division allows you to implement your own algorithms and compare them to other implementations. There's little limit to what
can be changed by you and as such, it is up to you to make sure that comparisons are fair.

In this division, mlbench merely provides a platform to easily perform and measure distributed machine learning experiments in a
standardized way.