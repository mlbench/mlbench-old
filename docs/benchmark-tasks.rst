======================
Benchmarking Divisions
======================

There are two divisions of benchmarking, the closed one which is restrictive to allow fair comparisons of hardware and implementation,
and the open divisions, which allows users to run their own models and code while still providing a reasonably fair comparison.


Closed Division
---------------

The Closed Division encompasses several subcategories to compare different dimensions of distributed machine learning algorithms. We provide precise reference implementations of each algorithm, including the precise communication patterns, such that they can be implemented strictly comparable between different hardware and software frameworks.

The two basic metrics for comparison are `Accuracy after Time` and `Time to Accuracy` (where accuracy will be test and/or training accuracy)

Variable dimensions in this category include:

- Algorithm
  - limited number of prescribed standard algorithms, according to strict reference implementations provided
- Hardware
  - GPU
  - CPU(s)
  - Memory
- Scalability
  - Number of workers
- Network
  - Impact of bandwidth and latency

Accuracy after Time
~~~~~~~~~~~~~~~~~~~

The system has a certain amount ot time for training (2 hours) and at the end, the accuracy of the final model is evaluated.
The higher the better

Time to Accuracy
~~~~~~~~~~~~~~~~
A certain Accuracy, e.g. 97% is defined for a task and the training time of the system until that accuracy is reached is measured.
The shorter the better.

.. _Deep Residual Learning for Image Recognition:
    https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf

.. _CIFAR-10:
    http://www.cs.toronto.edu/~kriz/cifar.html

Here is a validation error plot of training ResNet on `CIFAR-10`_ using settings in `Deep Residual Learning for Image Recognition`_.

.. image:: images/km2016deep.png
    :align: center


Open Division
-------------
The Open Division allows you to implement your own algorithms and training tricks and compare them to other implementations. There's little limit to what can be changed by you and as such, it is up to you to make sure that comparisons are fair.

In this division, mlbench merely provides a platform to easily perform and measure distributed machine learning experiments in a standardized way.


**********************
Machine Learning Tasks
**********************

One of our goals in MLBench is to cover different machine learning tasks
that are important both in the industry and in the academic community.
We provide precisely defined tasks and datasets to have a fair and
precise comparison of all algorithms, frameworks, and hardware.
Here is a list of our current ML tasks and some information about
their algorithms, datasets, and results for different settings.

1. Image Classification
####################

Image Classification is one of the most important and common problems
in computer vision. There are several powerful algorithms and
approaches exist to solve this problem.
Here, we provide the reference implementations for a few of
the state-of-the-art models in this area.

Models
~~~~~~

#. **Deep Residual Networks (ResNets)**
    A deep residual network is a kind of deep neural networks
    which enables training networks with large numbers of layers,
    without the need of having too many parameters,
    while improving performance.
    For more information on ResNets please refer to `this paper <https://arxiv.org/abs/1512.03385>`_.


Datasets
~~~~~~~~

#. **CIFAR-10**
    The `CIFAR-10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_
    dataset containing a set of images used to train machine learning
    and computer vision models.
    It contains 60,000 32x32 color images in 10 different classes,
    with 6000 images per class. The 10 different classes represent
    airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.


****************
Benchmark Tasks
****************
Here is some information about the current ML algorithms
implemented in MLBench.

1. Image Classification (ResNets)
###############################

We provide two implemented versions of ResNets based on prior work by He et al.
The first version (v1) is based on the ResNets defined in
`this paper <https://arxiv.org/abs/1512.03385>`_.
The second version (v2) is based on the ResNets defined `here
<https://arxiv.org/abs/1603.05027>`_.
For each version we have the network implementations
with 20, 32, 44, and 56 layers.

Optimizer
~~~~~~~~~

We use synchronous SGD as the optimizer,
and we have a learning rate scheduler that changes the learning rate
over time.

Dataset
~~~~~~~

We use CIFAR-10 as the dataset for this task.
The training and test data are selected as the dataset provides.

Data preprocessing
~~~~~~~~~~~~~~~~~~~

We followed the same approach as mentioned `here <https://arxiv.org/abs/1512.03385>`_.

Framework & Hardware
~~~~~~~~~~~~~~~~~~~~
We use PyTorch to implement the algorithms.
Hardware specification comes here!

Results
~~~~~~~~
Results come here!

