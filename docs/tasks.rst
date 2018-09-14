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

