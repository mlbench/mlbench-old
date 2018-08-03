===============================================
mlbench: Distributed Machine Learning Benchmark
===============================================

.. image:: https://travis-ci.com/mlbench/mlbench.svg?branch=develop
    :target: https://travis-ci.com/mlbench/mlbench

.. image:: https://readthedocs.org/projects/mlbench/badge/?version=latest
        :target: https://mlbench.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




A public and reproducible collection of reference implementations and benchmark suite for distributed machine learning systems. Benchmark for large scale solvers, implemented on different software frameworks & systems.
**This is a work in progress and not usable so far**


* Free software: Apache Software License 2.0
* Documentation: https://mlbench.readthedocs.io.


Features
--------

* For reproducibility and simplicity, we currently focus on standard **supervised ML**, namely classification and regression solvers.
* We provide **reference implementations** for each algorithm, to make it easy to port to a new framework.
* Our goal is to benchmark all/most currently relevant **distributed execution frameworks**. We welcome contributions of new frameworks in the benchmark suite
* We provide **precisely defined tasks** and datasets to have a fair and precise comparison of all algorithms and frameworks.
* Independently of all solver implementations, we provide universal **evaluation code** allowing to compare the result metrics of different solvers and frameworks.
* Our benchmark code is easy to run on the **public cloud**.
* Here is an older [design doc](https://docs.google.com/document/d/1jM4zXRDezEJmIKwoDOKNlGvuNNJk5_FxcBrn1mfYp0E/edit#) for this project.

TODO
----

Everything

Community
---------

About us: See `Authors`_

Mailing list: https://groups.google.com/d/forum/mlbench

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

.. _Authors: AUTHORS.rst
