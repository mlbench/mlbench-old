.. highlight:: shell

============
Installation
============


Stable release
--------------

Install `Helm <https://helm.sh/>`_

Set the following helm properties (via cmd or custom values.yaml):

.. code-block:: yaml

   limits:
     cpu:
     maximumWorkers:
     bandwidth:

- `limits.cpu` is the maximum number of CPUs (Cores) available on each worker node. Uses Kubernetes notation (`8` or `8000m` for 8 cpus/cores)
- `limits.maximumWorkers` is the maximum number of Nodes available to mlbench as workers.
- `limits.bandwidth` is the maximum network bandwidth available between workers, in mbit per second

Use helm to install the mlbench chart:

.. code-block:: bash

   $ helm install mlbench

From sources
------------

Build and deploy the `master` and `worker` docker images:

.. code-block:: bash

   $ make publish-docker component=master docker_registry=localhost:5000
   $ make publish-docker component=worker docker_registry=localhost:5000

Install `Helm <https://helm.sh/>`_

Set the following helm properties (via cmd or custom values.yaml):

.. code-block:: yaml

   limits:
     cpu:
     maximumWorkers:
     bandwidth:

- `limits.cpu` is the maximum number of CPUs (Cores) available on each worker node. Uses Kubernetes notation (`8` or `8000m` for 8 cpus/cores)
- `limits.maximumWorkers` is the maximum number of Nodes available to mlbench as workers.
- `limits.bandwidth` is the maximum network bandwidth available between workers, in mbit per second

Use helm to install the mlbench chart from the mlbench root folder:

.. code-block:: bash

   $ helm install charts/mlbench/
