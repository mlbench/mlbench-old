.. highlight:: shell

Installation
============

Make sure to read :doc:`prerequisites` before installing mlbench.

All guides assume you have checked out the mlbench github repository and have a terminal open in the checked-out ``mlbench`` directory.

.. _helm-charts:

Helm Chart values
-----------------

Since every Kubernetes is different, there are no reasonable defaults for some values, so the following properties have to be set.
You can save them in a yaml file of your chosing. This guide will assume you saved them in `myvalues.yaml`.

.. code-block:: yaml

   limits:
     maximumWorkers:
     cpu:
     bandwidth:

- ``limits.maximumWorkers`` is the maximum number of Nodes available to mlbench as workers. This sets the maximum number of nodes that can be chosen for an experiment in the UI. By default mlbench starts 2 workers on startup.
- ``limits.cpu`` is the maximum number of CPUs (Cores) available on each worker node. Uses Kubernetes notation (`8` or `8000m` for 8 cpus/cores). This is also the maximum number of Cores that can be selected for an experiment in the UI
- ``limits.bandwidth`` is the maximum network bandwidth available between workers, in mbit per second. This is the default bandwidth used and the maximum number selectable in the UI.

.. Caution::
   If you set ``maximumWorkers`` or ``cpu`` higher than available in your cluster, Kubernetes will not be able to allocate nodes to mlbench and the deployment will hang indefinitely, without throwing an exception.
   Kubernetes will just wait until nodes that fit the requirements become available. So make sure your cluster actually has the requirements avilable that you requested.


Basic Install
-------------

Set the :ref:`helm-charts`

Use helm to install the mlbench chart (Replace ``${RELEASE_NAME}`` with a name of your choice):

.. code-block:: bash

   $ helm upgrade --wait --recreate-pods -f values.yaml --timeout 900 --install ${RELEASE_NAME} charts/mlbench

Follow the instructions at the end of the helm install to get the dashboard URL. E.g.:

.. code-block:: bash
   :emphasize-lines: 5,6,7

   $ helm upgrade --wait --recreate-pods -f values.yaml --timeout 900 --install rel charts/mlbench
     [...]
     NOTES:
     1. Get the application URL by running these commands:
        export NODE_PORT=$(kubectl get --namespace default -o jsonpath="{.spec.ports[0].nodePort}" services rel-mlbench-master)
        export NODE_IP=$(kubectl get nodes --namespace default -o jsonpath="{.items[0].status.addresses[0].address}")
        echo http://$NODE_IP:$NODE_PORT

This outputs the URL the Dashboard is accessible at.


Google Cloud / Google Kubernetes Engine (GKE)
---------------------------------------------

Set the :ref:`helm-charts`

.. important::
   Make sure to read the prerequisites for :ref:`google-cloud`

Please make sure that ``kubectl`` is configured `correctly <https://cloud.google.com/kubernetes-engine/docs/quickstart>`_.

.. caution::
   Google installs several pods on each node by default, limiting the available CPU. This can take up to 0.5 CPU cores per node. So make sure to provision VM's that have at least 1 more core than the amount of cores you want to use for you mlbench experiment.
   See `here <https://cloud.google.com/kubernetes-engine/docs/concepts/cluster-architecture#memory_cpu>`_ for further details on node limits.

Install mlbench (Replace ``${RELEASE_NAME}`` with a name of your choice):

.. code-block:: bash

   helm upgrade --wait --recreate-pods -f values.yaml --timeout 900 --install ${RELEASE_NAME} charts/mlbench

To access mlbench, run these commands and open the URL that' is returned (**Note**: The default instructions returned by `helm` on the commandline return the internal cluster ip only):

.. code-block:: bash

   export NODE_PORT=$(kubectl get --namespace default -o jsonpath="{.spec.ports[0].nodePort}" services ${RELEASE_NAME}-mlbench-master)
   export NODE_IP=$(gcloud compute instances list|grep $(kubectl get nodes --namespace default -o jsonpath="{.items[0].status.addresses[0].address}") |awk '{print $5}')
   gcloud compute firewall-rules create --quiet mlbench --allow tcp:$NODE_PORT,tcp:$NODE_PORT
   echo http://$NODE_IP:$NODE_PORT

.. danger::
   The last command opens up a firewall rule to the google cloud. Make sure to delete the rule once it's not needed anymore:

   .. code-block:: bash

      gcloud compute firewall-rules delete --quiet mlbench

.. hint::
   If you want to build the docker images yourself and host it in the GC registry, follow these steps:

   Authenticate with GC registry:

   .. code-block:: bash

      gcloud auth configure-docker

   Build docker images (Replace **<gcloud project name>** with the name of your project):

   .. code-block:: bash

      make publish-docker component=master docker_registry=gcr.io/<gcloud project name>
      make publish-docker component=worker docker_registry=gcr.io/<gcloud project name>

   Use the following settings for your `myvalues.yaml` file when installing with helm:

   .. code-block:: yaml

      master:

        image:
          repository: gcr.io/<gcloud project name>/mlbench_master
          tag: latest
          pullPolicy: Always


      worker:

        image:
          repository: gcr.io/<gcloud project name>/mlbench_worker
          tag: latest
          pullPolicy: Always


Minikube
--------

Installing mlbench to `minikube <https://github.com/kubernetes/minikube>`_.

Set the :ref:`helm-charts`

First build docker images and push them to private registry `localhost:5000`.

.. code-block:: bash

  $ make publish-docker component=master docker_registry=localhost:5000
  $ make publish-docker component=worker docker_registry=localhost:5000

Then start minikube cluster

.. code-block:: bash

    $ minikube start

Use `tcp-proxy <https://github.com/Tecnativa/docker-tcp-proxy>`_ to forward node's 5000 port to host's port 5000
so that one can pull images from local registry.

.. code-block:: bash

    $ minikube ssh
    $ docker run --name registry-proxy -d -e LISTEN=':5000' -e TALK="$(/sbin/ip route|awk '/default/ { print $3 }'):5000" -p 5000:5000 tecnativa/tcp-proxy

Now we can pull images from private registry inside the cluster, check :code:`docker pull localhost:5000/mlbench_master:latest`.

Next install or upgrade a helm chart with desired configurations with name `${RELEASE_NAME}`

.. code-block:: bash

    $ helm init --kube-context minikube --wait
    $ helm upgrade --wait --recreate-pods -f myvalues.yaml --timeout 900 --install ${RELEASE_NAME} charts/mlbench

.. note::
    The minikube runs a single-node Kubernetes cluster inside a VM. So we need to fix the :code:`replicaCount=1` in `values.yaml`.

Once the installation is finished, one can obtain the url

.. code-block:: bash

    export NODE_PORT=$(kubectl get --namespace default -o jsonpath="{.spec.ports[0].nodePort}" services ${RELEASE_NAME}-mlbench-master)
    export NODE_IP=$(kubectl get nodes --namespace default -o jsonpath="{.items[0].status.addresses[0].address}")
    echo http://$NODE_IP:$NODE_PORT

Now the mlbench dashboard should be available at :code:`http://${NODE_IP}:${NODE_PORT}`.

.. note::
  To access :code:`http://$NODE_IP:$NODE_PORT` outside minikube, run the following command on the host:

  .. code-block:: bash

      $ ssh -i ${MINIKUBE_HOME}/.minikube/machines/minikube/id_rsa -N -f -L localhost:${NODE_PORT}:${NODE_IP}:${NODE_PORT} docker@$(minikube ip)

  where :code:`$MINIKUBE_HOME` is by default :code:`$HOME`. One can view mlbench dashboard at :code:`http://localhost:${NODE_PORT}`
