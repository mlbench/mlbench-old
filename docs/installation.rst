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

TODO:


Google Cloud
------------

How to install mlbench on the Google Kubernetes Engines (GKE)

Install `Google Cloud SDK <https://cloud.google.com/sdk/>`_

.. note::
   If you want to build the docker images yourself and host it in the GC registry, follow these steps:

   Authenticate with GC registry:

   .. code-block:: bash

      gcloud auth configure-docker

   Build docker images (Replace **<gcloud project name>** with the name of your project):

   .. code-block:: bash

      make publish-docker component=master docker_registry=gcr.io/<gcloud project name>
      make publish-docker component=worker docker_registry=gcr.io/<gcloud project name>

   Use the following settings for your `values.yaml` file when installing with helm:

   .. code-block:: yaml

      master:

        image:
          repository: gcr.io/mlbench-214014/mlbench_master
          tag: latest
          pullPolicy: Always


      worker:

        image:
          repository: gcr.io/mlbench-214014/mlbench_worker
          tag: latest
          pullPolicy: Always

`Create a Kubernetes Cluster on GKE <https://cloud.google.com/kubernetes-engine/docs/how-to/creating-a-cluster>`_. We will assume the cluster is called `mlbench` for the remainder of this section.

.. note::
   Google installs several pods on each node by default, limiting the available CPU. This can take up to 0.5 CPU cores per node. So make sure to provision VM's that have at least 1 more core than the amount of cores you want to use for you mlbench experiment.
   See `here <https://cloud.google.com/kubernetes-engine/docs/concepts/cluster-architecture#memory_cpu>`_ for further details on node limits.

Install the credentials for your cluster (use the correct zone for your cluster):

.. code-block:: bash

   gcloud container clusters get-credentials mlbench --zone us-central1-a

Grant cluster-admin rights to the service account mlbench is running under (in this case `default`):

.. code-block:: bash

   cat <<EOF | kubectl apply -f -
   apiVersion: rbac.authorization.k8s.io/v1beta1
   kind: ClusterRoleBinding
   metadata:
     name: default
   roleRef:
     apiGroup: rbac.authorization.k8s.io
     kind: ClusterRole
     name: cluster-admin
   subjects:
     - kind: ServiceAccount
       name: default
       namespace: kube-system
   EOF

Install `Helm <https://helm.sh/>`_ and initialize it:

.. code-block:: bash

   helm init


Finally, install mlbench (Assuming your custom values are in the file `values.yaml`). `rel` is the release name.

.. code-block:: bash

   helm upgrade --wait --recreate-pods -f values.yaml --timeout 900 --install rel charts/mlbench

To access mlbench, run these commands and open the URL that's returned:

.. code-block:: bash

   export NODE_PORT=$(kubectl get --namespace default -o jsonpath="{.spec.ports[0].nodePort}" services rel-mlbench-master)
   export NODE_IP=$(gcloud compute instances list|grep $(kubectl get nodes --namespace default -o jsonpath="{.items[0].status.addresses[0].address}") |awk '{print $5}')
   gcloud compute firewall-rules create --quiet mlbench --allow tcp:$NODE_PORT,tcp:$NODE_PORT
   echo http://$NODE_IP:$NODE_PORT

.. warning::
   The last command opens up a firewall rule to the google cloud. Make sure to delete the rule once it's not needed anymore:

   .. code-block:: bash

      gcloud compute firewall-rules delete --quiet mlbench


Minikube
--------

Installing mlbench to `minikube <https://github.com/kubernetes/minikube>`_.

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
    $ helm upgrade --wait --recreate-pods -f values.yaml --timeout 900 --install ${RELEASE_NAME} charts/mlbench

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
