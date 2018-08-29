=====
Usage
=====

Follow the instructions at the end of the helm install to get the dashboard URL. E.g.:

.. code-block:: bash

   $ helm install mlbench
     [...]
     NOTES:
     1. Get the application URL by running these commands:
        export NODE_PORT=$(kubectl get --namespace default -o jsonpath="{.spec.ports[0].nodePort}" services rel-mlbench-master)
        export NODE_IP=$(kubectl get nodes --namespace default -o jsonpath="{.items[0].status.addresses[0].address}")
        echo http://$NODE_IP:$NODE_PORT

This outputs the URL the Dashboard is accessible at.

The dashboard shows the status of all available worker nodes and allows you to run benchmark experiments or to start you own custom runs.



.. warning::
   Login to /admin/ on the dashboard with mlbench_admin:mlbench_password and change the password of the admin user for security reasons