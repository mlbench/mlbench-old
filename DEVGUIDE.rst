=======
Developer Guide
=======

Development Workflow
--------------------
`Git Flow <https://github.com/nvie/gitflow>`_ is used for features etc. This automatically handles pull requests.
Make sure to install the commandline tool at the link above



Code Style
----------
Python code should follow PEP8 guidelines. flake8 checks PEP8 compliance

Installation
------------
- Install docker and kubernetes binaries on your system

- Setup kubernetes dind (local multinode cluster): https://github.com/kubernetes-sigs/kubeadm-dind-cluster

- Install a private docker repository:

  - https://blog.cloudhelix.io/using-a-private-docker-registry-with-kubernetes-f8d5f6b8f646
        --> YAML is wrong on that link!!! Use one from https://docs.docker.com/registry/deploying/#deploy-your-registry-using-a-compose-file

  - Get secret: https://kubernetes.io/docs/tasks/configure-pod-container/pull-image-private-registry/

  - Add secret to default account (automatic login so we don't need to specify credentials in helm/kubernetes files): https://kubernetes.io/docs/tasks/configure-pod-container/configure-service-account/#add-imagepullsecrets-to-a-service-account

- run dind-proxy.sh to setup portforward for DIND cluster to local repository

- Install helm: https://github.com/kubernetes/helm/blob/master/docs/install.md

- Build & Publish docker to registry (repeat if code changes):

  - ./publish_docker.sh master

  - ./publish_docker.sh worker

- Deploy app to cluster with:

  - First time: helm install charts/mlbench

  - Redeploy: helm upgrade <release> charts/mlbench
    with release being something like "jumpy-rodent" (check deployment name)

  Access server at (replace <release>): http://localhost:8080/api/v1/namespaces/default/services/<release>-mlbench-master:http/proxy/main/


TODO: Use ksync to sync code changes directly. Doesn't work for me currently, filed issue @ https://github.com/vapor-ware/ksync/issues/212