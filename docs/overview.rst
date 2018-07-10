.. figure:: images/DeploymentArchitecture.png
   :scale: 50 %
   :alt: Deployment Architecture Overview

   Deployment Overview: Relation between Experiment and Coordinator


mlbench consists of two components, the **Coordinator** and the **Experiment** Docker containers.

Coordinator
-----------
The coordinator contains the Dashboard, the  main interface for the project. The dashboard allows 
you to start and run a distributed ML experiment and visualizes the progress and result of the 
experiment. It allows management of the mlbench nodes in the kubernetes cluster and for most
users constitutes the sole way they interact with the mlbench project.

It also contains a decidated metrics API that is used for the nodes to report their state back
to the Coordinator.

The Coordinator can also be granted access to the Kubernetes API to manage cluster settings
relevant to the experiments.


Experiment
----------
The experiment image contains all the boilerplate code needed for a distributed ML model and
as well as the actual model code. It takes care of training the distributed model, depending
on the settings provided by the Coordinator. So the Coordinator informs the Experiment nodes
what experiment the user would like to run and they run the relevant code by themselves.

Experiment nodes send status information to the metrics API of the Coordinator to inform it
of the progress and state of the current run.