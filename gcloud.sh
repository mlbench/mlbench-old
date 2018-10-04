#!/bin/bash

echo "=============================================================================="
echo "https://mlbench.github.io/2018/09/04/tutorial/"
echo ""
echo "Assume the project, zone, region has been set"
echo "    gcloud config set project [Project Name]"
echo "    gcloud config set container/cluster [Cluster Name]"
echo "    gcloud config set compute/zone [Zone]"
echo "    gcloud config set compute/region [region]"
echo "    gcloud container clusters get-credentials [cluster name] --zone [Zone] --project [project name]"
echo ""
echo "=============================================================================="

CLUSTER_NAME=${CLUSTER_NAME:-benchmark-settings}
MACHINE_TYPE=${MACHINE_TYPE:-n1-standard-4}
NUM_NODES=${NUM_NODES:-2}
NUM_GPU_PER_NODE=${NUM_GPU_PER_NODE:-1}
RELEASE_NAME=${RELEASE_NAME:-rel}
PREEMPTIBLE=${PREEMPTIBLE:-}
MYVALUES_FILE=${MYVALUES_FILE:-values.yaml}

CLUSTER_VERSION=1.10
INSTANCE_DISK_SIZE=50
MACHINE_ZONE=${MACHINE_ZONE:-europe-west1-b}
DISK_TYPE=pd-standard
GPU_TYPE=nvidia-tesla-k80

case $1 in
    create )
        # Create a CPU cluster
        gcloud container clusters create ${CLUSTER_NAME} \
            ${PREEMPTIBLE} \
            --zone ${MACHINE_ZONE} \
            --cluster-version ${CLUSTER_VERSION} \
            --enable-network-policy \
            --machine-type=${MACHINE_TYPE} \
            --num-nodes=${NUM_NODES} \
            --disk-type=${DISK_TYPE} \
            --disk-size=${INSTANCE_DISK_SIZE} \
            --scopes=storage-full

        # Get credential of the cluster
        gcloud container clusters get-credentials --zone ${MACHINE_ZONE} ${CLUSTER_NAME}

        # Create a tiller and add it to cluster-admin
        kubectl --namespace kube-system create sa tiller

        kubectl create clusterrolebinding tiller --clusterrole cluster-admin --serviceaccount=kube-system:tiller

        # Initialize helm to install charts
        helm init --wait --service-account tiller
        ;;

    create-gpu )
        # Create a GPU cluster
        gcloud container clusters create ${CLUSTER_NAME} \
            ${PREEMPTIBLE} \
            --zone ${MACHINE_ZONE} \
            --cluster-version ${CLUSTER_VERSION} \
            --enable-network-policy \
            --machine-type=${MACHINE_TYPE} \
            --num-nodes=${NUM_NODES} \
            --disk-type=${DISK_TYPE} \
            --disk-size=${INSTANCE_DISK_SIZE} \
            --accelerator type=${GPU_TYPE},count=${NUM_GPU_PER_NODE} \
            --scopes=storage-full

        # Get credential of the cluster
        gcloud container clusters get-credentials --zone ${MACHINE_ZONE} ${CLUSTER_NAME}

        # Install NVIDIA Driver for gpus
        kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/k8s-1.10/nvidia-driver-installer/cos/daemonset-preloaded.yaml

        # Create a tiller and add it to cluster-admin
        kubectl --namespace kube-system create sa tiller

        kubectl create clusterrolebinding tiller --clusterrole cluster-admin --serviceaccount=kube-system:tiller

        # Initialize helm to install charts
        helm init --wait --service-account tiller

        ;;

    install )
        echo "Installing helm charts..."
        echo MYVALUES_FILE=$MYVALUES_FILE

        helm upgrade --wait --recreate-pods -f ${MYVALUES_FILE} --timeout 900 --install ${RELEASE_NAME} charts/mlbench \
            --set limits.workers=${NUM_NODES} \
            --set worker.replicaCount=${NUM_NODES}

        export NODE_PORT=$(kubectl get --namespace default -o jsonpath="{.spec.ports[0].nodePort}" services ${RELEASE_NAME}-mlbench-master)
        export NODE_IP=$(gcloud compute instances list|grep $(kubectl get nodes --namespace default -o jsonpath="{.items[0].status.addresses[0].address}") |awk '{print $5}')
        echo http://$NODE_IP:$NODE_PORT

        gcloud compute firewall-rules create --quiet ${CLUSTER_NAME} --allow tcp:$NODE_PORT,tcp:$NODE_PORT
        ;;

    upgrade )
        helm upgrade --wait --recreate-pods -f ${MYVALUES_FILE} --timeout 900 --install ${RELEASE_NAME} charts/mlbench \
            --set limits.workers=${NUM_NODES} \
            --set worker.replicaCount=${NUM_NODES}
        ;;

    uninstall )
        helm delete --purge ${RELEASE_NAME}
        gcloud compute firewall-rules delete --quiet ${CLUSTER_NAME}
        ;;

    cleanup )
        gcloud compute firewall-rules delete --quiet ${CLUSTER_NAME}
        gcloud container clusters delete --quiet --zone ${MACHINE_ZONE}  ${CLUSTER_NAME}
        ;;

    dashboard )
        export NODE_PORT=$(kubectl get --namespace default -o jsonpath="{.spec.ports[0].nodePort}" services ${RELEASE_NAME}-mlbench-master)
        export NODE_IP=$(gcloud compute instances list|grep $(kubectl get nodes --namespace default -o jsonpath="{.items[0].status.addresses[0].address}") |awk '{print $5}')

        node=$(kubectl get pods -o wide | grep mlbench-master | awk '{print $7}')
        echo "Node of master pod is " $node
        
        externalip=$(kubectl get nodes $node -o wide | tail -n 1 | awk '{print $6}')
        echo "External IP of the node is " $externalip

        echo "Dashboard URL: " http://$externalip:$NODE_PORT
        ;;

    status)
        helm status ${RELEASE_NAME}
        ;;

    get-credentials)
        gcloud container clusters get-credentials --zone ${MACHINE_ZONE} ${CLUSTER_NAME}
        ;;

    list )
        echo "CLUSTER_NAME=${CLUSTER_NAME}"
        echo "MACHINE_TYPE=${MACHINE_TYPE}"
        echo "NUM_NODES=${NUM_NODES}"
        echo "NUM_GPU_PER_NODE=${NUM_GPU_PER_NODE}"
        echo "RELEASE_NAME=${RELEASE_NAME}"
        echo "PREEMPTIBLE=${PREEMPTIBLE}"
        echo "MYVALUES_FILE=${MYVALUES_FILE}"
        echo "CLUSTER_VERSION=${CLUSTER_VERSION}"
        echo "INSTANCE_DISK_SIZE=${INSTANCE_DISK_SIZE}"
        echo "MACHINE_ZONE=${MACHINE_ZONE}"
        echo "DISK_TYPE=${DISK_TYPE}"
        echo "GPU_TYPE=${GPU_TYPE}"
        ;;
esac
