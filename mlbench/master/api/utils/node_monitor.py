from redis import Redis
from rq import Queue
from rq_scheduler import Scheduler

from kubernetes import client, config

import os
from datetime import datetime


def setup():
    scheduler = Scheduler(connection=Redis())

    scheduler.schedule(
        scheduled_time=datetime.utcnow(),
        func=check_new_nodes,
        interval=60,
        repeat=None
    )
    print("scheduled")


def check_new_nodes():
    from api.models.kubepod import KubePod

    config.load_incluster_config()
    v1 = client.CoreV1Api()
    release_name = os.environ.get('MLBENCH_KUBE_RELEASENAME')
    ret = v1.list_namespaced_pod(
        "default",
        label_selector="component=worker,app=mlbench,release={}"
        .format(release_name))

    all_pods = KubePod.objects.all().values_list('name')

    for i in ret.items:
        if not KubePod.objects.filter(name=i.metadata.name):
            pod = KubePod(name=i.metadata.name,
                          labels=i.metadata.labels,
                          phase=i.status.phase,
                          ip=i.status.pod_ip)
            pod.save()
        else:
            all_pods.remove(i.metadata.name)

    for pod in all_pods:
        KubePod.objects.filter(name=pod).remove()


def check_node_status():
    pass


def check_node_metrics():
    pass
