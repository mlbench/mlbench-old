from redis import Redis
from rq import Queue
from rq_scheduler import Scheduler

from kubernetes import client, config

import os
from datetime import datetime
import json
import urllib

from prometheus_client.parser import text_string_to_metric_families


def setup():
    print(datetime.utcnow())
    connection = Redis()
    queue = Queue(connection=connection)
    scheduler = Scheduler(queue=queue, connection=connection)

    res = scheduler.schedule(
        scheduled_time=datetime.utcnow(),
        func=check_new_nodes,
        interval=60,
        repeat=None
    )
    print("scheduled")
    print(res)
    print(scheduler.get_jobs())


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
                          ip=i.status.pod_ip,
                          node_name=i.spec.node_name)
            pod.save()
        else:
            all_pods.remove(i.metadata.name)

    KubePod.objects.filter(name__in=all_pods).delete()


def check_node_status():
    pass


def check_node_metrics():
    from api.models.kubemetric import KubeMetric
    from api.models.kubepod import KubePod

    pods = KubePod.objects.all()
    nodes = {p.node_name for p in pods}
    pods = {p.name: p for p in pods}

    metrics = []

    for node in nodes:
        url = 'http://{}:10255/metrics/cadvisor'.format(node)
        with urllib.request.urlopen(url) as response:
            metrics.append(response.read().decode('utf-8'))

    metrics = "\n".join(metrics)

    for family in text_string_to_metric_families(metrics):
        for sample in family.samples:
            if ('pod_name' in sample[1]
                    and sample[1]['pod_name'] in pods.keys()):
                metric = KubeMetric(
                    name=sample[0],
                    date=datetime.utcnow(),
                    value=sample[2],
                    metadata=json.dumps(sample[1]),
                    pod=pods[sample[1]['pod_name']]
                )
                metric.save()
