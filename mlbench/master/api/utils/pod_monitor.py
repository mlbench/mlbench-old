from kubernetes import client, config
from django.utils import timezone
from django_rq import job

import os
import json
import urllib

from prometheus_client.parser import text_string_to_metric_families


@job
def check_new_pods():
    """Background Task to look for new pods available in cluster
    """

    from api.models.kubepod import KubePod

    config.load_incluster_config()
    v1 = client.CoreV1Api()

    release_name = os.environ.get('MLBENCH_KUBE_RELEASENAME')
    ns = os.environ.get('MLBENCH_NAMESPACE')

    ret = v1.list_namespaced_pod(
        ns,
        label_selector="component=worker,app=mlbench,release={}"
        .format(release_name))

    all_pods = list(KubePod.objects.all().values_list('name'))

    for i in ret.items:
        if not KubePod.objects.filter(name=i.metadata.name).count() > 0:
            ip = i.status.pod_ip
            if ip is None:
                ip = ""

            pod = KubePod(name=i.metadata.name,
                          labels=i.metadata.labels,
                          phase=i.status.phase,
                          ip=i.status.pod_ip,
                          node_name=i.spec.node_name)
            pod.save()
        if i.metadata.name in all_pods:
            all_pods.remove(i.metadata.name)

    KubePod.objects.filter(name__in=all_pods).delete()


@job
def check_pod_status():
    """Background Task to update status/phase of known pods
    """

    from api.models.kubepod import KubePod

    ns = os.environ.get('MLBENCH_NAMESPACE')

    config.load_incluster_config()
    v1 = client.CoreV1Api()

    pods = pods = KubePod.objects.all()

    for pod in pods:
        ret = v1.read_namespaced_pod(pod.name, ns)
        phase = ret.status.phase

        if phase != pod.phase:
            pod.phase = phase
            pod.save()


@job
def check_pod_metrics():
    """Background task to get metrics (cpu/memory etc.) of known pods
    """

    from api.models.kubemetric import KubeMetric
    from api.models.kubepod import KubePod

    pods = KubePod.objects.all()
    nodes = {p.node_name for p in pods}
    pods = {p.name: p for p in pods}

    metrics = []
    print(nodes)

    for node in nodes:
        url = 'http://{}:10255/metrics/cadvisor'.format(node)
        with urllib.request.urlopen(url) as response:
            metrics.append(response.read().decode('utf-8'))

    metrics = "\n".join(metrics)

    for family in text_string_to_metric_families(metrics):
        for sample in family.samples:
            if ('container_name' in sample[1]
                    and sample[1]['container_name'] != "mlbench-worker"):
                continue

            if ('pod_name' in sample[1]
                    and sample[1]['pod_name'] in pods.keys()):
                metric = KubeMetric(
                    name=sample[0],
                    date=timezone.now(),
                    value=sample[2],
                    metadata=json.dumps(sample[1]),
                    pod=pods[sample[1]['pod_name']]
                )
                metric.save()
