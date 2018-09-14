from kubernetes import client, config
from django_rq import job
from pid import PidFile, PidFileError
from django.db.models import Max

import os
import json
import urllib
import pytz
from datetime import datetime


@job
def check_new_pods():
    """Background Task to look for new pods available in cluster
    """

    from api.models.kubepod import KubePod

    try:
        with PidFile('new_pods') as p:
            print(p.pidname)
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
                if KubePod.objects.filter(name=i.metadata.name).count() == 0:
                    ip = i.status.pod_ip
                    if ip is None:
                        ip = ""

                    pod = KubePod(name=i.metadata.name,
                                  labels=i.metadata.labels,
                                  phase=i.status.phase,
                                  ip=ip,
                                  node_name=i.spec.node_name)
                    pod.save()
                if i.metadata.name in all_pods:
                    all_pods.remove(i.metadata.name)

            KubePod.objects.filter(name__in=all_pods).delete()

    except PidFileError:
        return


@job
def check_pod_status():
    """Background Task to update status/phase of known pods
    """

    from api.models.kubepod import KubePod

    try:
        with PidFile('pod_status') as p:
            print(p.pidname)

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

    except PidFileError:
        return


@job
def check_pod_metrics():
    """Background task to get metrics (cpu/memory etc.) of known pods
    """
    print("running pod metrics!")
    from api.models.kubemetric import KubeMetric
    from api.models.kubepod import KubePod

    try:
        with PidFile('pod_metrics') as p:
            print(p.pidname)

            all_pods = KubePod.objects.all()
            nodes = {p.node_name for p in all_pods}
            all_pods = {p.name: p for p in all_pods}

            if len(nodes) == 0:
                return

            pods = []

            for node in nodes:
                url = 'http://{}:10255/stats/summary/'.format(node)
                with urllib.request.urlopen(url) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    pods += data['pods']

            for pod in pods:
                if pod['podRef']['name'] not in all_pods:
                    continue

                current_pod = all_pods[pod['podRef']['name']]

                cont_data = pod['containers'][0]

                newest_cpu_time = current_pod.metrics.filter(name='cpu')\
                    .aggregate(Max('date'))['date__max']

                new_time = datetime.strptime(
                    cont_data['cpu']['time'],
                    "%Y-%m-%dT%H:%M:%SZ")
                new_time = pytz.utc.localize(new_time)

                if not newest_cpu_time or new_time > newest_cpu_time:
                    metric = KubeMetric(
                        name='cpu',
                        date=cont_data['cpu']['time'],
                        value=cont_data['cpu']['usageNanoCores'] / 10**9,
                        metadata="",
                        cumulative=False,
                        pod=current_pod
                    )
                    metric.save()

                newest_memory_time = current_pod.metrics.filter(name='memory')\
                    .aggregate(Max('date'))['date__max']

                new_time = datetime.strptime(
                    cont_data['memory']['time'],
                    "%Y-%m-%dT%H:%M:%SZ")
                new_time = pytz.utc.localize(new_time)

                if not newest_memory_time or new_time > newest_memory_time:
                    metric = KubeMetric(
                        name='memory',
                        date=cont_data['memory']['time'],
                        value=cont_data['memory']
                                       ['usageBytes'] / (1024 * 1024),
                        metadata="",
                        cumulative=False,
                        pod=current_pod
                    )
                    metric.save()

                newest_network_time = current_pod.metrics\
                    .filter(name='network_in')\
                    .aggregate(Max('date'))['date__max']

                new_time = datetime.strptime(
                    pod['network']['time'],
                    "%Y-%m-%dT%H:%M:%SZ")
                new_time = pytz.utc.localize(new_time)

                if not newest_network_time or new_time > newest_network_time:
                    metric = KubeMetric(
                        name='network_in',
                        date=pod['network']['time'],
                        value=pod['network']['rxBytes'] / (1024 * 1024),
                        metadata="",
                        cumulative=True,
                        pod=current_pod
                    )
                    metric.save()

                    metric = KubeMetric(
                        name='network_out',
                        date=pod['network']['time'],
                        value=pod['network']['txBytes'] / (1024 * 1024),
                        metadata="",
                        cumulative=True,
                        pod=current_pod
                    )
                    metric.save()
    except PidFileError:
        return
