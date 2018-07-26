from rest_framework.viewsets import ViewSet
from rest_framework.response import Response
from rest_framework import status
from api.models import KubePod
from api.serializers import KubePodSerializer

from kubernetes import client, config
import kubernetes.stream as stream
from prometheus_client.parser import text_string_to_metric_families

import os
from collections import defaultdict
import urllib.request


class KubePodView(ViewSet):
    serializer_class = KubePodSerializer

    def list(self, request, format=None):
        config.load_incluster_config()

        v1 = client.CoreV1Api()

        release_name = os.environ.get('MLBENCH_KUBE_RELEASENAME')

        ret = v1.list_namespaced_pod(
            "default",
            label_selector="component=worker,app=mlbench,release={}"
            .format(release_name))
        nodes = []
        for i in ret.items:
            node = KubePod(
                name=i.metadata.name,
                labels=str(i.metadata.labels),
                phase=i.status.phase,
                ip=i.status.pod_ip)
            nodes.append(node)

        serializer = KubePodSerializer(nodes, many=True)
        return Response(serializer.data)


class KubeMetricsView(ViewSet):
    def list(self, request, format=None):
        config.load_incluster_config()
        v1 = client.CoreV1Api()
        release_name = os.environ.get('MLBENCH_KUBE_RELEASENAME')
        ret = v1.list_namespaced_pod(
            "default",
            label_selector="component=worker,app=mlbench,release={}"
            .format(release_name))

        pods = defaultdict(list)
        pod_names = []
        for i in ret.items:
            pods[i.spec.node_name].append(i.metadata.name)
            pod_names.append(i.metadata.name)

        metrics = []

        for node in pods.keys():
            url = 'http://{}:10255/metrics/cadvisor'.format(node)
            with urllib.request.urlopen(url) as response:
                metrics.append(response.read().decode('utf-8'))

        metrics = "\n".join(metrics)

        result = {}

        for family in text_string_to_metric_families(metrics):
            for sample in family.samples:
                if ('pod_name' in sample[1]
                        and sample[1]['pod_name'] in pod_names):
                    if sample[1]['pod_name'] not in result:
                        result[sample[1]['pod_name']] = {}
                    result[sample[1]['pod_name']][sample[0]] = {
                        'labels': sample[1],
                        'value': sample[2]}

        return Response(result, status=status.HTTP_200_OK)

    def retrieve(self, request, pk=None, format=None):
        config.load_incluster_config()
        v1 = client.CoreV1Api()

        pod = v1.read_namespaced_pod(pk, "default")

        node = pod.spec.node_name

        url = 'http://{}:10255/metrics/cadvisor'.format(node)
        with urllib.request.urlopen(url) as response:
            metrics = response.read().decode('utf-8')

        result = {}

        for family in text_string_to_metric_families(metrics):
            for sample in family.samples:
                if ('pod_name' in sample[1]
                        and sample[1]['pod_name'] == pk):
                    result[sample[0]] = {
                        'labels': sample[1],
                        'value': sample[2]}

        return Response(result, status=status.HTTP_200_OK)



class MPIJobView(ViewSet):

    def create(self, request):
        config.load_incluster_config()

        v1 = client.CoreV1Api()

        release_name = os.environ.get('MLBENCH_KUBE_RELEASENAME')

        ret = v1.list_namespaced_pod(
            "default",
            label_selector="component=worker,app=mlbench,release={}"
            .format(release_name))

        result = {'nodes': []}
        hosts = []
        for i in ret.items:
            result['nodes'].append((i.status.pod_ip,
                                    i.metadata.namespace,
                                    i.metadata.name,
                                    str(i.metadata.labels)))
            hosts.append("{}.{}".format(i.metadata.name, release_name))

        exec_command = [
            'sh',
            '/usr/bin/mpirun',
            '--host', ",".join(hosts),
            '/usr/local/bin/python', '/app/main.py']
        result['command'] = str(exec_command)

        name = ret.items[0].metadata.name

        result['master_name'] = name

        resp = stream.stream(v1.connect_get_namespaced_pod_exec, name,
                             'default',
                             command=exec_command,
                             stderr=True, stdin=False,
                             stdout=True, tty=False)

        result["response"] = resp
        return Response(result, status=status.HTTP_201_CREATED)
