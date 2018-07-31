from rest_framework.viewsets import ViewSet
from rest_framework.response import Response
from rest_framework import status
from api.models import KubePod, KubeMetric
from api.serializers import KubePodSerializer

from kubernetes import client, config
import kubernetes.stream as stream

import os
from collections import defaultdict
import urllib.request
from itertools import groupby
from datetime import datetime


class KubePodView(ViewSet):
    serializer_class = KubePodSerializer

    def list(self, request, format=None):
        pod = KubePod.objects.all()

        serializer = KubePodSerializer(pod, many=True)
        return Response(serializer.data)


class KubeMetricsView(ViewSet):
    def list(self, request, format=None):
        result = {p.name: {
            g.key: [
                e.value for e in sorted(g, lambda x: x.date)
            ] for g in groupby(p.metrics, lambda m: m.name)
            } for p in KubePod.objects.all()}

        return Response(result, status=status.HTTP_200_OK)

    def retrieve(self, request, pk=None, format=None):
        since = self.request.query_params.get('since', None)

        if since is not None:
            since = datetime.strptime(since, "yyyy-MM-dd HH:mm:ss")

        pod = KubePod.objects.filter(name=pk).first()

        result = {
            g.key: [
                e.value for e in sorted(g, lambda x: x.date)
                if since is None or e.date > since
            ] for g in groupby(pod.metrics, lambda m: m.name)
            }

        return Response(result, status=status.HTTP_200_OK)

    def create(self, request):
        d = request.data
        pod = KubePod.objects.filter(name=d.pod_name).first()

        if pod is None:
            return Response({
                'status': 'Not Found',
                'message': 'Pod not found'
            }, status=status.HTTP_404_NOT_FOUND)

        metric = KubeMetric(
            name=d.name,
            date=d.date,
            value=d.value,
            metadata=d.metadata,
            pod=pod)
        metric.save()

        return Response(
            metric, status=status.HTTP_201_CREATED
        )


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
