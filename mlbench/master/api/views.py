from rest_framework.viewsets import ViewSet
from rest_framework.response import Response
from rest_framework import status
from api.models import KubeNode
from api.serializers import KubeNodeSerializer

from kubernetes import client, config
import kubernetes.stream as stream

import os


class KubeNodeView(ViewSet):
    serializer_class = KubeNodeSerializer

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
            node = KubeNode(
                name=i.metadata.name,
                labels=str(i.metadata.labels),
                phase=i.status.phase,
                ip=i.status.pod_ip)
            nodes.append(node)

        serializer = KubeNodeSerializer(nodes, many=True)
        return Response(serializer.data)


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
            '/.openmpi/bin/mpirun',
            '--host', ",".join(hosts),
            '/conda/bin/python', '/app/main.py']
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
