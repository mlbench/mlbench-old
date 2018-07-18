from rest_framework.viewsets import ViewSet
from rest_framework.response import Response
from api.models import KubeNode
from api.serializers import KubeNodeSerializer

from kubernetes import client, config

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
