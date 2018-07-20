from rest_framework import serializers
from api.models import KubeNode


class KubeNodeSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = KubeNode
        fields = ['name', 'labels', 'phase', 'ip']