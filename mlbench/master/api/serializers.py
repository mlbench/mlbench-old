from rest_framework import serializers
from api.models import KubePod


class KubePodSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = KubePod
        fields = ['name', 'labels', 'phase', 'ip']