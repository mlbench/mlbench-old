from rest_framework import serializers
from api.models import KubePod, ModelRun


class KubePodSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = KubePod
        fields = ['name', 'labels', 'phase', 'ip']


class ModelRunSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = ModelRun
        fields = ['name', 'created_at', 'state', 'job_id']
