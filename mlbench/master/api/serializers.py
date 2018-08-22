from rest_framework import serializers
from api.models import KubePod, ModelRun, KubeMetric


class KubePodSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = KubePod
        fields = ['name', 'labels', 'phase', 'ip']


class ModelRunSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = ModelRun
        fields = [
            'id',
            'name',
            'created_at',
            'state',
            'job_id',
            'job_metadata']


class KubeMetricsSerializer(serializers.HyperlinkedModelSerializer):
    name = serializers.CharField(max_length=50)
    value = serializers.CharField(max_length=100)
    date = serializers.DateTimeField(format='%Y-%m-%dT%H:%M:%S.%fZ')
    metadata = serializers.CharField()
    cumulative = serializers.BooleanField(default=False)

    class Meta:
        model = KubeMetric
        fields = ['name', 'value', 'date', 'metadata', 'cumulative']
