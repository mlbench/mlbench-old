from django.db import models
from api.models import KubePod


class KubeMetric(models.Model):
    name = models.CharField(max_length=50)
    date = models.DateTimeField()
    value = models.CharField(max_length=100)
    metadata = models.TextField()

    pod = models.ForeignKey(KubePod, related_name='metrics', on_delete=models.CASCADE)
