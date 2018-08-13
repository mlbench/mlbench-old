from django.db import models
from api.models import KubePod, ModelRun


class KubeMetric(models.Model):
    name = models.CharField(max_length=50)
    date = models.DateTimeField()
    value = models.CharField(max_length=100)
    metadata = models.TextField()

    pod = models.ForeignKey(
        KubePod,
        related_name='metrics',
        blank=True,
        null=True,
        on_delete=models.CASCADE)

    model_run = models.ForeignKey(
        ModelRun,
        related_name='metrics',
        blank=True,
        null=True,
        on_delete=models.CASCADE)
