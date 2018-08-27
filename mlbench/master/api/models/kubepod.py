from api.models import ModelRun

from django.db import models


class KubePod(models.Model):
    name = models.CharField(max_length=50)
    labels = models.CharField(max_length=255)
    phase = models.CharField(max_length=20)
    ip = models.CharField(max_length=15)
    node_name = models.CharField(max_length=100)

    model_run = models.ForeignKey(
        ModelRun,
        related_name='pods',
        blank=True,
        null=True,
        on_delete=models.SET_NULL)
