from django.db import models


class KubeNode(models.Model):
    name = models.CharField(max_length=50)
    labels = models.CharField(max_length=100)
    phase = models.CharField(max_length=20)
    ip = models.CharField(max_length=15)


    class Meta:
        managed = False
