from api.utils.run_utils import run_model_job

from django.db import models
from django.db.models.signals import pre_delete
from django.dispatch import receiver
import django_rq
from rq.job import Job


class ModelRun(models.Model):
    INITIALIZED = "initialized"
    STARTED = "started"
    FAILED = "failed"
    FINISHED = "finished"
    STATE_CHOICES = [
        (STARTED, STARTED),
        (FAILED, FAILED),
        (FINISHED, FINISHED)]

    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    state = models.CharField(
        max_length=20,
        choices=STATE_CHOICES,
        default=INITIALIZED)
    job_id = models.CharField(max_length=38, default="")

    cpu_limit = models.CharField(max_length=20, default="12000m")
    num_workers = models.IntegerField(default=2)

    job_metadata = {}

    def start(self):
        """Saves the model run and starts the RQ job

        Raises:
            ValueError -- Raised if state is not initialized
        """

        if self.job_id != "" or self.state != self.INITIALIZED:
            raise ValueError("Wrong State")
        self.save()

        run_model_job.delay(self)


@receiver(pre_delete, sender=ModelRun, dispatch_uid='run_delete_job')
def remove_run_job(sender, instance, using, **kwargs):
    """Signal to delete job when ModelRun is deleted"""
    redis_conn = django_rq.get_connection()
    job = Job.fetch(instance.job_id, redis_conn)
    job.delete()

