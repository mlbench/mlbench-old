from django.db import models
from api.utils.run_utils import run_model_job


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

    def start(self):
        if self.job_id != "" or self.state != self.INITIALIZED:
            raise ValueError("Wrong State")
        self.save()

        run_model_job.delay(self)
