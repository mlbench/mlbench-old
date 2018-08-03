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

    name = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)
    state = models.CharField(
        max_length=20,
        choices=STATE_CHOICES,
        default=INITIALIZED)
    job_id = models.CharField(max_length=20, default="")

    def start(self):
        if self.job_id != "" or self.state != self.INITIALIZED:
            raise ValueError("Wrong State")

        job = run_model_job.delay(self)

        self.job_id = job.id
        self.save()
