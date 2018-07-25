from django.apps import AppConfig
from redis import Redis
from rq import Queue
from rq_scheduler import Scheduler
from datetime import datetime


class ApiConfig(AppConfig):
    name = 'api'

    def ready(self):
        scheduler = Scheduler(connection=Redis())

        scheduler.schedule(scheduled_time=datetime.utcnow(), func=..., interval=1, repeat=None)

