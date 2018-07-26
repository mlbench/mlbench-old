from django.apps import AppConfig
from api.utils.node_monitor import setup

import sys


class ApiConfig(AppConfig):
    name = 'api'

    def ready(self):
        if 'migrate' not in sys.argv and 'collectstatic' not in sys.argv:
            setup()