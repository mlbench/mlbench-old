from rest_framework import routers
from django.urls import path, include

from api import views

router = routers.DefaultRouter()

router.register(r'nodes', views.KubePodView, base_name='node')
router.register(r'mpi_jobs', views.MPIJobView, base_name='mpi_job')
router.register(r'metrics', views.KubeMetricsView, base_name='metrics')

urlpatterns = [
    path(r'', include(router.urls)),
]