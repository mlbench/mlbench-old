from rest_framework import routers
from django.urls import path, include

from api import views

router = routers.DefaultRouter()

router.register(r'nodes', views.KubeNodeView, base_name='node')
router.register(r'mpi_jobs', views.MPIJobView, base_name='mpi_job')

urlpatterns = [
    path(r'', include(router.urls)),
]