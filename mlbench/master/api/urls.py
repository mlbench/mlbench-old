from rest_framework import routers
from django.urls import path, include

from api import views

router = routers.DefaultRouter()

router.register(r'pods', views.KubePodView, base_name='pod')
router.register(r'runs', views.ModelRunView, base_name='run')
router.register(r'metrics', views.KubeMetricsView, base_name='metrics')

urlpatterns = [
    path(r'', include(router.urls)),
]
