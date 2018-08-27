from rest_framework import routers
from rest_framework.urlpatterns import format_suffix_patterns
from django.urls import path, include

from api import views

app_name = 'api'

router = routers.DefaultRouter()
router.include_format_suffixes = False

router.register(r'pods', views.KubePodView, base_name='pod')
router.register(r'runs', views.ModelRunView, base_name='run')
router.register(r'metrics', views.KubeMetricsView, base_name='metrics')

urlpatterns = [
    path(r'', include(router.urls)),
]

urlpatterns = format_suffix_patterns(urlpatterns,
                                     allowed=['json', 'html', 'zip'])
