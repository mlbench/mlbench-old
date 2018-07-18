from rest_framework import routers
from django.urls import path, include

from api import views

router = routers.DefaultRouter()

router.register(r'nodes', views.KubeNodeView, base_name='node')

urlpatterns = [
    path(r'', include(router.urls)),
]