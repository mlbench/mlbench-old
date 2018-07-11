from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('first_time', views.first_time, name='first_time')
]