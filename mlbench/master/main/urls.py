from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('run_mpi', views.run_mpi, name='run_mpi')
]
