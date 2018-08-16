from django.urls import path

from . import views

app_name = 'main'

urlpatterns = [
    path('', views.index, name='index'),
    path('worker/<str:pod_name>/', views.worker, name='worker'),
    path('runs/<int:run_id>/', views.run, name='run'),
    path('runs', views.runs, name='runs')
]
