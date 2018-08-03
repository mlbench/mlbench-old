from django.shortcuts import render

from api.models import ModelRun


# Create your views here.
def index(request):
    return render(request, 'main/index.html')


def runs(request):
    runs = ModelRun.objects.all()
    return render(request, 'main/runs.html', {'runs': runs})
