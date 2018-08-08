from django.shortcuts import render
import django_rq
from rq.job import Job

from api.models import ModelRun


# Create your views here.
def index(request):
    return render(request, 'main/index.html')


def runs(request):
    runs = ModelRun.objects.all()
    return render(request, 'main/runs.html', {'runs': runs})


def run(request, run_id):
    run = ModelRun.objects.get(pk=run_id)

    redis_conn = django_rq.get_connection()
    job = Job.fetch(run.job_id, redis_conn)

    run.job_metadata = job.meta

    return render(request,
                  'main/run_detail.html',
                  {'run': run})
