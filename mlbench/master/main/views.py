from django.shortcuts import render
from django.utils import timezone
import django_rq
from rq.job import Job

from api.models import ModelRun, KubePod, KubeMetric

import os


# Create your views here.
def index(request):
    return render(request, 'main/index.html')


def worker(request, pod_name):
    worker = KubePod.objects.get(name=pod_name)
    metrics = worker.metrics.order_by('name').values('name').distinct()
    return render(request, 'main/worker_detail.html',
                  {'worker': worker, 'metrics': metrics})


def runs(request):
    """List all runs page"""
    runs = ModelRun.objects.all()

    max_workers = os.environ.get('MLBENCH_MAX_WORKERS')
    max_cpu = os.environ.get('MLBENCH_WORKERS_MAX_CPU')
    if "m" in max_cpu:
        max_cpu = int(max_cpu.replace("m")) / 1000
    else:
        max_cpu = int(max_cpu)


    return render(request, 'main/runs.html', {
        'runs': runs,
        'max_workers': max_workers,
        'max_cpus': max_cpu,
        'max_memory': 30000,
        'max_bandwidth': 1000})


def run(request, run_id):
    """Details of single run page"""
    run = ModelRun.objects.get(pk=run_id)

    redis_conn = django_rq.get_connection()
    job = Job.fetch(run.job_id, redis_conn)

    run.job_metadata = job.meta

    metric = KubeMetric(name="test",
                        date=timezone.now(),
                        value="0.0",
                        metadata="",
                        cumulative=False,
                        model_run=run)

    metric2 = KubeMetric(name="test",
                         date=timezone.now() + timezone.timedelta(minutes=1),
                         value="1.0",
                         metadata="",
                         cumulative=False,
                         model_run=run)

    metric.save()
    metric2.save()

    metrics = run.metrics.order_by('name').values('name').distinct()

    return render(request,
                  'main/run_detail.html',
                  {'run': run, 'metrics': metrics})
