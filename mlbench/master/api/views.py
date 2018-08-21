from rest_framework.viewsets import ViewSet
from rest_framework.response import Response
from rest_framework import status
from api.models import KubePod, KubeMetric, ModelRun
from api.serializers import KubePodSerializer, ModelRunSerializer, KubeMetricsSerializer
import django_rq
from rq.job import Job

from itertools import groupby
from datetime import datetime
import pytz


class KubePodView(ViewSet):
    """Handles the /api/pods endpoint
    """

    serializer_class = KubePodSerializer

    def list(self, request, format=None):
        pod = KubePod.objects.all()

        serializer = KubePodSerializer(pod, many=True)
        return Response(serializer.data)


class KubeMetricsView(ViewSet):
    """Handles the /api/metrics endpoint
    """

    def list(self, request, format=None):
        """Get all metrics

        Arguments:
            request {[Django request]} -- The request object

        Keyword Arguments:
            format {string} -- Output format to use (default: {None})

        Returns:
            Json -- Object containing all metrics
        """

        pod_metrics = {pod.name: {
            g[0]: [
                {'date': e.date, 'value': e.value, 'cumulative': e.cumulative}
                for e in sorted(g[1], key=lambda x: x.date)
            ] for g in groupby(
                sorted(pod.metrics.all(), key=lambda m: m.name),
                key=lambda m: m.name)
        } for pod in KubePod.objects.all()}

        run_metrics = {run.name: {
            g[0]: [
                {'date': e.date, 'value': e.value}
                for e in sorted(g[1], key=lambda x: x.date)
            ] for g in groupby(
                sorted(run.metrics.all(), key=lambda m: m.name),
                key=lambda m: m.name)
        } for run in ModelRun.objects.all()}

        return Response({
            'pod_metrics': pod_metrics,
            'run_metrics': run_metrics},
            status=status.HTTP_200_OK)

    def retrieve(self, request, pk=None, format=None):
        """Get all metrics for a pod

        Arguments:
            request {[Django request]} -- The request object

        Keyword Arguments:
            pk {string} -- Name of the pod
            format {string} -- Output format to use (default: {None})

        Returns:
            Json -- Object containing all metrics for the pod
        """
        since = self.request.query_params.get('since', None)

        if since is not None:
            since = datetime.strptime(since, "%Y-%m-%dT%H:%M:%S.%fZ")
            since = pytz.utc.localize(since)

        metric_type = self.request.query_params.get('metric_type', 'pod')

        if metric_type == 'pod':
            pod = KubePod.objects.filter(name=pk).first()
            metrics = pod.metrics.all()
        else:
            run = ModelRun.objects.get(pk=pk)
            metrics = run.metrics.all()

        result = {
            g[0]: [
                {
                    'date': e.date,
                    'value': e.value,
                    'cumulative': e.cumulative
                }
                for e in sorted(g[1], key=lambda x: x.date)
                if since is None or e.date > since
            ] for g in groupby(
                sorted(metrics, key=lambda m: m.name),
                key=lambda m: m.name)
        }

        return Response(result, status=status.HTTP_200_OK)

    def create(self, request):
        """Create a new metric

        Arguments:
            request {[Django request]} -- The request object

        Returns:
            Json -- Returns posted values
        """

        d = request.data

        metric = None

        if 'pod_name' in d:
            pod = KubePod.objects.filter(name=d['pod_name']).first()

            if pod is None:
                return Response({
                    'status': 'Not Found',
                    'message': 'Pod not found'
                }, status=status.HTTP_404_NOT_FOUND)

            metric = KubeMetric(
                name=d['name'],
                date=d['date'],
                value=d['value'],
                metadata=d['metadata'],
                cumulative=d['cumulative'],
                pod=pod)
            metric.save()

            return Response(
                metric, status=status.HTTP_201_CREATED
            )

        elif 'run_id' in d:
            run = ModelRun.objects.get(pk=d['run_id'])

            if run is None:
                return Response({
                    'status': 'Not Found',
                    'message': 'Run not found'
                }, status=status.HTTP_404_NOT_FOUND)

            metric = KubeMetric(
                name=d['name'],
                date=d['date'],
                value=d['value'],
                metadata=d['metadata'],
                cumulative=d['cumulative'],
                model_run=run)
            metric.save()

            serializer = KubeMetricsSerializer(metric, many=False)

            return Response(
                serializer.data, status=status.HTTP_201_CREATED
            )

        else:
            return Response({
                'status': 'Bad Request',
                'message': 'Pod Name or run id have to be supplied',
                'data': d,
            }, status=status.HTTP_400_BAD_REQUEST)


class ModelRunView(ViewSet):
    """Handles Model Runs
    """
    serializer_class = ModelRunSerializer

    def list(self, request, format=None):
        """Get all runs

        Arguments:
            request {[Django request]} -- The request object

        Keyword Arguments:
            format {string} -- Output format to use (default: {None})

        Returns:
            Json -- Object containing all runs
        """

        runs = ModelRun.objects.all()

        serializer = ModelRunSerializer(runs, many=True)

        return Response(serializer.data, status=status.HTTP_200_OK)

    def retrieve(self, request, pk=None, format=None):
        """Get all details for a run

        Arguments:
            request {[Django request]} -- The request object

        Keyword Arguments:
            pk {string} -- Id of the run
            format {string} -- Output format to use (default: {None})

        Returns:
            Json -- Object containing all metrics for the pod
        """
        run = ModelRun.objects.get(pk=pk)

        redis_conn = django_rq.get_connection()
        job = Job.fetch(run.job_id, redis_conn)
        run.job_metadata = job.meta

        serializer = ModelRunSerializer(run, many=False)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def create(self, request):
        """ Create and start a new Model run

        Arguments:
            request {[Django request]} -- The request object

        Returns:
            Json -- Returns posted values
        """
        # TODO: lock table, otherwise there might be concurrency conflicts
        d = request.data

        active_runs = ModelRun.objects.filter(state=ModelRun.STARTED)

        if active_runs.count() > 0:
            return Response({
                'status': 'Conflict',
                'message': 'There is already an active run'
            }, status=status.HTTP_409_CONFLICT)

        cpu = "{}m".format(float(d['num_cpus']) * 1000)

        run = ModelRun(
            name=d['name'],
            num_workers=d['num_workers'],
            cpu_limit=cpu,
            network_bandwidth_limit=d['max_bandwidth']
        )

        run.start()

        serializer = ModelRunSerializer(run, many=False)

        return Response(
            serializer.data, status=status.HTTP_201_CREATED
        )

    def destroy(self, request, pk=None):
        """Delete a run from the db

        Arguments:
            request {[Django request]} -- The request object

        Keyword Arguments:
            pk {int} -- [the id of the run] (default: {None})
        """
        run = ModelRun.objects.get(pk=pk)

        if run is not None:
            run.delete()

        return Response({
            'status': 'Deleted',
            'message': 'The run was deleted'
        }, status=status.HTTP_204_NO_CONTENT)
