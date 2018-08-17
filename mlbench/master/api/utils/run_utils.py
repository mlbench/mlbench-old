import django_rq
from kubernetes import client, config
import kubernetes.stream as stream
from rq import get_current_job

import os
from time import sleep


def limit_resources(model_run, kube_api, name, namespace):
    name = "{}-mlbench-worker".format(name)

    body = [
        {
            'op': 'replace',
            'path': '/spec/replicas',
            'value': model_run.num_workers
        },
        {
            'op': 'replace',
            'path': '/spec/template/spec/containers/0/resources/limits/cpu',
            'value': model_run.cpu_limit
        }
        ]

    kube_api.api_instance.patch_namespaced_stateful_set(name, namespace, body)

    while True:
        response = kube_api.read_namespaced_stateful_set_status(name,
                                                                namespace)
        s = response.status
        if (s.updated_replicas == response.spec.replicas and
            s.replicas == response.spec.replicas and
            s.available_replicas == response.spec.replicas and
            s.observed_generation >= response.metadata.generation):
            return

        sleep(1)



@django_rq.job('default', result_ttl=-1)
def run_model_job(model_run, experiment="test_MPI"):
    """RQ Job to execute OpenMPI

    Arguments:
        model_run {models.ModelRun} -- the database entry this job is associated with
    """

    from api.models import ModelRun

    try:
        job = get_current_job()

        model_run.job_id = job.id
        model_run.state = ModelRun.STARTED
        model_run.save()

        config.load_incluster_config()

        v1 = client.CoreV1Api()

        release_name = os.environ.get('MLBENCH_KUBE_RELEASENAME')
        ns = os.environ.get('MLBENCH_NAMESPACE')

        limit_resources(model_run, v1, release_name, ns)

        # start run
        ret = v1.list_namespaced_pod(
            ns,
            label_selector="component=worker,app=mlbench,release={}"
            .format(release_name))

        pods = []
        hosts = []
        for i in ret.items:
            pods.append((i.status.pod_ip,
                         i.metadata.namespace,
                         i.metadata.name,
                         str(i.metadata.labels)))
            hosts.append("{}.{}".format(i.metadata.name, release_name))

        job.meta['pods'] = pods
        job.save()

        exec_command = [
            '/.openmpi/bin/mpirun',
            "--mca", "btl_tcp_if_exclude", "docker0,lo",
            '--host', ",".join(hosts),
            '/conda/bin/python', "/codes/main.py",
            '--experiment', experiment,
            '--run_id',
            model_run.id]
        job.meta['command'] = str(exec_command)

        name = ret.items[0].metadata.name

        job.meta['master_name'] = name
        job.save()

        job.meta['stdout'] = []
        job.meta['stderr'] = []

        resp = stream.stream(v1.connect_get_namespaced_pod_exec, name,
                             ns,
                             command=exec_command,
                             stderr=True, stdin=False,
                             stdout=True, tty=False,
                             _preload_content=False)

        # keep writing openmpi output to job metadata
        while resp.is_open():
            resp.update(timeout=1)
            if resp.peek_stdout():
                out = resp.read_stdout()
                job.meta['stdout'] += out.splitlines()
            if resp.peek_stderr():
                err = resp.read_stderr()
                job.meta['stderr'] += err.splitlines()

            job.save()

        model_run.state = ModelRun.FINISHED
        model_run.save()
    except Exception as e:
        model_run.state = ModelRun.FAILED
        model_run.save()
