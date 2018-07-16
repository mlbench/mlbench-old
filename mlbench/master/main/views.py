from django.http import HttpResponse
from kubernetes import client, config
from kubernetes.stream import stream


# Create your views here.
def index(request):
    return HttpResponse("Welcome!")


def run_mpi(request):
    config.load_incluster_config()

    v1 = client.CoreV1Api()

    ret = v1.list_namespaced_pod(
        "default",
        label_selector="component=worker,app=mlbench")
    result = ""
    hosts = []
    for i in ret.items:
        result += "<br>{}}&nbsp;&nbsp;&nbsp;&nbsp;{}&nbsp;&nbsp;&nbsp;&nbsp;"\
            "{}&nbsp;&nbsp;&nbsp;&nbsp;{}".format(i.status.pod_ip,
                                                  i.metadata.namespace,
                                                  i.metadata.name,
                                                  str(i.metadata.labels))
        hosts.append(i.status.pod_ip)

    exec_command = [
        'sh',
        '/usr/bin/mpirun',
        '--host', ",".join(hosts),
        '/usr/local/bin/python', '/app/main.py']
    result += "<br>" + str(exec_command)

    name = ret.items[0].metadata.name

    result += "<br>" + name

    resp = stream(v1.connect_get_namespaced_pod_exec, name, 'default',
                  command=exec_command,
                  stderr=True, stdin=False,
                  stdout=True, tty=False)

    result += "<br>" + resp
    return HttpResponse(result)
