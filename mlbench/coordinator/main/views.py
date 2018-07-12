from django.http import HttpResponse
from kubernetes import client, config


# Create your views here.
def index(request):
    return HttpResponse("Welcome! yaynayfsdfsdfxcvxcvxd!")


def first_time(request):
    # Configs can be set in Configuration class directly or using helper utility
    try:
        config.load_incluster_config()

        v1 = client.CoreV1Api()
        result = "Listing pods with their IPs:"
        ret = v1.list_namespaced_pod("default")
        for i in ret.items:
            result += "<br>%s&nbsp;&nbsp;&nbsp;&nbsp;%s&nbsp;&nbsp;&nbsp;&nbsp;%s&nbsp;&nbsp;&nbsp;&nbsp;%s" % (i.status.pod_ip, i.metadata.namespace, i.metadata.name, str(i.metadata.labels))
        return HttpResponse(result)
    except Exception as e:
        return HttpResponse(str(e))