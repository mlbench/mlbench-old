"""
colors: https://stackoverflow.com/questions/5947742/how-to-change-the-output-color-of-echo-in-linux
"""
import logging
import torch.distributed as dist

logger = logging.getLogger('mlbench')


def _warp(string, symbol='*', length=80):
    one_side_length = (length - len(string) - 2) // 2
    if one_side_length > 0:
        return symbol * one_side_length + ' ' + string + ' ' + symbol * one_side_length
    else:
        return string


def centering(content, who='all', symbol='*', length=80):
    info(_warp(content, symbol, length), who=who)


def info(content, who='all'):
    if who == 'all' or who == dist.get_rank():
        logger.info(content)


def debug(content, who='all'):
    if who == 'all' or who == dist.get_rank():
        logger.debug(content)


def warning(content, who='all'):
    if who == 'all' or who == dist.get_rank():
        logger.warning("\033[0;31m{}\033[0m".format(content))


def critical(content, who='all'):
    if who == 'all' or who == dist.get_rank():
        logger.critical("\033[0;104m{}\033[0m".format(content))


def post_metrics(payload):
    """Post information via kubernetes client.

    Example:

        payload = {
            "run_id": "1",
            "name": "accuracy",
            "date": "2018-08-14T09:21:44.331823Z",
            "value": "1.0",
            "metadata": "some additional data"
        }

    See `KubeMetric` in kubemetric.py for the fields and types.

    """
    from kubernetes import config, client
    import requests
    import json

    config.load_incluster_config()
    configuration = client.Configuration()
    api_instance = client.CoreV1Api(client.ApiClient(configuration))

    namespace = 'default'
    label_selector = 'component=master,app=mlbench'
    try:
        api_response = api_instance.list_namespaced_pod(
            namespace, label_selector=label_selector)
    except ApiException as e:
        print("Exception when calling CoreV1Api->list_namespaced_pod: %s\n" % e)

    assert len(api_response.items) == 1
    ip = api_response.items[0].status.pod_ip

    response = requests.post("http://{ip}/api/metrics/".format(ip=ip), data=payload)
    debug('Post metrics to master - OK={} '.format(response.ok))


def todo(content, who='all'):
    if who == 'all' or who == dist.get_rank():
        logger.warning("\033[0;33m{}\033[0m".format(content))


def configuration_information(context):
    centering("Configuration Information", 0)

    centering('Opimizer', 0, symbol='=', length=40)
    context.log('optimizer')

    centering('Model', 0, symbol='=', length=40)
    context.log('model')

    centering('Dataset', 0, symbol='=', length=40)
    context.log('dataset')

    centering('Controlflow', 0, symbol='=', length=40)
    context.log('controlflow')

    centering("START TRAINING", 0)
