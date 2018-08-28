"""
"""
import datetime
import json
import logging
import subprocess
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
        logger.warning("{}".format(content))


def critical(content, who='all'):
    if who == 'all' or who == dist.get_rank():
        logger.critical("{}".format(content))


class AsyncMetricsPost(object):
    """Post metrics payload to endpoint in an asynchronized way."""

    def __init__(self):
        self._initialized = False
        self._incluster = True

    def init(self):
        from kubernetes import config, client
        try:
            config.load_incluster_config()
        except Exception as e:
            self._incluster = False
            return

        configuration = client.Configuration()

        class MyApiClient(client.ApiClient):
            """
            A bug introduced by a fix.

            https://github.com/kubernetes-client/python/issues/411
            https://github.com/swagger-api/swagger-codegen/issues/6392
            """

            def __del__(self):
                pass

        self.api_instance = client.CoreV1Api(MyApiClient(configuration))

        # TODO: remove hardcoded part in the future.
        self.namespace = 'default'
        label_selector = 'component=master,app=mlbench'

        try:
            api_response = self.api_instance.list_namespaced_pod(
                self.namespace, label_selector=label_selector)
        except Exception as e:
            print("Exception when calling CoreV1Api->list_namespaced_pod: %s\n" % e)

        assert len(api_response.items) == 1
        master_pod = api_response.items[0]
        ip = master_pod.status.pod_ip
        self.endpoint = "http://{ip}/api/metrics/".format(ip=ip)
        self._initialized = True

    def post(self, payload):
        """Post information via kubernetes client.

        Example:

            payload = {
                "run_id": "1",
                "name": "accuracy",
                "cumulative": False,
                "date": "2018-08-14T09:21:44.331823Z",
                "value": "1.0",
                "metadata": "some additional data"
            }

        See `KubeMetric` in kubemetric.py for the fields and types.

        """
        if not self._initialized:
            self.init()
            if not self._incluster:
                return

        command = [
            "/usr/bin/curl",
            "-d", json.dumps(payload),
            "-H", "Content-Type: application/json",
            "-X", "POST", self.endpoint
        ]
        subprocess.Popen(command)


async_post = AsyncMetricsPost()


def _post_metrics(payload, rank):
    if rank == 0 and async_post._incluster:
        async_post.post(payload)


def todo(content, who='all'):
    if who == 'all' or who == dist.get_rank():
        logger.warning("{}".format(content))


def configuration_information(options):
    centering("Configuration Information", 0)
    options.log('meta')
    options.log('optimizer')
    options.log('model')
    options.log('dataset')
    options.log('controlflow')
    centering("START TRAINING", 0)


def log_val(options, best_metric_name):
    info('{} for rank {}:(best epoch {}, current epoch {}): {:.3f}'.format(
        best_metric_name,
        options.rank,
        options.runtime['best_epoch'],
        options.runtime['current_epoch'],
        options.runtime[best_metric_name]), 0)


def post_metrics(options, metric_name, value):
    _post_metrics({
        "run_id": options.run_id,
        "name": metric_name,
        "value": "{:.3f}".format(value),
        "date": str(datetime.datetime.now()),
        "cumulative": False,
        "metadata": "Validation {} at epoch {}".format(
            metric_name, options.runtime['current_epoch'])
    }, options.rank)
