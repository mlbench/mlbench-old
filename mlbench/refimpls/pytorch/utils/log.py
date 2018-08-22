"""
colors: https://stackoverflow.com/questions/5947742/how-to-change-the-output-color-of-echo-in-linux
"""
import logging
import torch.distributed as dist
import json
import time
import subprocess
import datetime


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


def post_metrics(payload, rank):
    if rank == 0 and async_post._incluster:
        async_post.post(payload)


def todo(content, who='all'):
    if who == 'all' or who == dist.get_rank():
        logger.warning("{}".format(content))


def configuration_information(context):
    centering("Configuration Information", 0)
    context.log('meta')
    context.log('optimizer')
    context.log('model')
    context.log('dataset')
    context.log('controlflow')
    centering("START TRAINING", 0)


def log_train(context, batch_idx, loss):
    debug("Training Batch {:5}: loss={:.3f}".format(batch_idx, loss))
    post_metrics({
        "run_id": context.meta.run_id,
        "name": "train loss @ rank{}".format(context.meta.rank),
        "value": loss,
        "date": str(datetime.datetime.now()),
        "cumulative": False,
        "metadata":
        "Training loss at rank {}, epoch {} and batch {}".format(
            context.meta.rank, context.runtime.current_epoch,
            batch_idx
        )
    }, context.meta.rank)


def log_val(context, val_prec1):
    log.info('best accuracy for rank {}:(best epoch {}, current epoch {}): {:.3f} %'.format(
        context.meta.rank,
        context.runtime.best_epoch[-1] if len(context.runtime.best_epoch) != 0 else '',
        context.runtime.current_epoch, context.runtime.best_prec1), 0)

    log.post_metrics({
        "run_id": context.meta.run_id,
        "name": "Prec@1",
        "value": "{:.3f}".format(val_prec1),
        "date": str(datetime.datetime.now()),
        "cumulative": False,
        "metadata":
        "Validation Prec1 at epoch {}".format(
            context.runtime.current_epoch
        )
    }, context.meta.rank)
