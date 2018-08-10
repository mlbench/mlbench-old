from mlbench.utils import log
from .base import Train


def get_controlflow(context):
    log.centering("GET CONTROLFLOW")
    context.log('controlflow')

    if context.controlflow.name == 'train':
        cf = Train()
    else:
        raise NotImplementedError

    return cf
