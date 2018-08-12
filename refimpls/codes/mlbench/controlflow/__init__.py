from mlbench.utils import log
from mlbench.controlflow.base import TrainValidation


def get_controlflow(context):
    if context.controlflow.name == 'train':
        cf = TrainValidation()
    else:
        raise NotImplementedError

    return cf
