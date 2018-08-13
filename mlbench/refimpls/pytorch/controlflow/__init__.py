from utils import log
from .base import TrainValidation


def get_controlflow(context):
    if context.controlflow.name == 'train':
        cf = TrainValidation()
    else:
        raise NotImplementedError

    return cf
