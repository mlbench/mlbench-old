from .testnet import TestNet

from mlbench.utils import log


def get_model(context):
    if context.model.name == 'testnet':
        model = TestNet()
    else:
        raise NotImplementedError

    log.todo("TODO: convert model to the corresponding device.", 0)

    return model
