from .testnet import TestNet

from mlbench.utils import log


def get_model(context):
    log.centering("GET MODELS")
    if context.model.name == 'testnet':
        model = TestNet()
    else:
        raise NotImplementedError

    context.log('model')

    log.todo("TODO: convert model to the corresponding device.")

    return model
