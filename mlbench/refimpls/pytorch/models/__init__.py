from .testnet import TestNet

from utils import log


def get_model(context):
    if context.model.name == 'testnet':
        model = TestNet()
    elif context.model.name == 'resnet18_bkj':
        from .resnet import resnet18_bkj
        # Use the resnet18 used in https://github.com/bkj/basenet/
        model = resnet18_bkj(context)
    else:
        raise NotImplementedError

    log.todo("TODO: convert model to the corresponding device.", 0)

    if context.meta.use_cuda:
        model.cuda()

    return model
