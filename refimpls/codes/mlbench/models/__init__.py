from .testnet import TestNet


def get_model(context):
    print("Get Models.", context)
    if context.model.name == 'testnet':
        return TestNet(context.dataset.channels, context.dataset.num_classes)
    else:
        raise NotImplementedError
