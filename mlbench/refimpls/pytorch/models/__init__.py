import models.resnet
import models.testnet


def get_model(options):
    """Return model in the device."""
    if options.model_name == 'testnet':
        model = models.testnet.TestNet()
    elif options.model_name.startswith('resnet'):
        from models.resnet import get_model
        model = get_model(options)
    elif options.model_name in ['logistic_regression', 'ridge_regression']:
        from models.linear_models import get_model
        model = get_model(options)
    else:
        raise NotImplementedError

    return model
