from utils import log
from .train_val import TrainValidation


def get_controlflow(context):
    """Get optimizer and scheduler for the given configuration.

    Using the configurations in the `context`, create an optimizer associated with
    parameters of `model`. A learning rate for optimizer is created as well.

    Parameters
    ----------
    context : {Context}
         A context object containing all configurations.

    Returns
    -------
    controlflow
        A controlflow object.

    Raises
    ------
    NotImplementedError
        The controlflow specified by `context` is not implemented.
    """
    if context.controlflow.name == 'train_val':
        cf = TrainValidation()
    else:
        raise NotImplementedError("The controlflow `{}` specified by `context` is not implemented."
                                  .format(context.controlflow.name))

    return cf
