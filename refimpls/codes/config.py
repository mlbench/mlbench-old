

def set_seed(manual_seed):
    pass


def set_logging():
    pass


def set_pytorch():
    pass


def load_config_file(config_file):
    # if config_file is None:
    #     return {}
    # else:
    #     with open(config_file, 'r') as f:
    #         return json.load(f)
    pass


class Context(object):
    def __init__(self):
        self.optimizer = {}
        self.dataset = {}
        self.model = {}
        self.controlflow = {}


def init_context(args):
    print("init_context", args)
    return Context()
