
class ControlFlow(object):
    def __call__(*args, **kwargs):
        pass


def get_controlflow(params):
    print("Get controlflow.", params)
    return ControlFlow()
