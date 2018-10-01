import time
import itertools
from utils import log


class Timeit(object):
    def __init__(self, cumu=0):
        self.t = time.time()
        self._cumu = cumu
        self._paused = False

    def pause(self):
        if not self._paused:
            self._cumu += time.time() - self.t
            self.t = time.time()
            self._paused = True

    def resume(self):
        if self._paused:
            self.t = time.time()
            self._paused = False

    @property
    def cumu(self):
        return self._cumu


def maybe_range(maximum):
    """Map an integer or None to an integer iterator starting from 0 with strid 1.

    If maximum number of batches per epoch is limited, then return an finite
    iterator. Otherwise, return an iterator of infinite length. 
    """
    if maximum is None:
        counter = itertools.count(0)
    else:
        counter = range(maximum)
    return counter


def update_best_runtime_metric(options, metric_value, metric_name):
    """Update the runtime information to options if the metric value is the best."""
    best_metric_name = "best_{}".format(metric_name)
    if best_metric_name in options.runtime:
        is_best = metric_value > options.runtime[best_metric_name]
    else:
        is_best = True

    if is_best:
        options.runtime[best_metric_name] = metric_value
        options.runtime['best_epoch'] = options.runtime['current_epoch']

    return is_best, best_metric_name
