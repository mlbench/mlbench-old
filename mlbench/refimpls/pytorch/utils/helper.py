import time


class Timeit(object):
    def __enter__(self):
        self.t = time.time()
        self._cumu = 0
        return self

    def __exit__(self, type, value, traceback):
        pass

    def pause(self):
        self._cumu += time.time() - self.t
        self.t = time.time()

    def resume(self):
        self.t = time.time()

    @property
    def cumu(self):
        return self._cumu
