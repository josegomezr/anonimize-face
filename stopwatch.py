import time
from contextlib import ContextDecorator
import logging


class Stopwatch(ContextDecorator):
    def __init__(self, msg=None, logger=None):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.msg = msg

    def __enter__(self):
        self.start = time.monotonic()
        self.logger.debug('{} - start'.format(self.msg))
        return self

    def __exit__(self, *exc):
        self.end = time.monotonic()
        self.duration = self.end - self.start
        self.logger.debug('{} - finished in {}'.format(self.msg, self.duration))

        self.msg = None
        return False

    def __call__(self, msg=None):
        self.msg = msg
        return self
