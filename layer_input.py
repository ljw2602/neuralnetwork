import numpy as np


class Input(object):

    def __init__(self, size_):
        self._size = size_

    def get_size(self):
        """
        Return size
        :return: int
        """
        if self._size is not None:
            return self._size
        else:
            raise ValueError
