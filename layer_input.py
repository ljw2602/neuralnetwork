import numpy as np


class Input(object):

    def __init__(self, size_):
        self._size = size_
        self._a = None

    def set_a(self, a_):
        """
        Set a = x
        :param a_ : np.array
        :return: np.array
        """
        if isinstance(a_, np.ndarray):
            self._a = a_
            return
        else:
            raise TypeError

    def get_size(self):
        """
        Return size
        :return: int
        """
        if self._size is not None:
            return self._size
        else:
            raise ValueError

    def get_a(self):
        """
        Return a = sigma(z)
        :return: np.array
        """
        if self._a is not None:
            return self._a
        else:
            raise ValueError

    def feedforward(self, *args):
        return self.get_a()

    def backpropagate(self, *args):
        return