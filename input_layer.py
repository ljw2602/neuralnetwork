class InputLayer(object):

    def __init__(self, size_):
        self._size = size_
        self._x = None

    def size(self):
        """
        Return size
        :return: int
        """
        return self._size

    def setx(self, x_):
        """
        Set x
        :return:
        """
        self._x = x_
        return

    def feedforward(self):
        """
        Return x
        :return: np.array
        """
        if self._x is not None:
            return self._x
        else:
            raise ValueError

    def a(self):
        """
        Return a = x
        :return: np.array
        """
        if self._x is not None:
            return self._x
        else:
            raise ValueError


if __name__ == "__main__":
    import numpy as np

    x = np.array([-2, 0, 2])

    q = InputLayer(3)
    q.setx(x)
    print(q.a())
