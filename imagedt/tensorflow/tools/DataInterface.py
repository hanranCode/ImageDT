# coding: utf-8
import numpy as np


# 数据
class DataSet(object):
    def __init__(self):
        super(DataSet, self).__init__()
        self._x = None
        self._y = None
        self._index_in_epoch = 0

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def count(self):
        if isinstance(self._x, list):
            return len(self._x)
        else:
            return self._x.shape[0]

    def set_x(self, x):
        self._x = x

    def set_y(self, y):
        self._y = y

    def reset_step(self):
        self._index_in_epoch = 0

    def batch_count(self, batch_size, isFixed=True):
        if (self.count % batch_size) != 0 and not isFixed:
            return int(self.count / batch_size) + 1
        else:
            return int(self.count / batch_size)

    def next_batch(self, batch_size, isFixed=True):
            start = self._index_in_epoch
            end = self._index_in_epoch + batch_size
            if self._index_in_epoch == 0:
                perm = np.arange(self.count)
                np.random.shuffle(perm)
                if isinstance(self._x, list):
                    self._x = [self._x[i] for i in perm]
                else:
                    self._x = self._x[perm]
                if isinstance(self._y, list):
                    self._y = [self._y[i] for i in perm]
                else:
                    self._y = self._y[perm]

            self._index_in_epoch += batch_size

            if end > self.count:
                if isFixed:
                    self._index_in_epoch = 0
                    return self.next_batch(batch_size, isFixed)
                else:
                        self._index_in_epoch = 0
                        end = self.count

            return self._x[start:end], self._y[start:end]

# 数据集
class DataSets(object):
    def __init__(self):
        self._train = DataSet()
        self._validate = DataSet()
        self._test = DataSet()
        self._x_shape = None
        self._y_shape = None

    @property
    def test(self):
        return self._test

    @property
    def validate(self):
        return self._validate

    @property
    def train(self):
        return self._train

    @property
    def x_shape(self):
        if self._x_shape is None:
            raise ValueError('invalid argument')
        return self._x_shape

    def set_x_shape(self, x_shape):
        self._x_shape = x_shape

    @property
    def y_shape(self):
        if self._y_shape is None:
            raise ValueError('invalid argument')
        return self._y_shape

    def set_y_shape(self, y_shape):
        self._y_shape = y_shape
