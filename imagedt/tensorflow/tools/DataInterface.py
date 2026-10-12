import numpy as np


# 数据
class DataSet:
    def __init__(self):
        super().__init__()
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
        if self._x is None:
            return 0
        if isinstance(self._x, list):
            return len(self._x)
        else:
            return self._x.shape[0]

    def set_x(self, x):
        self._x = x
        self.reset_step()

    def set_y(self, y):
        self._y = y

    def reset_step(self):
        self._index_in_epoch = 0

    def _validate_batch(self, batch_size):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

    def batch_count(self, batch_size, isFixed=True):
        self._validate_batch(batch_size)
        return (
            self.count // batch_size
            if isFixed
            else (self.count + batch_size - 1) // batch_size
        )

    def next_batch(self, batch_size, isFixed=True):
        self._validate_batch(batch_size)
        if self._y is None or self.count != len(self._y):
            raise ValueError("x and y must have equal length")
        if self.count == 0 or (isFixed and batch_size > self.count):
            raise ValueError("dataset is too small for requested batch")
        if self._index_in_epoch >= self.count or (
            isFixed and self._index_in_epoch + batch_size > self.count
        ):
            self.reset_step()
        if self._index_in_epoch == 0:
            perm = np.random.permutation(self.count)
            self._x = (
                [self._x[i] for i in perm]
                if isinstance(self._x, list)
                else self._x[perm]
            )
            self._y = (
                [self._y[i] for i in perm]
                if isinstance(self._y, list)
                else self._y[perm]
            )
        start = self._index_in_epoch
        end = min(start + batch_size, self.count)
        self._index_in_epoch = end
        return self._x[start:end], self._y[start:end]


# 数据集
class DataSets:
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
            raise ValueError("invalid argument")
        return self._x_shape

    def set_x_shape(self, x_shape):
        self._x_shape = x_shape

    @property
    def y_shape(self):
        if self._y_shape is None:
            raise ValueError("invalid argument")
        return self._y_shape

    def set_y_shape(self, y_shape):
        self._y_shape = y_shape
