import collections
from categorical_cross_entropy import CategoricalCrossEntropy
from mean_squared_error import MeanSquaredError

class Initializer(collections.Mapping):

    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]


loss = Initializer(categorical_crossentropy=CategoricalCrossEntropy(),
                   mse=MeanSquaredError())