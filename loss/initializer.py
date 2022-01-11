import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))))
from loss.mean_squared_error import MeanSquaredError
from loss.binary_cross_entropy import BinaryCrossEntropy

class Initializer(collections.Mapping):

    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]


loss = Initializer(mse=MeanSquaredError(), binary_crossentropy=BinaryCrossEntropy())