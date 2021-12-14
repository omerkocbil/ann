import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))))
from init_omer.uniform_weight import Uniform
from init_omer.zero_bias import Zero

class Initializer(collections.Mapping):

    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]


initializer = Initializer(uniform=Uniform(), zero=Zero())