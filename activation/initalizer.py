import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))))
from activation.relu import ReLu
from activation.sigmoid import Sigmoid

class Initializer(collections.Mapping):

    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]


activation = Initializer(relu=ReLu(), sigmoid=Sigmoid())