"""Moved to :mod:`scritmo.ml.model.training`; this path stays as an alias."""
import sys

from .model import training as _moved

sys.modules[__name__] = _moved
