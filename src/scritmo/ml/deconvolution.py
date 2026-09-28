"""Moved to :mod:`scritmo.ml.desync.deconvolution`; this path stays as an alias."""
import sys

from .desync import deconvolution as _moved

sys.modules[__name__] = _moved
