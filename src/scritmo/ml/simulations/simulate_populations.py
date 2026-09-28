"""Moved to :mod:`scritmo.ml.desync.technical_sim`; this path stays as an alias."""
import sys

from ..desync import technical_sim as _moved

sys.modules[__name__] = _moved
