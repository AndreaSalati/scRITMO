"""
Moved to :mod:`scritmo.ml.tools.null_model`. This module re-exports every
name from it, and keeps ``NullModelMixin`` (now delegating to the plain
functions there via :mod:`scritmo.ml.tools._compat`) so old imports keep
working.
"""
from .tools import null_model as _moved
from .tools._compat import NullModelMixin

globals().update(
    {k: v for k, v in vars(_moved).items() if not k.startswith("__")}
)
