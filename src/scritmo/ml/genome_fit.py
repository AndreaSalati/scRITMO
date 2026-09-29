"""
Moved to :mod:`scritmo.ml.tools.genome_fit`. This module re-exports every
name from it, and keeps ``GenomeFitMixin`` (now delegating to the plain
functions there via :mod:`scritmo.ml.tools._compat`) so old imports keep
working.
"""
from .tools import genome_fit as _moved
from .tools._compat import GenomeFitMixin

globals().update(
    {k: v for k, v in vars(_moved).items() if not k.startswith("__")}
)
