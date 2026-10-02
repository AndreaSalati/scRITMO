"""
Moved to :mod:`scritmo.ml.desync.results` (results tables, desync aggregation).
This module re-exports every name so old imports keep working.
"""
from .desync import results as _results

globals().update(
    {k: v for m in (_results,) for k, v in vars(m).items() if not k.startswith("__")}
)
