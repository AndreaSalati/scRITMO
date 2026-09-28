"""
Moved to :mod:`scritmo.ml.desync.results` (results tables, desync aggregation)
and :mod:`scritmo.ml.desync.harmonic_floor` (harmonic technical floor). This
module re-exports every name from both so old imports keep working.
"""
from .desync import harmonic_floor as _harmonic_floor
from .desync import results as _results

globals().update(
    {k: v for m in (_results, _harmonic_floor) for k, v in vars(m).items() if not k.startswith("__")}
)
