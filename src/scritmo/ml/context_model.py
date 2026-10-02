"""
Historical home of the model class, now :mod:`scritmo.ml.model.scritmo`.

Kept because pickled models refer to ``scritmo.ml.context_model.Scritmo`` /
``ContextModel`` and because scripts import from here. Every name this module
used to provide is still available.
"""
from functools import partial  # noqa: F401
from tqdm import tqdm  # noqa: F401
from sklearn.preprocessing import OneHotEncoder  # noqa: F401

from scritmo import cstd2R, median_dispersion, w  # noqa: F401
from .model import scritmo as _moved
from .utils import circ_std_P, harmonic_dm_torch  # noqa: F401
from .marginalization import MarginalizationMixin  # noqa: F401
from .unspliced.fisher import FisherUncertaintyMixin  # noqa: F401
from .desync.results import desync_means, desync_results  # noqa: F401
from .desync.deconvolution import aggregate_technical_deconvolution  # noqa: F401

globals().update({k: v for k, v in vars(_moved).items() if not k.startswith("__")})

circSTD = partial(_moved.cSTD, adjust=True)
