"""
The Scritmo model: the class (:mod:`.scritmo`), its likelihood maths
(:mod:`.likelihood`), gene-parameter transforms (:mod:`.parameters`) and
training (:mod:`.training`).
"""
from .scritmo import Scritmo, ContextModel
from .training import train_ritmo, fit_model, warmup_and_train
