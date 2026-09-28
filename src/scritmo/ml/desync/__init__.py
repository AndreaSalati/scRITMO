"""
Biological phase desynchrony and its technical (inference) floor.

- :func:`estimate_phase_desynchrony` — end-to-end pipeline on a fitted model.
- :mod:`.results` — per-cell results table and the desync aggregation
  (:func:`create_results_dataframe`, :func:`desync_results`, :func:`desync_means`).
- :mod:`.technical_sim` — the synchronized "technical twin" simulations.
- :mod:`.harmonic_floor`, :mod:`.deconvolution` — the two phase-resolved floors.
"""
from .results import (
    create_results_dataframe,
    desync_results,
    desync_means,
    aggregate_real_results,
    aggregate_simulated_results,
    append_first_timepoint_periodic,
)
from .harmonic_floor import aggregate_technical_harmonic
from .deconvolution import aggregate_technical_deconvolution
from .technical_sim import simulate_cell_populations, simulate_technical_grid
from .estimate import estimate_phase_desynchrony
