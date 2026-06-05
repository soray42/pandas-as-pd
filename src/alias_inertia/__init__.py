"""alias-inertia: a controlled behavioral probe of import-alias binding vs. corpus prior.

See ``alias-inertia_scope.md`` for the full design. This package provides:
  lexicons   - discriminative continuation sets + swap-pair / canonical-alias definitions
  stimuli    - programmatic minimal-pair generator (3 binding conditions, token-depth filler)
  backends   - teacher-forced scorers (hf | llamacpp), one interface
  scoring    - disk-cached scoring wrapper
  metrics    - the prior-pull metric
  determinism- seeds, environment fingerprinting, stable hashing (reproducibility core)
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import determinism, lexicons, metrics, scoring, stimuli  # noqa: F401

__all__ = ["determinism", "lexicons", "metrics", "scoring", "stimuli", "__version__"]
