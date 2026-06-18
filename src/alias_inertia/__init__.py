"""alias-inertia: a controlled behavioral probe of import-alias binding versus corpus prior.

Package modules:
  lexicons     discriminative continuation sets and swap-pair / canonical-alias definitions
  stimuli      programmatic minimal-pair generator (three binding conditions, token-depth filler)
  backends     teacher-forced scorers (hf, llamacpp) behind one interface
  scoring      disk-cached scoring wrapper
  metrics      the prior-pull metric
  determinism  seeds, environment fingerprinting, stable hashing
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import determinism, lexicons, metrics, scoring, stimuli  # noqa: F401

__all__ = ["determinism", "lexicons", "metrics", "scoring", "stimuli", "__version__"]
