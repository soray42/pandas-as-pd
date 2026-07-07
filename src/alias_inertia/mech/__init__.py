"""Mechanistic-interpretability arm for alias-inertia (Phase M).

Sub-modules:
  env           MECH_MODELS, REPO_ROOT, get_tokenizer, load_model, names_filter_full
  stimuli_mech  MechStimulus, AlignmentError, build_mech_stimuli
  proxy         ProxyLexicon, build_proxy_lexicon, proxy_pull
  manifest      update_manifest, model_revision
  logitlens     logit-lens trajectories (M1)
  patching      activation patching (M2)
  heads         attention + DLA analysis (M3)
  ablate        head ablation (M4)
"""

from __future__ import annotations

from .env import MECH_MODELS, REPO_ROOT, get_tokenizer, load_model, names_filter_full
from .manifest import model_revision, update_manifest
from .proxy import ProxyLexicon, build_proxy_lexicon, proxy_pull
from .stimuli_mech import AlignmentError, MechStimulus, assert_triple_aligned, build_mech_stimuli

__all__ = [
    # env
    "MECH_MODELS",
    "REPO_ROOT",
    "get_tokenizer",
    "load_model",
    "names_filter_full",
    # stimuli_mech
    "AlignmentError",
    "MechStimulus",
    "assert_triple_aligned",
    "build_mech_stimuli",
    # proxy
    "ProxyLexicon",
    "build_proxy_lexicon",
    "proxy_pull",
    # manifest
    "model_revision",
    "update_manifest",
]
