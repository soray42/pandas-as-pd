"""Mech manifest: read-merge-write mech/results/mech_manifest.json.

update_manifest(section, payload) atomically merges a new section into
``mech/results/mech_manifest.json``, adding environment fingerprint and
transformer_lens version for reproducibility.
"""

from __future__ import annotations

import json
import os

from .env import REPO_ROOT
from alias_inertia.determinism import environment_fingerprint, utc_now_iso


def _manifest_path() -> str:
    return os.path.join(REPO_ROOT, "mech", "results", "mech_manifest.json")


def model_revision(name: str) -> str:
    """Return the HF cache snapshot hash for a model, or 'unknown'."""
    try:
        import huggingface_hub as hh  # noqa: PLC0415

        info = hh.model_info(name)
        return info.sha or "unknown"
    except Exception:
        return "unknown"


def update_manifest(section: str, payload: dict) -> None:
    """Read-merge-write mech_manifest.json, adding environment fingerprint and timestamp."""
    path = _manifest_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    manifest: dict = {}
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (json.JSONDecodeError, OSError):
            manifest = {}
    entry = dict(payload)
    entry["environment"] = environment_fingerprint()
    entry["timestamp_utc"] = utc_now_iso()
    try:
        import importlib.metadata as _md  # noqa: PLC0415

        entry["transformer_lens_version"] = _md.version("transformer_lens")
    except Exception:
        entry["transformer_lens_version"] = "unknown"
    # Union-merge list-valued provenance keys so per-model incremental runs do not
    # erase each other's record (the manifest must describe ALL data on disk).
    prior_entry = manifest.get(section)
    if isinstance(prior_entry, dict):
        for key in ("models",):
            old = prior_entry.get(key)
            new = entry.get(key)
            if isinstance(old, list) and isinstance(new, list):
                entry[key] = sorted(set(old) | set(new))
    manifest[section] = entry
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
