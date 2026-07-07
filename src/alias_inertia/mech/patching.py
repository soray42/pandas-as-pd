"""Activation patching experiments (M2).

Patches resid_post at one layer+group from a source run into a destination run
and measures the resulting proxy_pull shift.

API:
  GROUPS: tuple[str, ...]
  positions_for(stim, group) -> list[int]
  patch_run(model, dst_stim, src_resids, layer, group, lex) -> float
  run_m2(model, stimuli, lex, directions) -> list[dict]
"""

from __future__ import annotations

from typing import Sequence

import torch

from .proxy import ProxyLexicon, proxy_pull
from .stimuli_mech import MechStimulus

# Canonical group names for position sets used in patching experiments.
GROUPS = ("import_span", "use_alias", "final_pos", "filler_span")

# names_filter for source caching: resid_post all layers + ln_final.hook_scale.
_SRC_NAMES_FILTER = lambda n: "resid_post" in n or n == "ln_final.hook_scale"

# names_filter for final proxy_pull of the destination run (logits only; no cache needed).
# run_with_hooks does not use names_filter; we only need logits output.


def positions_for(stim: MechStimulus, group: str) -> list[int]:
    """Token positions for a named group in a stimulus.

    Groups:
      import_span  token indices of the import line (incl. alias token in it)
      use_alias    the alias token in the use line ('{alias}' before '.')
      final_pos    the final '.' token (== len(token_ids) - 1)
      filler_span  filler tokens (empty list at depth 0)
    """
    if group == "import_span":
        return list(range(stim.import_span[0], stim.import_span[1]))
    if group == "use_alias":
        return [stim.use_alias_pos]
    if group == "final_pos":
        return [stim.final_pos]
    if group == "filler_span":
        start, end = stim.filler_span
        return list(range(start, end))
    raise ValueError(f"unknown group {group!r}; expected one of {GROUPS}")


def _cache_source_resids(model, stim: MechStimulus) -> dict[int, torch.Tensor]:
    """Run model on stim and return resid_post for every layer as CPU fp16 tensors.

    Dict key = layer index (0-based). Tensor shape = [seq_len, d_model].
    Caching to CPU avoids holding all layers on GPU simultaneously.
    """
    device = next(model.parameters()).device
    ids = torch.tensor([stim.token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            ids,
            prepend_bos=False,
            names_filter=_SRC_NAMES_FILTER,
        )
    n_layers = model.cfg.n_layers
    resids: dict[int, torch.Tensor] = {}
    for layer in range(n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        # cache[key] shape: [1, seq_len, d_model]; squeeze batch dim, move to CPU
        resids[layer] = cache[key][0].to(dtype=torch.float16, device="cpu")
    return resids


def _run_dst_get_pull(
    model,
    dst_stim: MechStimulus,
    src_resids: dict[int, torch.Tensor],
    layer: int,
    positions: list[int],
    lex: ProxyLexicon,
) -> float:
    """Run the destination stimulus with one patched layer and return proxy_pull at final_pos."""
    device = next(model.parameters()).device
    ids = torch.tensor([dst_stim.token_ids], dtype=torch.long, device=device)
    patch_tensor = src_resids[layer].to(device=device)  # [seq_len, d_model]

    def hook_fn(activations, hook):
        # activations shape: [1, seq_len, d_model]
        for pos in positions:
            activations[0, pos, :] = patch_tensor[pos, :]
        return activations

    hook_name = f"blocks.{layer}.hook_resid_post"
    with torch.no_grad():
        logits = model.run_with_hooks(
            ids,
            prepend_bos=False,
            fwd_hooks=[(hook_name, hook_fn)],
        )
    # logits shape: [1, seq_len, d_vocab]; take final position
    final_logits = logits[0, dst_stim.final_pos, :].float()
    return float(proxy_pull(final_logits, lex).item())


def patch_run(
    model,
    dst_stim: MechStimulus,
    src_resids: dict[int, torch.Tensor],
    layer: int,
    group: str,
    lex: ProxyLexicon,
) -> float:
    """Patch resid_post at `layer` and `group` positions from src into dst; return proxy_pull.

    src_resids: dict {layer_idx: [seq_len, d_model] CPU fp16 tensor} from the source run.
    Patches ONLY at the specified group's positions; all other positions use dst activations.
    Returns proxy_pull (raw logit logsumexp contrast) at dst_stim.final_pos.
    """
    positions = positions_for(dst_stim, group)
    if not positions:
        # empty filler at depth 0: no positions to patch, return unpatched pull
        return _get_pull_unpatched(model, dst_stim, lex)
    return _run_dst_get_pull(model, dst_stim, src_resids, layer, positions, lex)


def _get_pull_unpatched(model, stim: MechStimulus, lex: ProxyLexicon) -> float:
    """Run stim without any patch; return proxy_pull at final_pos."""
    device = next(model.parameters()).device
    ids = torch.tensor([stim.token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(ids, prepend_bos=False)
    final_logits = logits[0, stim.final_pos, :].float()
    return float(proxy_pull(final_logits, lex).item())


def run_m2(
    model,
    stimuli,
    lex: ProxyLexicon,
    directions: Sequence[str] = ("noprior_to_swapped", "swapped_to_noprior"),
) -> list[dict]:
    """Per-layer per-group activation patching between swapped and no_prior stimuli.

    Pairs swapped and no_prior stimuli by base_id (guaranteed token-aligned within a base).

    fraction_restored sign convention:
      fraction_restored = (pull_dst - pull_patched) / (pull_dst - pull_src)
      0 = patch has no effect (pull_patched == pull_dst)
      1 = patch fully restores source value (pull_patched == pull_src)
      Values outside [0,1] indicate the patch overshot.

    Each record includes raw pulls (pull_src, pull_dst, pull_patched) in addition to the
    fraction, so downstream code can recompute or choose alternative normalizations.
    """
    from collections import defaultdict

    # Index stimuli by (base_id, condition).
    by_base: dict[str, dict[str, MechStimulus]] = defaultdict(dict)
    for stim in stimuli:
        if stim.condition in ("swapped", "no_prior"):
            by_base[stim.base_id][stim.condition] = stim

    # Keep only complete pairs.
    paired_bases = [
        bid for bid, cond_map in by_base.items()
        if "swapped" in cond_map and "no_prior" in cond_map
    ]

    n_layers = model.cfg.n_layers
    records: list[dict] = []

    for base_id in paired_bases:
        stim_swapped = by_base[base_id]["swapped"]
        stim_noprior = by_base[base_id]["no_prior"]

        # Cache unpatched pulls once per item.
        pull_swapped = _get_pull_unpatched(model, stim_swapped, lex)
        pull_noprior = _get_pull_unpatched(model, stim_noprior, lex)

        # Cache source residuals once per source item (all layers, to CPU fp16).
        src_resids_swapped: dict[int, torch.Tensor] | None = None
        src_resids_noprior: dict[int, torch.Tensor] | None = None

        for direction in directions:
            if direction == "noprior_to_swapped":
                src_stim, dst_stim = stim_noprior, stim_swapped
                pull_src, pull_dst = pull_noprior, pull_swapped
                if src_resids_noprior is None:
                    src_resids_noprior = _cache_source_resids(model, stim_noprior)
                src_resids = src_resids_noprior
            elif direction == "swapped_to_noprior":
                src_stim, dst_stim = stim_swapped, stim_noprior
                pull_src, pull_dst = pull_swapped, pull_noprior
                if src_resids_swapped is None:
                    src_resids_swapped = _cache_source_resids(model, stim_swapped)
                src_resids = src_resids_swapped
            else:
                raise ValueError(f"unknown direction {direction!r}")

            denom = pull_dst - pull_src
            # Avoid zero division if src == dst (self-patch / identical items).
            denom_safe = denom if abs(denom) > 1e-6 else float("nan")

            for layer in range(n_layers):
                for group in GROUPS:
                    pull_patched = patch_run(model, dst_stim, src_resids, layer, group, lex)
                    if abs(denom_safe) != abs(denom_safe):  # nan check
                        frac = float("nan")
                    else:
                        frac = (pull_dst - pull_patched) / denom_safe
                    records.append(
                        {
                            "base_id": base_id,
                            "direction": direction,
                            "layer": layer,
                            "group": group,
                            "pair_name": dst_stim.pair_name,
                            "depth": dst_stim.depth,
                            "template_id": dst_stim.template_id,
                            "rep": dst_stim.rep,
                            "src_stimulus_id": src_stim.stimulus_id,
                            "dst_stimulus_id": dst_stim.stimulus_id,
                            "pull_src": pull_src,
                            "pull_dst": pull_dst,
                            "pull_patched": pull_patched,
                            "fraction_restored": frac,
                        }
                    )
    return records
