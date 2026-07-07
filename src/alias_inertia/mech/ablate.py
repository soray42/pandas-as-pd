"""Head ablation experiments (M4).

Scales hook_z for specified (layer, head) pairs by a factor at ALL positions,
then reports proxy_pull per stimulus.

API:
  ablate_heads(model, stimuli, heads, factor, lex) -> list[dict]
"""

from __future__ import annotations

import torch

from .proxy import ProxyLexicon, proxy_pull


def ablate_heads(
    model,
    stimuli,
    heads: list[tuple[int, int]],
    factor: float,
    lex: ProxyLexicon,
) -> list[dict]:
    """Scale hook_z for the listed (layer, head) pairs by `factor` at ALL positions.

    factor=0.0 zeroes the head output (full ablation).
    factor=0.5 halves it (partial ablation).
    Returns proxy_pull at the final position per stimulus.

    hook_z shape: [batch, pos, n_heads, d_head] (per QUERY head; GQA already expanded).
    Hooks are set per unique layer to avoid redundant hook registrations.

    Caller should run this on SWAPPED stimuli (to measure the effect on prior-pull)
    and on CONVENTIONAL stimuli (collateral damage check: correct behavior should not
    degrade significantly when ablating prior-promoting heads).
    """
    # Group targeted head indices by layer for efficient hook building.
    heads_by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        heads_by_layer.setdefault(layer, []).append(head)

    records: list[dict] = []
    device = next(model.parameters()).device

    for stim in stimuli:
        ids = torch.tensor([stim.token_ids], dtype=torch.long, device=device)

        def _make_ablate_hook(head_indices: list[int], _factor: float = factor):
            def hook_fn(z, hook):
                # z shape: [batch, pos, n_heads, d_head]
                for h in head_indices:
                    z[:, :, h, :] = z[:, :, h, :] * _factor
                return z
            return hook_fn

        fwd_hooks = [
            (f"blocks.{layer}.attn.hook_z", _make_ablate_hook(h_list))
            for layer, h_list in heads_by_layer.items()
        ]
        with torch.no_grad():
            logits = model.run_with_hooks(
                ids,
                prepend_bos=False,
                fwd_hooks=fwd_hooks,
            )
        # logits shape: [1, seq_len, d_vocab]
        final_logits = logits[0, stim.final_pos, :].float()
        pull = float(proxy_pull(final_logits, lex).item())

        records.append(
            {
                "stimulus_id": stim.stimulus_id,
                "base_id": stim.base_id,
                "pair_name": stim.pair_name,
                "condition": stim.condition,
                "depth": stim.depth,
                "template_id": stim.template_id,
                "rep": stim.rep,
                "prompt_sha256": stim.prompt_sha256,
                "ablated_heads": [(int(l), int(h)) for l, h in heads],
                "factor": factor,
                "proxy_pull": pull,
            }
        )
    return records
