"""Attention-to-import and direct logit attribution for heads (M3).

API:
  attention_to_import(model, stimuli) -> list[dict]
  head_dla(model, stimuli, lex) -> list[dict]
"""

from __future__ import annotations

import torch

from .proxy import ProxyLexicon, proxy_pull

# names_filter for head DLA: needs hook_z (per query head) and ln_final.hook_scale.
# hook_pattern captured via run_with_hooks (not run_with_cache) to store only final row.
_DLA_NAMES_FILTER = lambda n: n.endswith("hook_z") or n == "ln_final.hook_scale"


def attention_to_import(model, stimuli) -> list[dict]:
    """Per stimulus x layer x head: attention mass from final position onto key groups.

    Captures hook_pattern via run_with_hooks, storing ONLY the final query row
    (pattern[:, :, -1, :]) inside the hook to avoid caching full O(seq^2) attention.

    Returned fields per record:
      attn_import_span  sum of pattern[-1, import_span_cols] over import token cols
      attn_use_alias    pattern[-1, use_alias_pos]
      attn_final_pos    pattern[-1, final_pos]  (self-attention)
    """
    records: list[dict] = []
    device = next(model.parameters()).device
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    for stim in stimuli:
        ids = torch.tensor([stim.token_ids], dtype=torch.long, device=device)
        # Collect final-row attention patterns per layer.
        # pattern_rows[layer] = [n_heads, seq_len] (final query position only)
        pattern_rows: dict[int, torch.Tensor] = {}

        def _make_hook(layer_idx):
            def hook_fn(pattern, hook):
                # pattern shape: [batch, n_heads, q_pos, k_pos]
                # Store final query row only; move to CPU to save GPU memory.
                pattern_rows[layer_idx] = pattern[0, :, -1, :].detach().cpu()
                return pattern
            return hook_fn

        fwd_hooks = [
            (f"blocks.{l}.attn.hook_pattern", _make_hook(l))
            for l in range(n_layers)
        ]
        with torch.no_grad():
            model.run_with_hooks(
                ids,
                prepend_bos=False,
                fwd_hooks=fwd_hooks,
            )

        import_cols = list(range(stim.import_span[0], stim.import_span[1]))
        for layer in range(n_layers):
            row = pattern_rows[layer]  # [n_heads, seq_len]
            for head in range(n_heads):
                h_row = row[head]  # [seq_len]
                attn_import = float(h_row[import_cols].sum().item()) if import_cols else 0.0
                attn_alias = float(h_row[stim.use_alias_pos].item())
                attn_final = float(h_row[stim.final_pos].item())
                records.append(
                    {
                        "stimulus_id": stim.stimulus_id,
                        "base_id": stim.base_id,
                        "pair_name": stim.pair_name,
                        "condition": stim.condition,
                        "depth": stim.depth,
                        "layer": layer,
                        "head": head,
                        "attn_import_span": attn_import,
                        "attn_use_alias": attn_alias,
                        "attn_final_pos": attn_final,
                    }
                )
    return records


def head_dla(model, stimuli, lex: ProxyLexicon) -> list[dict]:
    """Per stimulus x head: direct logit attribution (DLA) via mean-unembed direction.

    Direction: d = (W_U[:, prior_ids].mean(-1) - W_U[:, bound_ids].mean(-1))
    Note: this is a linearization of proxy_pull (logsumexp is nonlinear); mean-unembed
    is the standard DLA proxy used in mechanistic interpretability.

    Per-head contributions at the FINAL position are computed manually as
    z[final] @ W_O per head (b_O is a shared bias, excluded, matching the
    stack_head_results convention), then passed through the frozen final RMSNorm
    (divide by the cached ln_final scale at the final position, multiply by
    ln_final.w; models load unprocessed so w is not folded). This avoids
    cache.stack_head_results, whose z * W_O broadcast materialises a
    [pos, heads, d_head, d_model] intermediate (~2.4 GB per layer for 1.5B at
    depth 512) and OOMs the 8 GB GPU.
    Class: prior_promoting if contribution > 0, binding_promoting if < 0.
    """
    records: list[dict] = []
    device = next(model.parameters()).device
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    W_U = model.unembed.W_U  # [d_model, d_vocab]
    prior_ids = torch.tensor(lex.prior_ids, dtype=torch.long, device=device)
    bound_ids = torch.tensor(lex.bound_ids, dtype=torch.long, device=device)
    # direction: [d_model]
    direction = (
        W_U[:, prior_ids].float().mean(dim=-1) - W_U[:, bound_ids].float().mean(dim=-1)
    ).detach()

    for stim in stimuli:
        ids = torch.tensor([stim.token_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                ids,
                prepend_bos=False,
                names_filter=_DLA_NAMES_FILTER,
            )
            scale = cache["ln_final.hook_scale"][0, stim.final_pos]  # [1]
            w_ln = model.ln_final.w                                  # [d_model]
            per_layer = []
            for layer in range(n_layers):
                z_f = cache[f"blocks.{layer}.attn.hook_z"][0, stim.final_pos]  # [n_heads, d_head]
                W_O = model.blocks[layer].attn.W_O                             # [n_heads, d_head, d_model]
                per_layer.append(torch.einsum("hd,hdm->hm", z_f.to(W_O.dtype), W_O))
            head_results = torch.stack(per_layer)                    # [n_layers, n_heads, d_model]
            head_results = ((head_results / scale) * w_ln).float()

        # Also compute final proxy_pull (from actual logits for consistency check)
        with torch.no_grad():
            logits_all = model(ids, prepend_bos=False)  # [1, seq, d_vocab]
        final_pull = float(proxy_pull(logits_all[0, stim.final_pos, :].float(), lex).item())

        for layer in range(n_layers):
            for head in range(n_heads):
                h_result = head_results[layer, head, :]  # [d_model]
                contribution = float((h_result @ direction).item())
                head_class = "prior_promoting" if contribution > 0 else "binding_promoting"
                records.append(
                    {
                        "stimulus_id": stim.stimulus_id,
                        "base_id": stim.base_id,
                        "pair_name": stim.pair_name,
                        "condition": stim.condition,
                        "depth": stim.depth,
                        "layer": layer,
                        "head": head,
                        "dla_contribution": contribution,
                        "head_class": head_class,
                        "final_proxy_pull": final_pull,
                    }
                )
    return records
