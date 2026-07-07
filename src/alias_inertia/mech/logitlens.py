"""Logit-lens trajectories: proxy_pull at each accumulated-residual checkpoint (M1).

API:
  trajectory(model, stim, lex) -> np.ndarray [n_layers+1]
  run_m1(model, stimuli, lex) -> list[dict]
"""

from __future__ import annotations

import numpy as np
import torch

from .proxy import ProxyLexicon, proxy_pull
from .stimuli_mech import MechStimulus

# names_filter for logit-lens: needs resid_pre (for accumulated_resid layer 0) + resid_post
# + ln_final.hook_scale (apply_ln=True uses it). hook_z and hook_pattern NOT needed here.
_NAMES_FILTER = lambda n: "resid_pre" in n or "resid_post" in n or n == "ln_final.hook_scale"


def trajectory(model, stim: MechStimulus, lex: ProxyLexicon) -> np.ndarray:
    """Proxy_pull at every residual checkpoint, shape [n_layers+1].

    Index i (i < n_layers) = stream BEFORE block i (i.e. after embedding + i blocks applied).
    Index n_layers = final output stream (reproduces true final logits).

    Uses accumulated_resid(layer=-1, apply_ln=True, pos_slice=-1) which returns
    [n_layers+1, batch, d_model]; projects via model.unembed.W_U + b_U.
    Raw logits are used (not log_softmax): the additive shift cancels in proxy_pull.
    """
    device = next(model.parameters()).device
    ids = torch.tensor([stim.token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            ids,
            prepend_bos=False,
            names_filter=_NAMES_FILTER,
        )
        # [n_layers+1, 1, d_model]: checkpoint 0 = embed only; checkpoint n_layers = after
        # last block. Both the (lazy) accumulation and the unembed projection must stay
        # under no_grad: W_U requires grad and would otherwise attach a graph.
        resid_stack = cache.accumulated_resid(layer=-1, apply_ln=True, pos_slice=-1)
        W_U = model.unembed.W_U   # [d_model, d_vocab]
        b_U = model.unembed.b_U   # [d_vocab]
        # [n_layers+1, d_model] @ [d_model, d_vocab] -> [n_layers+1, d_vocab]
        logits = resid_stack[:, 0, :].to(W_U.dtype) @ W_U + b_U
        pulls = proxy_pull(logits, lex)   # [n_layers+1]
    return pulls.cpu().float().numpy()


def run_m1(model, stimuli, lex) -> list[dict]:
    """Compute logit-lens trajectories for all stimuli.

    Returns one record per stimulus: stimulus metadata + trajectory (list) + final_proxy_pull.
    """
    records: list[dict] = []
    for stim in stimuli:
        traj = trajectory(model, stim, lex)
        records.append(
            {
                "stimulus_id": stim.stimulus_id,
                "base_id": stim.base_id,
                "pair_name": stim.pair_name,
                "condition": stim.condition,
                "alias": stim.alias,
                "depth": stim.depth,
                "template_id": stim.template_id,
                "rep": stim.rep,
                "prompt_sha256": stim.prompt_sha256,
                "trajectory": traj.tolist(),
                "final_proxy_pull": float(traj[-1]),
                "n_checkpoints": len(traj),
            }
        )
    return records
