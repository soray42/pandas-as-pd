"""MechStimulus dataclass and builder for the mechanistic arm.

Builds condition-triples (conventional / swapped / no_prior) where all three
conditions share the *same filler* so token positions outside the import line
are identical across conditions. This alignment is required for activation
patching (M2): patching positions in swapped -> no_prior would be meaningless
if the sequences differed at every filler position.

Key invariant (the "triple-alignment contract"):
  For each base_id, the swapped and no_prior stimuli have the same import-line
  token count, the same filler tokens at the same positions, and the same
  use_alias_pos (assuming the treatment alias and nonce alias are both single
  tokens under the model's tokenizer). AlignmentError is raised on violation;
  misaligned triples are skipped with a RuntimeWarning and counted.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from alias_inertia.determinism import sha256_text, stable_hash
from alias_inertia.lexicons import IMPORT_NAMES, SWAP_PAIRS
from alias_inertia.stimuli import (  # private helpers - same package
    _TEMPLATE_HEADERS,
    _build_filler,
    _rng_for,
)

# Mech study uses a single template (t1 = bare script, no header framing) to keep
# the filler and span computations unambiguous.
_TEMPLATE = "t1"

# Nonce alias for the no_prior condition.
_NONCE_ALIAS = "zz"

# Use-line assignment variables. Reps cycle through this pool so depth-0 bases
# differ (there is no filler to vary there); at depth > 0 the filler varies per
# rep as well. None of these may equal an alias, a nonce, or a library name.
_USE_VARS = (
    "x", "y", "v", "w", "u", "q", "r", "s", "t", "out",
    "res", "val", "tmp", "obj", "acc", "buf", "vec", "seq", "ret", "cur",
    "item", "node", "elem", "unit", "cell", "slot", "box", "tag", "tok", "dat",
)


def _render_use(header: str, import_line: str, filler: str, alias: str, use_var: str) -> str:
    """Assemble the prompt, ending exactly at ``{use_var} = {alias}.``."""
    tail = f"{use_var} = {alias}."
    if filler:
        return f"{header}{import_line}\n{filler}\n{tail}"
    return f"{header}{import_line}\n{tail}"


class AlignmentError(Exception):
    """Triple alignment failed: token_ids or spans differ where they must match."""


@dataclass
class MechStimulus:
    """One stimulus in the mech factorial design.

    base_id is shared across the three conditions for the same (pair, depth, rep);
    stimulus_id includes the condition. All three conditions in a base triple must
    have the same token_ids outside the import line (guaranteed by construction when
    the import lines tokenize to the same length).
    """

    base_id: str
    stimulus_id: str
    pair_name: str
    condition: str               # conventional | swapped | no_prior
    alias: str
    depth: int
    template_id: str             # "t1" (fixed for the mech arm)
    rep: int
    use_var: str                 # assignment variable in the use line
    prompt: str
    token_ids: list[int]
    import_span: tuple[int, int]   # [start, end) token indices of the import line
    filler_span: tuple[int, int]   # [start, end); (k, k) at depth 0
    use_alias_pos: int             # first token index of the alias in the use line
    final_pos: int                 # '.' token index == len(token_ids) - 1
    prompt_sha256: str


# ---------------------------------------------------------------------------
# Span-finding helpers
# ---------------------------------------------------------------------------

def _char_to_tok_range(offsets, cstart: int, cend: int) -> tuple[int, int] | None:
    """First and one-past-last token indices whose char span overlaps [cstart, cend).

    Returns None when no token overlaps the range (caller must handle).
    """
    ts = te = None
    for i, (cs, ce) in enumerate(offsets):
        if ce > cstart and cs < cend:
            ts = i if ts is None else ts
            te = i + 1
    if ts is None:
        return None
    return (ts, te)


def _build_spans(
    prompt: str,
    alias: str,
    header: str,
    import_line: str,
    token_ids: list[int],
    offsets,
    use_var: str,
) -> tuple[tuple[int, int], tuple[int, int], int, int]:
    """Compute (import_span, filler_span, use_alias_pos, final_pos) from char offsets.

    Prompt structure: ``{header}{import_line}\\n{filler}\\n{use_var} = {alias}.``
    or (depth=0):    ``{header}{import_line}\\n{use_var} = {alias}.``
    """
    n = len(token_ids)

    # Import line character range (excluding the trailing newline).
    imp_start_char = len(header)
    imp_end_char = imp_start_char + len(import_line)
    _imp = _char_to_tok_range(offsets, imp_start_char, imp_end_char)
    if _imp is None:
        raise AlignmentError(
            f"import_span empty: no tokens overlap char range [{imp_start_char},{imp_end_char})"
        )
    import_span: tuple[int, int] = _imp

    # Usage suffix is always "\\n{use_var} = {alias}." at the very end of the prompt.
    usage_suffix = f"\n{use_var} = {alias}."
    usage_nl_char = len(prompt) - len(usage_suffix)   # position of '\\n' before the use line

    # Filler: chars between the newline after the import line and the newline before usage.
    filler_c_start = imp_end_char + 1   # skip '\\n' after import line
    filler_c_end = usage_nl_char        # up to (not including) '\\n' before usage

    # Alias at usage: directly before the final dot.
    alias_usage_start_char = len(prompt) - 1 - len(alias)
    alias_usage_end_char = len(prompt) - 1   # the dot is the last char
    _alias_tok = _char_to_tok_range(offsets, alias_usage_start_char, alias_usage_end_char)
    if _alias_tok is None:
        raise AlignmentError(
            f"alias usage span empty: no tokens overlap char range "
            f"[{alias_usage_start_char},{alias_usage_end_char})"
        )
    use_alias_pos: int = _alias_tok[0]

    if filler_c_start >= filler_c_end:
        # depth=0: no filler; empty range pointing to the usage site.
        filler_span: tuple[int, int] = (use_alias_pos, use_alias_pos)
    else:
        _fill = _char_to_tok_range(offsets, filler_c_start, filler_c_end)
        filler_span = _fill if _fill is not None else (use_alias_pos, use_alias_pos)

    final_pos = n - 1   # '.' is always the last token

    # Sanity checks.
    assert use_alias_pos >= import_span[1], (
        f"use_alias_pos={use_alias_pos} < import_span end={import_span[1]}"
    )
    assert final_pos == n - 1, f"final_pos={final_pos} != n_tokens-1={n - 1}"

    return import_span, filler_span, use_alias_pos, final_pos


def _tokenize_with_offsets(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    return list(enc["input_ids"]), list(enc["offset_mapping"])


# ---------------------------------------------------------------------------
# Alignment checker
# ---------------------------------------------------------------------------

def assert_triple_aligned(
    conv: MechStimulus,
    swapped: MechStimulus,
    no_prior: MechStimulus,
) -> None:
    """Assert structural alignment across all three conditions of a base triple.

    Raises AlignmentError when:
      - token_ids lengths differ between swapped and no_prior
      - import_span, filler_span, use_alias_pos, or final_pos differ between swapped and no_prior
      - token_ids differ at positions outside import_span and use_alias_pos

    The conv argument is accepted for API symmetry but alignment is checked between
    swapped and no_prior (the pair required for M2 activation patching).
    """
    _ = conv  # accepted for API symmetry; swapped/no_prior are the critical pair
    if len(swapped.token_ids) != len(no_prior.token_ids):
        raise AlignmentError(
            f"token_ids length: swapped={len(swapped.token_ids)} "
            f"no_prior={len(no_prior.token_ids)} (base={swapped.base_id})"
        )
    if swapped.import_span != no_prior.import_span:
        raise AlignmentError(
            f"import_span: swapped={swapped.import_span} "
            f"no_prior={no_prior.import_span} (base={swapped.base_id})"
        )
    if swapped.filler_span != no_prior.filler_span:
        raise AlignmentError(
            f"filler_span: swapped={swapped.filler_span} "
            f"no_prior={no_prior.filler_span} (base={swapped.base_id})"
        )
    if swapped.use_alias_pos != no_prior.use_alias_pos:
        raise AlignmentError(
            f"use_alias_pos: swapped={swapped.use_alias_pos} "
            f"no_prior={no_prior.use_alias_pos} (base={swapped.base_id})"
        )
    if swapped.final_pos != no_prior.final_pos:
        raise AlignmentError(
            f"final_pos: swapped={swapped.final_pos} "
            f"no_prior={no_prior.final_pos} (base={swapped.base_id})"
        )
    # Element-wise check outside import_span and use_alias_pos (those positions carry alias tokens).
    imp_start, imp_end = swapped.import_span
    exempt: set[int] = set(range(imp_start, imp_end)) | {swapped.use_alias_pos}
    for i, (a, b) in enumerate(zip(swapped.token_ids, no_prior.token_ids)):
        if i not in exempt and a != b:
            raise AlignmentError(
                f"token_ids differ at position {i} (base={swapped.base_id}): "
                f"swapped={a} no_prior={b}"
            )


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_mech_stimuli(
    tokenizer,
    depths: tuple = (0, 512),
    n_per_cell: int = 30,
    seed: int = 20260618,
    pair_names: tuple[str, ...] | None = None,
) -> list[MechStimulus]:
    """Build the mech stimulus set, optionally restricted to a subset of pairs.

    Args:
        tokenizer: HF tokenizer with offset_mapping support.
        depths: filler token depths to generate (default: (0, 512)).
        n_per_cell: target reps per (pair, depth) cell (default: 30).
        seed: global seed for filler RNGs (default: 20260618).
        pair_names: if given, restrict to pairs whose name is in this set
            (e.g. ('numpy__pandas',)). Default None means all SWAP_PAIRS.

    For each (pair, depth, rep), filler is generated once using a
    condition-independent RNG so all three conditions share identical filler
    tokens. Import lines may still differ in token count for some pairs;
    those base triples are skipped with a RuntimeWarning. assert_triple_aligned
    is called on each surviving (swapped, no_prior) pair; AlignmentError triggers
    a warn-and-skip with the skip counted.

    Template is fixed to "t1" (bare script, no header). Base variety comes from
    the use-line variable pool (_USE_VARS, cycled by rep) plus per-rep filler at
    depth > 0; at depth 0 reps are capped at len(_USE_VARS) so no two bases are
    identical. Nonce alias for the no_prior condition is "zz".
    """
    template_id = _TEMPLATE
    header = _TEMPLATE_HEADERS[template_id]

    _pair_filter: set[str] | None = set(pair_names) if pair_names is not None else None

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    results: list[MechStimulus] = []
    n_align_skipped = 0

    for pair in SWAP_PAIRS:
        if _pair_filter is not None and pair.name not in _pair_filter:
            continue
        prior_lib = pair.prior_lib
        other_lib = pair.other_lib
        treatment_alias = pair.treatment_alias

        # Map condition -> (import_lib, alias)
        cond_params: dict[str, tuple[str, str]] = {
            "conventional": (prior_lib, treatment_alias),
            "swapped":      (other_lib, treatment_alias),
            "no_prior":     (other_lib, _NONCE_ALIAS),
        }

        for depth in depths:
            # Depth 0 has no filler to vary, so its variety comes entirely from
            # the use-line variable pool; cap reps to avoid duplicate prompts.
            reps = n_per_cell if depth > 0 else min(n_per_cell, len(_USE_VARS))
            for rep in range(reps):
                use_var = _USE_VARS[rep % len(_USE_VARS)]
                base_id = stable_hash(
                    [seed, pair.name, depth, template_id, rep, use_var], length=12
                )

                # Generate filler once per base using the conventional import line
                # as the reference for token-depth measurement.
                ref_import_line = (
                    f"import {IMPORT_NAMES[prior_lib]} as {treatment_alias}"
                )
                filler_rng = _rng_for(seed, pair.name, depth, rep)
                filler, _depth_actual = _build_filler(
                    header=header,
                    import_line=ref_import_line,
                    alias=treatment_alias,
                    depth_tokens=depth,
                    count_tokens=count_tokens,
                    rng=filler_rng,
                    tolerance=24,
                )

                # Build prompts for all conditions with the shared filler.
                cond_prompts: dict[str, tuple[str, str, str]] = {}
                for cond, (imp_lib, alias) in cond_params.items():
                    imp_line = f"import {IMPORT_NAMES[imp_lib]} as {alias}"
                    prompt = _render_use(header, imp_line, filler, alias, use_var)
                    cond_prompts[cond] = (imp_line, alias, prompt)

                # --- Alignment check: swapped vs no_prior import line lengths ---
                sw_imp_line = cond_prompts["swapped"][0]
                np_imp_line = cond_prompts["no_prior"][0]
                sw_imp_n = count_tokens(sw_imp_line)
                np_imp_n = count_tokens(np_imp_line)

                if sw_imp_n != np_imp_n:
                    warnings.warn(
                        f"Skipping {pair.name} depth={depth} rep={rep}: "
                        f"swapped import line has {sw_imp_n} tokens, "
                        f"no_prior has {np_imp_n} tokens (misaligned).",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue

                # --- Verify total sequence lengths match for swapped / no_prior ---
                sw_ids, _ = _tokenize_with_offsets(tokenizer, cond_prompts["swapped"][2])
                np_ids, _ = _tokenize_with_offsets(tokenizer, cond_prompts["no_prior"][2])
                if len(sw_ids) != len(np_ids):
                    warnings.warn(
                        f"Skipping {pair.name} depth={depth} rep={rep}: "
                        f"swapped has {len(sw_ids)} tokens, no_prior has {len(np_ids)} "
                        f"(length mismatch despite equal import line counts).",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue

                # --- Build one MechStimulus per condition ---
                built: dict[str, MechStimulus] = {}
                skip_triple = False
                for cond, (imp_line, alias, prompt) in cond_prompts.items():
                    ids, offsets = _tokenize_with_offsets(tokenizer, prompt)
                    try:
                        import_span, filler_span, use_alias_pos, final_pos = _build_spans(
                            prompt, alias, header, imp_line, ids, offsets, use_var
                        )
                    except AlignmentError as exc:
                        warnings.warn(
                            f"Skipping {pair.name} depth={depth} rep={rep} cond={cond}: "
                            f"span error: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        skip_triple = True
                        break

                    stimulus_id = stable_hash(
                        [base_id, cond, alias, pair.name, depth, rep, sha256_text(prompt)],
                        length=16,
                    )
                    built[cond] = MechStimulus(
                        base_id=base_id,
                        stimulus_id=stimulus_id,
                        pair_name=pair.name,
                        condition=cond,
                        alias=alias,
                        depth=depth,
                        template_id=template_id,
                        rep=rep,
                        use_var=use_var,
                        prompt=prompt,
                        token_ids=ids,
                        import_span=import_span,
                        filler_span=filler_span,
                        use_alias_pos=use_alias_pos,
                        final_pos=final_pos,
                        prompt_sha256=sha256_text(prompt),
                    )

                if skip_triple:
                    n_align_skipped += 1
                    continue

                # --- assert_triple_aligned on the full triple ---
                try:
                    assert_triple_aligned(built["conventional"], built["swapped"], built["no_prior"])
                except AlignmentError as exc:
                    warnings.warn(
                        f"Skipping {pair.name} depth={depth} rep={rep}: "
                        f"alignment check failed: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    n_align_skipped += 1
                    continue

                results.extend(built.values())

    if n_align_skipped:
        warnings.warn(
            f"build_mech_stimuli: {n_align_skipped} base triple(s) skipped due to alignment failures.",
            RuntimeWarning,
            stacklevel=1,
        )
    return results
