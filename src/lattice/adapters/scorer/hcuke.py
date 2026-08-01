import math
from collections.abc import Sequence

from lattice.core.types import Mention, ScoredMention, Unit
from lattice.core.vectors import cosine
from lattice.ports import Embedder, Scorer
from lattice.registry.registry import register


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def _min_max(factor: dict[str, float]) -> dict[str, float]:
    """Min-max normalize one Eq. (7) factor to [0, 1] over the candidate set.

    A degenerate factor (every candidate equal, so a zero range) maps to 1.0
    rather than 0/0: it carries no ranking information, so it must leave the
    product unchanged instead of collapsing every score to zero."""
    if not factor:
        return {}
    lowest = min(factor.values())
    span = max(factor.values()) - lowest
    if span == 0.0:
        return dict.fromkeys(factor, 1.0)
    return {key: (value - lowest) / span for key, value in factor.items()}


@register(Scorer, "hcuke")
class HCUKEScorer(Scorer):
    """HCUKE (Xu et al., Knowledge-Based Systems 304 (2024) 112511):
    hierarchical context-aware unsupervised keyphrase extraction, implemented
    per the paper's Algorithm 1 with the pipeline's units as the sentence
    level (pair with the "sentence" segmenter for paper-faithful runs).

    - Position weights (Eq. 3): W(x) = softmax of 1/position over candidates
      (1-based document word position of the first occurrence, SIFRank-style)
      and over sentences (1-based order).
    - Global significance (Alg. 1 lines 8-13): R_g(c) = sum over sentences s
      containing c of W(s) * cos(H_s, H_d) * cos(H_c, H_s). Note: Eq. (5)'s
      prose would apply W(s) twice (it already sits inside Eq. (4)); Algorithm
      1 and the §3.3 worked example apply it once — we follow Algorithm 1.
    - Local significance (Eq. 6): R_l(c_i) = sum over ALL j (including j=i) of
      (cos(H_ci, H_cj) - lambda * mu), with mu = (1/n) sum_i (1/n) sum_j
      dist(H_ci, H_cj) as §3.4 defines it: the mean over all n^2 ORDERED
      pairs, unit diagonal included. Algorithm 1 line 15's inner loop has no
      guard excluding j=i, so the self-comparison (cos(H_ci,H_ci)=1) is
      included by design -- a constant per candidate that, because the final
      score is a product (Eq. 7), contributes a term proportional to that
      candidate's own global significance and position weight rather than
      cancelling out.
    - Normalization (not in the paper; see the deviations below): R_g and R_l
      are min-max normalized to [0, 1] across the document's candidates before
      Eq. (7) multiplies them, so that neither can be negative and neither
      dominates by raw scale. W(c) is deliberately left alone -- the asymmetry
      is the point, not an oversight. R_g and R_l are unbounded raw sums over
      sentences and candidate pairs, with no intrinsic scale; W(c) is already
      normalized, being a softmax (Eq. 3) and so a positive distribution over
      candidates summing to 1. Min-max stretching a softmax is not
      normalization but distortion: it would manufacture spread the paper
      never intended, inflating a signal Eq. (3) deliberately keeps weak and
      near-uniform, and handing the last-positioned candidate an exact 0. The
      rule is: normalize the unbounded factors, leave the bounded one as it is.
    - Final score (Eq. 7): R(c) = R_g(c) * R_l(c) * W(c), the first two
      normalized; top_k unique surfaces by (-score, surface).

    Documented deviations: candidates, sentences, and documents are embedded
    as whole strings through the injected Embedder (paper: BERT token vectors
    + max-pooling); candidates come from the injected Extractor (paper:
    CoreNLP POS regex); word positions use whitespace tokens (paper: CoreNLP
    tokens). denoise_lambda defaults to the paper's Inspec-tuned 1.3 (§4.2).
    The normalization step is ours in all but name: §3.4's only mention of it
    is the phrase "simple filtering and normalization operations" (a contrast
    with prior work's "complex filtering techniques"), and neither §3.4 nor
    Algorithm 1 says what is normalized or how -- Algorithm 1 line 19 forms
    the Eq. (7) product from the raw R_g, R_l and W_c. Min-max on the two
    unbounded factors is our reading, forced less by the text than by Eq. (7)
    being a product: Eq. (6) subtracts lambda*mu n times, so raw R_l is
    negative for most candidates at the paper's lambda=1.3, and a negative
    factor in a product ranks the least central candidates first. One
    consequence of normalizing R_l in particular is exact, not
    merely approximate, insensitivity to denoise_lambda: -lambda*mu is the
    same offset for every candidate, so min-max cancels it and neither lambda
    nor mu can change a score. The paper's Fig. 3 shows F1@10 on Inspec moving
    under one point as lambda sweeps 0 -> 1.5, so near-flatness is expected,
    but flat-to-the-bit is ours; denoise_lambda is kept for config
    compatibility and to keep the Eq. (6) trace visible."""

    def __init__(self, embedder: Embedder, top_k: int = 10, denoise_lambda: float = 1.3):
        self.embedder = embedder
        self.top_k = top_k
        self.denoise_lambda = denoise_lambda

    def score(
        self, mentions: Sequence[Mention], units: Sequence[Unit]
    ) -> list[ScoredMention]:
        if not mentions:
            return []
        surfaces = sorted({m.surface for m in mentions})
        document_text = "\n".join(unit.text for unit in units)
        vectors = self.embedder.embed(
            [document_text, *(unit.text for unit in units), *surfaces]
        )
        document_vector = vectors[0]
        unit_vectors = {u.id: v for u, v in zip(units, vectors[1 : 1 + len(units)])}
        surface_vectors = dict(zip(surfaces, vectors[1 + len(units) :]))

        first_position = self._first_word_positions(mentions, units)
        candidate_weight = dict(zip(surfaces, _softmax(
            [1.0 / first_position[s] if s in first_position else 0.0 for s in surfaces]
        )))
        sentence_weight = dict(zip(
            (u.id for u in units), _softmax([1.0 / (u.order + 1) for u in units])
        ))

        units_of_surface: dict[str, set[str]] = {}
        for m in mentions:
            if m.unit_id in unit_vectors:
                units_of_surface.setdefault(m.surface, set()).add(m.unit_id)
        global_sig = {
            surface: sum(
                sentence_weight[unit_id]
                * cosine(unit_vectors[unit_id], document_vector)
                * cosine(surface_vectors[surface], unit_vectors[unit_id])
                for unit_id in sorted(units_of_surface.get(surface, ()))
            )
            for surface in surfaces
        }

        pair_sim = {
            (a, b): cosine(surface_vectors[a], surface_vectors[b])
            for i, a in enumerate(surfaces)
            for b in surfaces[i + 1 :]
        }
        # Row i of the candidate similarity matrix: sum_j dist(H_ci, H_cj) over
        # all j, self-pair included at an exact 1.0 (no cosine(v, v) drift).
        row_total = {
            s: sum(
                1.0 if other == s else pair_sim[(min(s, other), max(s, other))]
                for other in surfaces
            )
            for s in surfaces
        }
        # mu (§3.4) is the mean over all n^2 ordered pairs, diagonal included.
        mu = sum(row_total.values()) / len(surfaces) ** 2
        # Eq. (6): R_l(c_i) = sum_j (dist(H_ci, H_cj) - lambda*mu) = the row
        # total less n identical offsets.
        local_sig = {
            s: total - len(surfaces) * self.denoise_lambda * mu
            for s, total in row_total.items()
        }

        # Normalize the two unbounded factors before the Eq. (7) product, so
        # that neither can be negative and neither dominates by raw scale. Not
        # in the paper -- see the class docstring's deviations ledger. W(c) is
        # passed through raw: it is already a softmax over candidates, so
        # min-max would distort a deliberately weak signal rather than
        # normalize it.
        norm_global = _min_max(global_sig)
        norm_local = _min_max(local_sig)
        salience = {
            s: norm_global[s] * norm_local[s] * candidate_weight[s] for s in surfaces
        }
        ranked = sorted(salience.items(), key=lambda kv: (-kv[1], kv[0]))
        top_surfaces = {surface for surface, _ in ranked[: self.top_k]}
        return [
            ScoredMention(
                mention=m, salience=salience[m.surface], selected=m.surface in top_surfaces
            )
            for m in mentions
        ]

    @staticmethod
    def _first_word_positions(
        mentions: Sequence[Mention], units: Sequence[Unit]
    ) -> dict[str, int]:
        """1-based document word position of each surface's first occurrence.
        Mentions pointing at units not present in `units` are skipped; a
        surface with no resolvable position falls back to a zero position
        score in the softmax (uniform weight in the degenerate case)."""
        unit_by_id = {u.id: u for u in units}
        unit_offset: dict[str, int] = {}
        offset = 0
        for u in sorted(units, key=lambda u: u.order):
            unit_offset[u.id] = offset
            offset += len(u.text.split())
        positions: dict[str, int] = {}
        for m in mentions:
            unit = unit_by_id.get(m.unit_id)
            if unit is None:
                continue
            pos = unit_offset[unit.id] + len(unit.text[: m.span[0]].split()) + 1
            positions[m.surface] = min(pos, positions.get(m.surface, pos))
        return positions
