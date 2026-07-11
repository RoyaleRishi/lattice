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
    - Local significance (Eq. 6): R_l(c_i) = sum over j != i of
      (cos(H_ci, H_cj) - lambda * mu), mu = mean pairwise candidate
      similarity. Self-pairs are excluded (the paper is ambiguous; a
      self-pair adds the same constant to every candidate).
    - Final score (Eq. 7): R(c) = R_g(c) * R_l(c) * W(c); top_k unique
      surfaces by (-score, surface).

    Documented deviations: candidates, sentences, and documents are embedded
    as whole strings through the injected Embedder (paper: BERT token vectors
    + max-pooling); candidates come from the injected Extractor (paper:
    CoreNLP POS regex); word positions use whitespace tokens (paper: CoreNLP
    tokens). denoise_lambda defaults to the paper's Inspec-tuned 1.3 (§4.2)."""

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
        mu = sum(pair_sim.values()) / len(pair_sim) if pair_sim else 0.0
        local_sig = {
            s: sum(
                pair_sim[(min(s, other), max(s, other))] - self.denoise_lambda * mu
                for other in surfaces
                if other != s
            )
            for s in surfaces
        }

        salience = {
            s: global_sig[s] * local_sig[s] * candidate_weight[s] for s in surfaces
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
