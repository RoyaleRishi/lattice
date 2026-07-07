"""Contract every Resolver adapter must satisfy. The heart of lattice
(spec §1): the same surface in different documents must resolve to the SAME
concept, with identity provenance preserved."""

from lattice.ports import Resolver
from tests.helpers import make_document, make_scored_mention


class ResolverContract:
    def make_resolver(self) -> Resolver:
        raise NotImplementedError(
            "subclass must provide a fully wired adapter (own embedder + store)"
        )

    def test_new_surface_creates_new_concept(self):
        resolver = self.make_resolver()
        [resolution] = resolver.resolve(
            [make_scored_mention(surface="vector store")], make_document(id="d1")
        )
        assert resolution.is_new
        assert resolution.concept.first_seen == "d1"
        assert resolution.concept.updated_at == "d1"

    def test_same_surface_across_documents_resolves_to_one_concept(self):
        resolver = self.make_resolver()
        [r1] = resolver.resolve(
            [make_scored_mention(surface="vector store", unit_id="d1:u0")],
            make_document(id="d1"),
        )
        [r2] = resolver.resolve(
            [make_scored_mention(surface="vector store", unit_id="d2:u0")],
            make_document(id="d2"),
        )
        assert r2.concept.id == r1.concept.id
        assert r1.is_new and not r2.is_new
        assert r2.concept.first_seen == "d1"
        assert r2.concept.updated_at == "d2"

    def test_empty_input_yields_no_resolutions(self):
        assert self.make_resolver().resolve([], make_document(id="d1")) == []
