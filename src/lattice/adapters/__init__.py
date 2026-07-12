"""Importing this package registers all built-in adapters (spec §7.1)."""

from lattice.adapters.concept_store import in_memory  # noqa: F401
from lattice.adapters.dataset import (  # noqa: F401
    inspec,
    mention_clusters,
    taxonomy,
    toy,
)
from lattice.adapters.document_metric import clustering, f1_at_k  # noqa: F401
from lattice.adapters.embedder import hashing, sentence_transformer  # noqa: F401
from lattice.adapters.extractor import (  # noqa: F401
    gazetteer,
    gold_mentions,
    noun_chunk,
    token,
)
from lattice.adapters.graph_integrator import in_memory as graph_in_memory  # noqa: F401
from lattice.adapters.metric import edge_f1, label_f1  # noqa: F401
from lattice.adapters.relation_inducer import (  # noqa: F401
    co_occurrence,
    compound,
    hearst,
    union,
)
from lattice.adapters.resolver import embedding_nn, exact_label  # noqa: F401
from lattice.adapters.scorer import (  # noqa: F401
    embedding_cosine,
    frequency,
    hcuke,
    mderank,
    passthrough,
)
from lattice.adapters.segmenter import block, sentence  # noqa: F401
