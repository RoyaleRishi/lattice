"""Importing this package registers all built-in adapters (spec §7.1)."""

from lattice.adapters.concept_store import in_memory  # noqa: F401
from lattice.adapters.dataset import inspec, toy  # noqa: F401
from lattice.adapters.document_metric import f1_at_k  # noqa: F401
from lattice.adapters.embedder import hashing  # noqa: F401
from lattice.adapters.extractor import token  # noqa: F401
from lattice.adapters.graph_integrator import in_memory as graph_in_memory  # noqa: F401
from lattice.adapters.metric import label_f1  # noqa: F401
from lattice.adapters.relation_inducer import co_occurrence  # noqa: F401
from lattice.adapters.resolver import exact_label  # noqa: F401
from lattice.adapters.scorer import frequency  # noqa: F401
from lattice.adapters.segmenter import block  # noqa: F401
