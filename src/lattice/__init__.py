"""lattice: concept-memory engine — documents in, an accreting normalized
concept graph out. The public contract is __all__ (M6 spec §3); everything
else may change without notice pre-1.0."""

from lattice.core.types import (
    Concept,
    Document,
    GraphDelta,
    GraphSnapshot,
    Relation,
)
from lattice.engine import Engine
from lattice.graph_view import GraphView

__version__ = "0.2.0"

__all__ = [
    "Concept",
    "Document",
    "Engine",
    "GraphDelta",
    "GraphSnapshot",
    "GraphView",
    "Relation",
    "__version__",
]
