from lattice.core.types import GraphSnapshot
from lattice.ports import Metric
from lattice.registry.registry import register


def _tarjan_sccs(
    nodes: list[str], adjacency: dict[str, list[str]]
) -> list[list[str]]:
    """Iterative Tarjan (recursion-free: real IS_A chains can be deep)."""
    index_of: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    sccs: list[list[str]] = []
    counter = 0
    for root in nodes:
        if root in index_of:
            continue
        work = [(root, iter(adjacency.get(root, ())))]
        index_of[root] = lowlink[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, neighbours = work[-1]
            advanced = False
            for neighbour in neighbours:
                if neighbour not in index_of:
                    index_of[neighbour] = lowlink[neighbour] = counter
                    counter += 1
                    stack.append(neighbour)
                    on_stack.add(neighbour)
                    work.append((neighbour, iter(adjacency.get(neighbour, ()))))
                    advanced = True
                    break
                if neighbour in on_stack:
                    lowlink[node] = min(lowlink[node], index_of[neighbour])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])
            if lowlink[node] == index_of[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                sccs.append(component)
    return sccs


def _longest_path(allowed: set[str], adjacency: dict[str, list[str]]) -> int:
    """Longest path in edges over an acyclic subgraph, iterative post-order."""
    depth: dict[str, int] = {}
    for root in sorted(allowed):
        if root in depth:
            continue
        stack: list[tuple[str, bool]] = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if not expanded and node in depth:
                continue
            children = [c for c in adjacency.get(node, ()) if c in allowed]
            if expanded:
                depth[node] = 1 + max((depth[c] for c in children), default=-1)
            else:
                stack.append((node, True))
                stack.extend((c, False) for c in children if c not in depth)
    return max(depth.values(), default=0)


def _is_shortcut(
    source: str, target: str, adjacency: dict[str, list[str]]
) -> bool:
    """True when target is reachable from source without the direct edge —
    i.e. the edge duplicates a >= 2-step path. The direct edge must be
    excluded everywhere in the walk, not only at the first step: a cycle
    can revisit `source` mid-path and would otherwise re-offer the very
    edge under test (a<->b with b->c must NOT make b->c a shortcut)."""
    stack = [n for n in adjacency.get(source, ()) if n != target]
    seen = set(stack)
    while stack:
        node = stack.pop()
        if node == target:
            return True
        for child in adjacency.get(node, ()):
            if node == source and child == target:
                continue
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return False


@register(Metric, "hierarchy-sanity")
class HierarchySanity(Metric):
    """Structural sanity of the induced IS_A hierarchy (M5 spec §4.3), in
    the spirit of TExEval-2's structural analysis: cycles, self-loops,
    depth, transitive shortcuts. No gold needed; all stdlib."""

    def evaluate(
        self, snapshot: GraphSnapshot, ground_truth: dict[str, object]
    ) -> dict[str, float]:
        edges = [
            (r.source_id, r.target_id)
            for r in snapshot.relations
            if r.type == "IS_A"
        ]
        self_loops = sum(1 for a, b in edges if a == b)
        proper = [(a, b) for a, b in edges if a != b]
        nodes = sorted({n for edge in proper for n in edge})
        adjacency: dict[str, list[str]] = {}
        for a, b in proper:
            adjacency.setdefault(a, []).append(b)
        cycle_components = [
            c for c in _tarjan_sccs(nodes, adjacency) if len(c) >= 2
        ]
        cycle_nodes = {n for component in cycle_components for n in component}
        allowed = {n for n in nodes if n not in cycle_nodes}
        acyclic = {
            node: [c for c in children if c in allowed]
            for node, children in adjacency.items()
            if node in allowed
        }
        shortcuts = sum(1 for a, b in proper if _is_shortcut(a, b, adjacency))
        return {
            "cycle-components": float(len(cycle_components)),
            "cycle-nodes": float(len(cycle_nodes)),
            "self-loops": float(self_loops),
            "max-depth": float(_longest_path(allowed, acyclic)),
            "transitive-shortcuts": float(shortcuts),
            "is-a-edges": float(len(edges)),
        }
