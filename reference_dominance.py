"""Independent reference implementation used only by tests and demos.

Two references are provided:

``brute_force_idom(n, root, edges)``
    Definitional, *deletion based*, for small graphs.  A node ``u`` strictly
    dominates ``v`` iff ``v`` is reachable from ``root`` and becomes
    unreachable when ``u`` is deleted.  The immediate dominator of ``v`` is the
    strict dominator whose own strict-dominator set has size
    ``|strict_dominators(v)| - 1`` (the strict dominators of a node always form
    a chain, so this picks the closest one).  This is deliberately a different
    algorithm from Lengauer-Tarjan: it deletes one vertex per query and runs a
    fresh reachability search.

``bitset_idom(n, root, edges)``
    Iterative dominator dataflow over Python integer bitsets.  It is only
    practical for moderate ``n`` (say a few thousand) but it processes all
    nodes at once with no DFS numbering, path compression, or semidominators,
    so it is a genuinely independent full-answer check.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence


def _build(n: int, edges: Iterable[Sequence[int]]):
    succ = [[] for _ in range(n)]
    for u, v in edges:
        succ[u].append(v)
    return succ


def _reachable(succ, root: int, n: int, blocked: int = -1) -> list[bool]:
    """BFS/DFS reachability from ``root`` avoiding ``blocked``."""
    seen = [False] * n
    if root == blocked:
        return seen
    seen[root] = True
    stack = [root]
    while stack:
        v = stack.pop()
        for w in succ[v]:
            if w != blocked and not seen[w]:
                seen[w] = True
                stack.append(w)
    return seen


def brute_force_idom(
    n: int, root: int, edges: Iterable[Sequence[int]]
) -> list[Optional[int]]:
    """Definitional immediate dominators by vertex deletion (small graphs)."""
    edges = list(edges)
    succ = _build(n, edges)
    reachable = _reachable(succ, root, n)

    # strict_dom[v] = sorted list of u such that deleting u makes v unreachable
    strict_dom: list[list[int]] = [[] for _ in range(n)]
    for v in range(n):
        if not reachable[v]:
            continue
        strict_dom[v] = [
            u for u in range(n) if u != v and not _reachable(succ, root, n, u)[v]
        ]

    result: list[Optional[int]] = [None] * n
    for v in range(n):
        if not reachable[v]:
            continue
        if v == root:
            result[v] = root
            continue
        candidates = [
            u for u in strict_dom[v] if len(strict_dom[u]) == len(strict_dom[v]) - 1
        ]
        if len(candidates) != 1:
            raise AssertionError(
                f"deletion reference is inconsistent at node {v}: {candidates}"
            )
        result[v] = candidates[0]
    return result


def bitset_idom(
    n: int, root: int, edges: Iterable[Sequence[int]]
) -> list[Optional[int]]:
    """Iterative dominator dataflow with one Python int bitset per node."""
    edges = list(edges)
    pred = [[] for _ in range(n)]
    succ = _build(n, edges)
    for u, v in edges:
        pred[v].append(u)

    reachable = _reachable(succ, root, n)
    reachable_nodes = [v for v in range(n) if reachable[v]]

    # dom[root] = {root}; dom[v] = everything, for v != root, v reachable.
    full = 0
    for v in reachable_nodes:
        full |= 1 << v
    dom = {v: full for v in reachable_nodes}
    dom[root] = 1 << root

    changed = True
    while changed:
        changed = False
        for v in reachable_nodes:
            if v == root:
                continue
            new = full
            any_pred = False
            for u in pred[v]:
                if not reachable[u]:
                    continue
                any_pred = True
                new &= dom[u]
            if any_pred:
                new |= 1 << v
            else:  # only reachable via edges from unreachable nodes: unreachable
                new = 0
            if new != dom[v]:
                dom[v] = new
                changed = True

    result: list[Optional[int]] = [None] * n
    for v in reachable_nodes:
        if v == root:
            result[v] = root
            continue
        s = dom[v] & ~(1 << v)
        # Immediate dominator: the strict dominator strictly dominated by all
        # the others, i.e. its own dominator set is s without itself.
        found = None
        for u in reachable_nodes:
            if not (s >> u) & 1:
                continue
            if (dom[u] & ~(1 << u)) == (s & ~(1 << u)):
                found = u
                break
        if found is None:
            raise AssertionError(f"bitset reference found no idom for {v}")
        result[v] = found
    return result


def dominates(n: int, root: int, edges: Iterable[Sequence[int]], u: int, v: int) -> bool:
    """True iff ``u`` lies on every path from root to v (u != v, v reachable)."""
    if u == v:
        return False
    succ = _build(n, list(edges))
    if u == root:
        return _reachable(succ, root, n)[v]
    if not _reachable(succ, root, n)[v]:
        return False
    return not _reachable(succ, root, n, u)[v]
