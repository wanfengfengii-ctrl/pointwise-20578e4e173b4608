"""Immediate-dominator (direct dominator) analysis.

Public entry point
------------------
``analyze(n, root, edges) -> list``

Nodes are integers ``0 .. n-1``.  Every ``(u, v)`` in ``edges`` means data can
flow from ``u`` to ``v``.  Cycles, self loops and parallel edges are valid.

The result has length ``n``:

* ``result[root] == root``;
* ``result[v] is None`` when ``v`` is unreachable from ``root``;
* otherwise ``result[v]`` is the *immediate dominator* of ``v``: the unique
  node (other than ``v``) which lies on every directed path from ``root`` to
  ``v`` and is closest to ``v``.

Algorithm
---------
Lengauer-Tarjan with path compression (the ``eval``/``link`` formulation).
The DFS used to build the DFS tree is iterative, so arbitrarily long chains do
not depend on Python's recursion limit.

For ``N`` reachable nodes and ``M`` edges the implementation runs in
``O((N + M) log(N + M + 2))`` time (Lengauer-Tarjan is near-linear; this is
also within the simpler ``O((N + M) log N)`` bound) and uses ``O(N + M)``
extra space.  In particular it does not delete vertices one at a time, does not
enumerate paths, and does not store a dominator set per vertex.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence


def analyze(
    n: int,
    root: int,
    edges: Iterable[Sequence[int]],
) -> list[Optional[int]]:
    """Return the immediate dominator of every node, or ``None`` if unreachable.

    Parameters
    ----------
    n:
        Number of nodes, ``1 <= n``.  Nodes are ``0 .. n-1``.
    root:
        A legal node id.
    edges:
        Iterable of ``(u, v)`` pairs, meaning a directed edge ``u -> v``.
        The input is only read, never modified.

    Notes
    -----
    The graph is handled as a directed graph with parallel edges allowed (they
    are harmless because only reachability matters).  Self loops are harmless
    for the same reason.
    """

    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return []
    if not (0 <= root < n):
        raise ValueError("root must be in range(0, n)")

    succ: list[list[int]] = [[] for _ in range(n)]
    pred: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        # Trust the documented input contract; no validation is done here to
        # keep the hot path fast.
        succ[u].append(v)
        pred[v].append(u)

    # ------------------------------------------------------------------
    # 1. Iterative DFS preorder.  Index 1 is the root; vertex[i] is the
    #    node with DFS number i; parent[i] is its DFS-tree parent.
    # ------------------------------------------------------------------
    dfn = [0] * n
    vertex: list[int] = [0]  # vertex[0] is a sentinel
    parent: list[int] = [0]  # parent[0] is a sentinel; parent[1] = 0
    dfn[root] = 1
    vertex.append(root)
    parent.append(0)

    next_edge = [0] * n
    stack = [root]
    while stack:
        v = stack[-1]
        out = succ[v]
        i = next_edge[v]
        degree = len(out)
        while i < degree and dfn[out[i]] != 0:
            i += 1
        if i == degree:
            # All outgoing edges have been examined.
            next_edge[v] = i
            stack.pop()
            continue
        w = out[i]
        next_edge[v] = i + 1
        if dfn[w] == 0:  # undiscovered: make it a DFS-tree child
            dfn[w] = len(vertex)
            vertex.append(w)
            parent.append(dfn[v])
            stack.append(w)

    reachable = len(vertex) - 1
    if reachable == 1:  # Only the root is reachable.
        result: list[Optional[int]] = [None] * n
        result[root] = root
        return result

    # ------------------------------------------------------------------
    # 2. Lengauer-Tarjan.  All arrays below are indexed by DFS number.
    # ------------------------------------------------------------------
    semi = list(range(reachable + 1))
    label = list(range(reachable + 1))
    ancestor = [0] * (reachable + 1)
    idom = [0] * (reachable + 1)
    bucket: list[list[int]] = [[] for _ in range(reachable + 1)]

    def evaluate(v: int) -> int:
        """Lengauer-Tarjan ``eval`` with iterative path compression."""
        if ancestor[v] == 0:
            return v
        # Nodes strictly between v and the top of the current link path.
        path: list[int] = []
        x = v
        while True:
            a = ancestor[x]
            if a == 0 or ancestor[a] == 0:
                break
            path.append(x)
            x = a
        for x in reversed(path):
            a = ancestor[x]
            la = label[a]
            lx = label[x]
            if semi[la] < semi[lx]:
                label[x] = la
            ancestor[x] = ancestor[a]
        return label[v]

    # Process vertices from the deepest DFS number down to 2.  The root
    # (DFS number 1) is the only vertex whose idom is not computed.
    for i in range(reachable, 1, -1):
        w = vertex[i]
        si = semi[i]

        # Semidominator: minimum DFS number reachable by one predecessor
        # followed by a (possibly empty) tree path.
        for u_node in pred[w]:
            j = dfn[u_node]
            if j != 0:
                u = evaluate(j)
                su = semi[u]
                if su < si:
                    si = su
        semi[i] = si

        bucket[si].append(i)
        p = parent[i]
        ancestor[i] = p

        # Resolve vertices whose semidominator has just been passed.
        entries = bucket[p]
        if entries:
            for v in entries:
                u = evaluate(v)
                if semi[u] < semi[v]:
                    idom[v] = u
                else:
                    idom[v] = p
            entries.clear()

    # ------------------------------------------------------------------
    # 3. Turn semidominators into immediate dominators, then map DFS
    #    numbers back to node ids.
    # ------------------------------------------------------------------
    result = [None] * n
    result[root] = root
    for i in range(2, reachable + 1):
        if idom[i] != semi[i]:
            idom[i] = idom[idom[i]]
        result[vertex[i]] = vertex[idom[i]]

    return result
