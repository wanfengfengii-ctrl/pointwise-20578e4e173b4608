"""Real lineage demonstration: which upstream artifact is truly unavoidable?

Run with::

    python3 demo.py

The graph models a small analytics warehouse:

    source_events -> raw_orders -----> stg_orders ------> int_order_items
                 -> raw_customers --> stg_customers ---> (also feeds
                                                        dim_customers)

`int_order_items` has two upstream branches, and two marts join them.  On top of
that the demo includes a circular dependency (snapshot_orders <-> int_order_items),
a "side door" edge that makes a tempting dominator redundant, an entry back edge
into the source, an unreachable orphan pipeline, and a long linear chain of 40
nodes.  For every artifact the demo prints:

* the immediate dominator found by ``dominance.analyze``;
* the parent that a naive BFS traversal saw first, to show why a traversal
  parent must not be reported as a must-pass dependency;
* an independent audit by vertex deletion, re-checking the definition and the
  "closest" requirement without using any part of the algorithm.
"""

from __future__ import annotations

from collections import deque

from dominance import analyze
from reference_dominance import _build, _reachable, dominates

# ---------------------------------------------------------------------------
# Lineage graph.  Edge (u, v) means data flows u -> v.
# ---------------------------------------------------------------------------
NODES = [
    "source_events",        # 0  root / original source
    "raw_orders",           # 1
    "raw_customers",        # 2
    "stg_orders",           # 3
    "stg_customers",        # 4
    "int_order_items",      # 5  joins both staging branches
    "int_orders_enriched",  # 6
    "dim_customers",        # 7
    "fct_orders",           # 8  joins enriched + dim
    "dash_revenue",         # 9
    "dash_ops",             # 10
    "snapshot_orders",      # 11 cycle with int_order_items
    "adhoc_export",         # 12 reached via two unrelated branches
    "orphan_stg",           # 13 unreachable pipeline
    "orphan_mart",          # 14 unreachable pipeline
    "chain_0",              # 15 .. 15+L-1 long linear chain
]
L = 40  # chain nodes chain_0 .. chain_{L-1}
N = len(NODES) + (L - 1)
names = NODES + [f"chain_{i}" for i in range(1, L)]
assert len(names) == N

EDGES = [
    (0, 1),    # source_events    -> raw_orders
    (0, 2),    # source_events    -> raw_customers
    (1, 3),    # raw_orders       -> stg_orders
    (2, 4),    # raw_customers    -> stg_customers
    (3, 5),    # stg_orders       -> int_order_items
    (4, 5),    # stg_customers    -> int_order_items
    (5, 6),    # int_order_items  -> int_orders_enriched
    (4, 7),    # stg_customers    -> dim_customers
    (6, 8),    # int_orders_enriched -> fct_orders
    (7, 8),    # dim_customers    -> fct_orders
    (8, 9),    # fct_orders       -> dash_revenue
    (6, 10),   # int_orders_enriched -> dash_ops
    (5, 11),   # int_order_items  -> snapshot_orders    (cycle)
    (11, 5),   # snapshot_orders  -> int_order_items    (cycle)
    (3, 12),   # stg_orders       -> adhoc_export      (side door)
    (10, 12),  # dash_ops         -> adhoc_export      (join)
    (13, 14),  # orphan pipeline, unreachable from root
    (9, 0),    # entry back edge: dash_revenue feeds source_events
    (10, 15),  # dash_ops         -> chain_0
] + [(15 + i, 15 + i + 1) for i in range(L - 1)]

ROOT = 0


def naive_bfs_parents(n, root, edges):
    """The parent each node was first discovered through -- NOT idom."""
    succ = _build(n, edges)
    parent = [None] * n
    seen = [False] * n
    seen[root] = True
    q = deque([root])
    while q:
        u = q.popleft()
        for v in succ[u]:
            if not seen[v]:
                seen[v] = True
                parent[v] = u
                q.append(v)
    return parent


def audit(n, root, edges, result):
    """Re-check every answer by deleting the reported vertex, definitions only."""
    succ = _build(n, edges)
    reach = _reachable(succ, root, n)
    failures = []
    for v in range(n):
        d = result[v]
        if v == root:
            continue
        if d is None:
            if reach[v]:
                failures.append(f"{names[v]} reported unreachable but is reachable")
            continue
        if not dominates(n, root, edges, d, v):
            failures.append(f"removing {names[d]} does not cut off {names[v]}")
            continue
        # Each strict dominator of v must also dominate d; otherwise a nearer
        # strict dominator than d would exist.
        strict = [u for u in range(n)
                  if u != v and reach[u]
                  and not _reachable(succ, root, n, u)[v]]
        for u in strict:
            if u != d and not dominates(n, root, edges, u, d):
                failures.append(
                    f"{names[d]} is not the closest dominator of {names[v]} "
                    f"({names[u]} is nearer)")
                break
    return failures


def main() -> int:
    result = analyze(N, ROOT, EDGES)
    parent = naive_bfs_parents(N, ROOT, EDGES)

    print(f"Lineage graph: {N} nodes, {len(EDGES)} edges, "
          f"original source = {names[ROOT]}\n")
    print(f"{'artifact':<20} {'must-pass upstream (idom)':<24} "
          f"{'naive BFS parent':<20}")
    print("-" * 68)
    for v in range(N):
        d = result[v]
        idom_txt = ("(root)" if v == ROOT else
                    "UNREACHABLE" if d is None else names[d])
        par = parent[v]
        par_txt = "(root)" if v == ROOT else "-" if par is None else names[par]
        flag = "" if (v == ROOT or d is None) else (
            "  <-- traversal parent was wrong" if par != d else "")
        print(f"{names[v]:<20} {idom_txt:<24} {par_txt:<20}{flag}")

    print("\nHow to read this:")
    print("  * A naive traversal's parent is a witness of *one* path only.")
    print("    For example fct_orders was first reached through")
    print("    int_orders_enriched, so BFS calls that its parent, yet a path")
    print("    through dim_customers exists, so int_orders_enriched cannot be")
    print("    a must-pass dependency.  Only source_events is on every path.")
    print("  * adhoc_export was first reached through stg_orders, but its")
    print("    true must-pass upstream is source_events: the dash_ops branch")
    print("    avoids stg_orders and the side door avoids dash_ops.")
    print("  * snapshot_orders and int_order_items form a cycle, so neither")
    print("    dominates the other; the cycle is entered at int_order_items.")
    print("  * orphan_stg / orphan_mart are None: no path from source_events.")
    print("  * chain_* is a 40-node linear tail: each node's must-pass")
    print("    upstream is exactly its predecessor (verified below).")
    print("  * dash_revenue -> source_events is an entry back edge; the source")
    print("    still maps to itself by definition.")

    chain_ok = all(result[15 + i] == 15 + i - 1 for i in range(1, L))
    cycle_ok = result[5] == 0 and result[11] == 5
    orphans_ok = result[13] is None and result[14] is None

    failures = audit(N, ROOT, EDGES, result)
    print("\nIndependent audit by vertex deletion:")
    print(f"  reachable non-root nodes audited: "
          f"{sum(1 for x in result if x is not None) - 1}")
    for f in failures:
        print("  FAIL:", f)
    if not failures:
        print("  all reported idoms are indispensable and closest")

    ok = not failures and chain_ok and cycle_ok and orphans_ok
    print(f"\nchain idoms exact: {chain_ok}; cycle entry correct: {cycle_ok}; "
          f"orphans unreachable: {orphans_ok}")
    print("Result:", "DEMO OK" if ok else "DEMO FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
