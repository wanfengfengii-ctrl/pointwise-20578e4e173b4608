"""Verification suite for ``dominance.analyze``.

Run with::

    python3 verify.py

Exit code 0 means every check passed.  The checks cover:

* the stated example;
* hand-written structural cases (cycles, unreachable regions, entry back edges,
  long chains, self loops, parallel edges);
* exhaustive small-graph cross-checks against two independent references
  (vertex-deletion brute force and bitset dataflow);
* random medium-graph agreement with the bitset reference;
* an end-to-end ``n = 200000`` scale run with time and memory reported.
"""

from __future__ import annotations

import itertools
import random
import sys
import time
import tracemalloc

from dominance import analyze
from reference_dominance import bitset_idom, brute_force_idom

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def expect(name: str, got, want) -> None:
    check(name, got == want, f"got {got}, want {want}")


# ---------------------------------------------------------------------------
# 1. Stated example and structural cases
# ---------------------------------------------------------------------------
def structural_cases() -> None:
    print("\n== structural cases ==")
    expect(
        "example: diamond 0->1,0->2,1->3,2->3",
        analyze(5, 0, [(0, 1), (0, 2), (1, 3), (2, 3)]),
        [0, 0, 0, 0, None],
    )
    expect("single node", analyze(1, 0, []), [0])

    # Cycle: 0->1->2->3->1.  The only way into the cycle is through 0, but the
    # cycle has no alternative entry, so each member is still entered by its
    # tree predecessor: idom(1)=0, idom(2)=1, idom(3)=2.  4 hangs off 3.
    expect(
        "cycle 0->1->2->3->1, 3->4",
        analyze(6, 0, [(0, 1), (1, 2), (2, 3), (3, 1), (3, 4)]),
        [0, 0, 1, 2, 3, None],
    )

    # Cycle where the entry node is also reached through the cycle, so 1 no
    # longer dominates 3 (there is a direct edge 0->3).  idom(2)=1 still.
    expect(
        "cycle with back edge to entry",
        analyze(4, 0, [(0, 1), (1, 2), (2, 1), (2, 3), (0, 3)]),
        [0, 0, 1, 0],
    )

    # Unreachable region behind a cycle that is itself unreachable.
    expect(
        "unreachable component",
        analyze(6, 0, [(0, 1), (4, 5), (5, 4), (2, 3)]),
        [0, 0, None, None, None, None],
    )

    # Entry back edge: root receives an edge from a reachable descendant.
    expect(
        "entry back edge 2->0",
        analyze(4, 0, [(0, 1), (1, 2), (2, 0), (2, 3)]),
        [0, 0, 1, 2],
    )

    # A node reachable only through the root and an entry-back-edge cycle.
    # 0->1, 1->2, 2->1, 1->3, 0->3: the direct edge 0->3 means 1 does not
    # dominate 3, so idom(3)=0.  idom(2)=1 because 2 is only entered from 1.
    expect(
        "back edge cycle before leaf",
        analyze(5, 0, [(0, 1), (1, 2), (2, 1), (1, 3), (0, 3)]),
        [0, 0, 1, 0, None],
    )

    # Same graph without the shortcut edge: now 1 dominates 2 and 3.
    expect(
        "back edge cycle, no shortcut",
        analyze(5, 0, [(0, 1), (1, 2), (2, 1), (1, 3)]),
        [0, 0, 1, 1, None],
    )

    # Parallel edges and self loops change nothing.
    expect(
        "self loop and parallel edges",
        analyze(3, 0, [(0, 0), (0, 1), (0, 1), (1, 1), (1, 2), (2, 2)]),
        [0, 0, 1],
    )

    # Long chain, 20000 nodes, deeper than the default recursion limit.
    n = 20000
    chain = [(i, i + 1) for i in range(n - 1)]
    got = analyze(n, 0, chain)
    expect(
        "long chain n=20000",
        got,
        [0] + list(range(0, n - 1)),
    )

    # Maximum-size chain under a deliberately tiny recursion limit.  The
    # implementation never recurses, so lowering the limit must not matter.
    n = 200000
    chain = [(i, i + 1) for i in range(n - 1)]
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(100)
    try:
        got = analyze(n, 0, chain)
    finally:
        sys.setrecursionlimit(old_limit)
    check("max chain n=200000 with recursionlimit=100",
          got == [0] + list(range(0, n - 1)),
          "long-chain failure under low recursion limit")

    # Root in the middle of a chain: the prefix is unreachable, the suffix is
    # a plain chain rooted at the new source.
    mid = 123457
    got = analyze(n, mid, chain)
    ok = (all(x is None for x in got[:mid])
          and got[mid] == mid
          and all(got[i] == i - 1 for i in range(mid + 1, n)))
    check("root mid-chain n=200000", ok, "bad mid-chain result")

    # Root mid-graph, edges pointing both ways, unreachable side.
    expect(
        "root not 0",
        analyze(5, 2, [(2, 3), (3, 4), (0, 1)]),
        [None, None, 2, 2, 3],
    )

    # Join reached through two branches.  DFS happens to make 1 the tree
    # parent of 3, but 0 is on every path to 3, so idom(3)=0: the tree parent
    # is *not* the answer.  idom(4)=3.
    expect(
        "idom is not the DFS tree parent",
        analyze(5, 0, [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]),
        [0, 0, 0, 0, 3],
    )

    # Two disjoint cycles in the reachable part.
    expect(
        "two cycles sharing root",
        analyze(7, 0, [(0, 1), (1, 2), (2, 1), (0, 3), (3, 4), (4, 5), (5, 3), (4, 6)]),
        [0, 0, 1, 0, 3, 4, 4],
    )

    # No edges at all.
    expect("no edges", analyze(4, 0, []), [0, None, None, None])

    # The input list must not be modified.
    src = [(0, 1), (1, 2), (2, 1), (2, 3)]
    before = list(src)
    analyze(4, 0, src)
    check("input edges unmodified", src == before, f"got {src}, want {before}")


# ---------------------------------------------------------------------------
# 2. Exhaustive small-graph cross-check against the deletion reference
# ---------------------------------------------------------------------------
def all_edge_sets(n: int):
    pairs = [(u, v) for u in range(n) for v in range(n)]
    # 0 to 4 edges is enough to exercise cycles, self loops and joins; 5 edges
    # would be 2 million graphs for n=5.
    for k in range(0, 5):
        for combo in itertools.combinations(pairs, k):
            yield list(combo)


def exhaustive_small() -> None:
    print("\n== exhaustive small graphs vs deletion reference ==")
    cases = 0
    for n in range(1, 5):
        for root in range(n):
            for edges in all_edge_sets(n):
                got = analyze(n, root, edges)
                want = brute_force_idom(n, root, edges)
                if got != want:
                    check(
                        f"exhaustive n={n} root={root} edges={edges}", False,
                        f"got {got}, want {want}",
                    )
                    return
                cases += 1
    check(f"exhaustive {cases} (n<=4, every root, <=4 edges)", True)

    # Sample n=5..7 with parallel edges and self loops; the deletion reference
    # is definitional and independent of DFS numbering entirely.
    rng = random.Random(99)
    sampled = 0
    for _ in range(4000):
        n = rng.randint(5, 7)
        m = rng.randint(0, 14)
        edges = [(rng.randrange(n), rng.randrange(n)) for _ in range(m)]
        root = rng.randrange(n)
        got = analyze(n, root, edges)
        want = brute_force_idom(n, root, edges)
        if got != want:
            check(f"sampled n={n} root={root} edges={edges}", False,
                  f"got {got}, want {want}")
            return
        sampled += 1
    check(f"sampled {sampled} graphs n=5..7 vs deletion reference", True)


# ---------------------------------------------------------------------------
# 3. Random medium graphs against the independent bitset dataflow
# ---------------------------------------------------------------------------
def random_medium() -> None:
    print("\n== random medium graphs vs bitset reference ==")
    rng = random.Random(20261114)
    total = 0
    for trial in range(60):
        n = rng.randint(1, 120)
        density = rng.choice([0.01, 0.03, 0.1, 0.3, 0.8])
        edges = []
        for u in range(n):
            for v in range(n):
                if rng.random() < density:
                    edges.append((u, v))
        if trial % 5 == 0 and edges:
            # Duplicate an edge to exercise parallel edges.
            edges.append(edges[0])
        root = rng.randrange(n)
        got = analyze(n, root, edges)
        want = bitset_idom(n, root, edges)
        if got != want:
            check(f"random trial {trial} n={n} root={root}", False,
                  f"got {got}, want {want}")
            return
        total += 1
    check(f"random {total} medium graphs", True)


# ---------------------------------------------------------------------------
# 4. Scale test
# ---------------------------------------------------------------------------
def scale_test() -> None:
    print("\n== scale test n=200000, 500000 edges ==")
    n = 200000
    rng = random.Random(7)
    edges = []
    # Long chain of 50000 nodes first: worst case for deep DFS.
    for i in range(0, 49999):
        edges.append((i, i + 1))
    # Randomised rest: many cycles, back edges and unreachable tail.  Extra
    # edges never target the chain prefix, so the chain idoms must stay exact.
    for _ in range(500000 - 49999):
        u = rng.randrange(0, 150000)
        v = rng.randrange(50000, 199000)
        edges.append((u, v))
    root = 0

    def run():
        return analyze(n, root, edges)

    # Time once without the profiler (tracemalloc dominates the measurement).
    t0 = time.perf_counter()
    got = run()
    elapsed = time.perf_counter() - t0

    tracemalloc.start()
    peak_edges_free = None
    try:
        analyze(n, root, edges)
        peak_edges_free = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    reachable = sum(1 for x in got if x is not None)
    chain_ok = all(got[i] == i - 1 for i in range(1, 50000))
    if got[0] != 0 or not chain_ok or reachable == 0:
        check("scale run correctness", False, f"root={got[0]} chain_ok={chain_ok}")
        return
    print(f"  reachable: {reachable}/{n}")
    print(f"  analyze time (no profiler): {elapsed:.2f}s")
    print(f"  analyze peak traced memory: {peak_edges_free / 1e6:.1f} MB")
    check("scale run", elapsed < 30.0, f"took {elapsed:.2f}s")
    check("scale run chain idoms", chain_ok, "chain 0..49999 not a pure chain")
    check("scale run result shape",
          len(got) == n and got[0] == root and got[0] == 0,
          "bad result shape")
    check("scale run memory", peak_edges_free < 400e6,
          f"peak {peak_edges_free / 1e6:.1f} MB exceeds 400 MB")

    # Adversarial deep-path case: a full chain plus an edge from the last node
    # to *every* rootward node.  This maximises the length of link paths that
    # path compression must walk.  Every chain node is still unavoidable, so
    # the expected answer is known structurally.
    print("\n== adversarial deep back edges: n=100000, m=199998 ==")
    n2 = 100000
    chain = [(i, i + 1) for i in range(n2 - 1)]
    last = (n2 - 1, 0)  # entry back edge into the root
    deep = [(n2 - 1, i) for i in range(1, n2 - 1)]
    edges2 = chain + [last] + deep
    t0 = time.perf_counter()
    got2 = analyze(n2, 0, edges2)
    elapsed2 = time.perf_counter() - t0
    want2 = [0] + list(range(0, n2 - 1))
    print(f"  analyze time (no profiler): {elapsed2:.2f}s")
    check("deep back edges correctness", got2 == want2,
          "chain idoms broken by rootward back edges")
    check("deep back edges time", elapsed2 < 30.0, f"took {elapsed2:.2f}s")


# ---------------------------------------------------------------------------
# 5. Corner cases for the API contract
# ---------------------------------------------------------------------------
def api_contract() -> None:
    print("\n== api contract ==")
    check("n=1, root=0, self loop only", analyze(1, 0, [(0, 0)]) == [0])
    empty_like = analyze(5, 3, [(0, 1)])
    expect("root isolated -> only root set", empty_like, [None, None, None, 3, None])
    # edges is a generator: input can be any iterable.
    expect("generator input", analyze(3, 0, ((u, v) for u, v in [(0, 1), (1, 2), (2, 1)]))
           , [0, 0, 1])


def main() -> int:
    structural_cases()
    exhaustive_small()
    random_medium()
    api_contract()
    scale_test()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
