# Exact optimization for the ten-hour campaign — 2026-09-22

Three changes to the simulator, each verified **bitwise identical** to the
unpatched tree. No equation, timestep, connectivity,
plasticity semantic, RNG stream or draw order was altered; no parameter was
retuned. `weight_dependence` remains 0.0 by default.

## What changed and why

| # | change | file | what it removes per timestep |
|---|---|---|---|
| 1 | Clip only the synapses just modified, not all 160,000 | `sim/plasticity.py` | 1.22 MB read + 1.22 MB written |
| 2 | Evaluate the weight-dependent step at the touched indices only | `sim/plasticity.py` | two full-matrix temporaries (`weight_dependence > 0` only) |
| 3 | Preallocate the PF→PKJ drive product instead of allocating it fresh | `sim/simulate.py` | one 1.22 MB allocation |

### Rejected: sparse spike ring

A fourth change stored the plasticity spike ring as fired indices rather than a
dense boolean array, shrinking it from 15.4 MB to 2.6 MB so it would fit in the
8 MB per-CCX L3. It was **verified bitwise identical across all 17 cases** and
raised peak aggregate throughput to 11.34, but it is not shipped.

It broke `test_exact_pickle_with_pending_eligibility_and_hidden_state`, which
asserts that a checkpointed rule and its clone pickle to identical bytes. The
simulated state stayed identical (weights matched exactly after 43 further steps);
what changed was pickle *memoization* — with a list of arrays the shared `int64`
dtype object is re-created across a round-trip instead of being a back-reference,
so the byte streams diverge. That test guards checkpoint/resume fidelity, which is
the highest-risk part of an eight-run, ten-hour campaign taking sixty checkpoints
per run. The change was reverted rather than the test weakened: it altered the
persisted representation for a 4-8% gain, which is not a trade worth making.
The other three changes leave every persisted structure unchanged in type and shape.

### Why each is exactly equivalent

1. **Sparse clip.** Only `ltd_idx`/`ltp_idx` are written on a step. Every other
   weight was already inside `[w_min, w_max]` when it was last modified, and
   `np.clip` is the identity on an in-range float, so clipping them again is a
   no-op. Note `weights[idx]` is a fancy-index *copy*: the clipped values must be
   assigned back, not passed as `out=`.
2. **Sparse weight dependence.** `(w - w_min) / span` is elementwise, so gathering
   the touched weights first and dividing after yields the same IEEE result as
   dividing the whole matrix and then gathering.
3. **Preallocated buffer.** `np.multiply(..., out=buf)` performs the same products
   in the same order; only the destination is reused. The subsequent
   `.sum(axis=1)` still reduces over all `n_pf` columns, so NumPy's pairwise
   summation tree — and therefore the rounding — is unchanged. (This is why the
   PF drive is *not* computed sparsely: dropping the zero terms would regroup the
   pairwise tree and change the last bits.)
4. **Sparse ring.** The buffer is consumed only through `np.flatnonzero`, so
   storing that result directly holds identical information. It is kept at native
   index width (`intp`): an int32 ring is rejected by the native cascade kernels,
   which require int64 indices.

## Bitwise equivalence

Harness: `evidence/tenhour_20260922/equiv.py`. Compares SHA-256 over final weights,
IO membrane voltage and calcium, PKJ/DCN voltages, the DCN→IO conductance, the
generator state, LTD/LTP counters, cascade states and counters, and homeostatic
state, after 2,000 burn-in plus 25,000 recorded steps at a fixed seed.

```
case                                match   orig x    opt x  speedup
additive (shipped default)          EXACT    1.145    1.174    1.03x
weight_dependence=1.0               EXACT    0.841    1.145    1.36x
weight_dependence=0.5               EXACT    0.838    1.153    1.38x
null_window 200ms (H3)              EXACT    1.161    1.203    1.04x
null_window 50ms (H3)               EXACT    1.148    1.177    1.03x
binary cascade                      EXACT    1.146    1.156    1.01x
abbott_cascade                      EXACT    1.140    1.171    1.03x
mauk_cascade                        EXACT    1.160    1.172    1.01x
homeostatic scaling                 EXACT    1.145    1.164    1.02x
ablate_dcn_io                       EXACT    1.147    1.180    1.03x
open loop misrouted                 EXACT    1.140    1.175    1.03x
gap off                             EXACT    1.144    1.168    1.02x
gap strong + heterogeneity 0.30     EXACT    1.150    1.190    1.03x
pf pool 125 (ceiling)               EXACT    2.258    2.271    1.01x
pf pool 2000 (floor)                EXACT    0.275    0.303    1.10x
h2 window 200ms                     EXACT    1.149    1.187    1.03x
h2 ratio 19                         EXACT    0.999    1.181    1.18x
ALL BITWISE IDENTICAL
```

## Throughput versus concurrency

Measured with production recording (`evidence/tenhour_20260922/{scale,worker}.py`),
20 biological seconds per process, one exclusive CPU per process, round-robin
across the eight L3 domains. `per-proc` is the median; `slowest` is what actually
sets a batch's completion time.

| procs | baseline total | **shipped** | +sparse ring (rejected) | shipped slowest proc | a 10 h run takes |
|---|---|---|---|---|---|
| 1 | 1.02 | — | — | 1.024× | 9.8 h |
| 4 | 4.16 | — | — | 1.012× | 9.9 h |
| 8 | 7.77 | 8.12 | 8.29 | 0.952× | 10.5 h |
| 12 | — | 10.05 | 10.24 | 0.765× | 13.1 h |
| 16 | 9.04 | 10.65 | 11.34 | 0.616× | 16.2 h |
| 20 | — | 9.81 | 10.39 | 0.460× | 21.7 h |
| 24 | 7.47 | 9.76 | 9.54 | 0.358× | 27.9 h |
| 32 | 6.61 | — | — | 0.180× | 55.5 h |
| 48 | 6.70 | — | — | 0.119× | 84.4 h |

**The machine is memory-bandwidth-bound, not core-bound.** Total throughput peaks
near 10 biological-seconds per wall-second and then *falls*: this is a Threadripper
2990WX whose NUMA nodes 1 and 3 report 0 kB of local memory, so half the cores reach
DRAM only over the fabric. Adding processes past ~12 slows every run without adding
total work.

The optimizations help most exactly where the box is most contended
(7.47 → 9.54 at 24 processes), but they do
not move the point at which a single run drops below real time. Eight concurrent
runs is the largest batch whose slowest member still finishes ten simulated hours
inside ten wall hours.

## Scope

Backups of the four touched files are in
`evidence/tenhour_20260922/pre_optimization_backup/`. Pre-existing uncommitted work
in the tree is untouched and is not attributed to this change. No experimental data
was deleted.

