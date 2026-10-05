# io_equilibrium_sim

A small, CPU-only Python simulation of a simplified olivo-cerebellar microzone,
built to test three hypotheses from an internal Mauk Lab white paper about what
the Inferior Olive (IO) is for. Architecturally inspired by
[CbmSim](https://github.com/mmauk/CbmSim) (Mauk Lab, UT Austin) but **not** a
port of it — no CUDA, no C++, and no granule-cell/mossy-fiber front end.

Every neuron spikes. The olive is a conductance-based cell with real ionic
currents; the Purkinje and nuclear cells are conductance-based
integrate-and-fire. The PF→PKJ synapses are plastic. Nothing in the loop is a
rate approximation.

This README describes what was actually built, what it found, and every knob you
can turn. The original spec (`io_equilibrium_sim_MVP_spec.md`) is an internal
document and is **not** part of this repository; where the spec left an
implementation detail open, the choice made here is documented at the point it
is made.

---

## Quick start

```bash
pip install -r requirements.txt   # numpy, matplotlib

python3 run.py                    # run the model — see the loop equilibrate
python3 run.py DURATION_S=600     # ...or override any setting on the command line
python3 run.py CUT_DCN_TO_IO=1    # ...or cut the feedback and watch it collapse
python -m unittest discover -s tests
```

**`run.py` is the only file you edit.** Every knob the model has — coupling on or
off, the population and convergence counts, the parallel fibres per Purkinje
cell, the plasticity window and step sizes, the lesions — is a plain variable in
a settings block at the top of it. Edit, save, run. Any of them can also be
overridden as `NAME=VALUE` on the command line without touching the file.

If the numbers you choose cannot build a network (the PKJ→DCN counts have to
close), `run.py` says which ones disagree and what to change, before it starts.

Plots land in `results/`, which is **gitignored** — nothing under it ships with
the repo.

> The `experiments/` scripts this README refers to below — the sweeps, the
> calibration, the characterization runs — are not on this branch. They live on
> the **`bloated`** branch, along with the campaign pipeline, the cascade
> plasticity rules and the native backend.

---

## What's in here

| script | what it answers | runs a simulation? | cost |
|---|---|---|---|
| `experiments/run_network_snapshot.py` | What does the wiring look like? | no | instant |
| `experiments/run_baseline.py` | Does the loop settle? (H1, one seed) | yes | ~3 min |
| `experiments/run_network.py` | Same, plus coupling analysis and wiring figures | yes | ~3 min |
| `experiments/run_io_characterization.py` | Does the olivary cell behave like one? | yes, single-cell | ~10 min |
| `experiments/run_calibration.py` | Where do the three synaptic gains come from? | yes, stage by stage | ~10 min |
| `experiments/run_gap_sweep.py` | What do the gap junctions do to the loop? | yes, one per point | ~10 min |
| `experiments/run_window_sweep.py` | Which δ−/δ+ ratio zeroes the drift? (H2, direct) | yes, one per candidate | ~30 min |
| `experiments/run_three_window.py` | Does a null window slow the random walk? (H3) | yes, one per candidate | ~30 min |
| `experiments/sweep.py` | **All of the above, replicated across seeds** | yes, 400 of them | ~4 h on 60 cores |
| `experiments/analyze_sweeps.py` | Turn the sweep into tables | no | instant |
| `experiments/figures.py` | Turn the sweep into the four figures | no | seconds |

Everything in the **Results** section below comes from `sweep.py`. The
single-run scripts are for looking at one condition in detail; they are `n=1`
and they say so. Every script takes `--help`, and every flag is documented in
[Command reference](#command-reference).

---

## The network

CbmSim's microzone, with its connection numbers taken directly from
`src/cbm_state/connectivityparams.cpp` and its populations scaled up 10×:

```
        40 IO,  80 DCN,  320 PKJ,  160,000 PF units

   PF ---(+, plastic)--> PKJ ---(-)--> DCN ---(-)--> IO ---(CF)--> PKJ
                                                      ^                |
                                                      |                |
                                           IO <--> IO gap junctions    |
                                                                       |
                               (CF also gates LTD at the PF synapses) -+
```

**Populations are scaled; per-cell convergence is not.** Each Purkinje cell
still receives one climbing fiber and contacts 3 nuclear cells; each nuclear
cell still integrates 12 Purkinje cells; each olivary cell still integrates 8
nuclear cells. Convergence is what sets a cell's synaptic load, so holding it
fixed means the operating point carries over and **no synaptic gain needed
refitting** for the scale-up.

| quantity | CbmSim | here |
|---|---|---|
| olivary cells | `num_io` = 4 | **40** (×10) |
| nuclear cells | `num_nc` = 8 | **80** (×10) |
| Purkinje cells | `num_pc` = 32 | **320** (×10) |
| PKJ per climbing fiber | `num_p_io_from_io_to_pc` = 8 | **8** |
| PKJ converging on one DCN | `num_p_nc_from_pc_to_nc` = 12 | **12** |
| DCN reached by one PKJ | `num_p_pc_from_pc_to_nc` = 3 | **3** |
| DCN converging on one IO | `num_p_io_from_nc_to_io` = 8 | **8** of 80 — see below |
| IO reached by one DCN | `num_p_nc_from_nc_to_io` = 4 | **4** |
| IO–IO coupling partners | `num_p_io_in_io_to_io` = 3 | **4** (local, not all-to-all) |
| PF onto one PKJ | `num_p_pc_from_gr_to_pc` = 32,768 | **500** — scaled down |

Two entries change meaning at this scale, and both changes are forced.

**DCN→IO stops being complete.** At CbmSim's size, 8 nuclear cells converging on
each olivary cell *is* the whole nucleus — `connectNCtoIO` is complete
bipartite. Holding that convergence at 8 while the nucleus grows to 80 makes the
projection topographic: olivary cell `i` reads a contiguous block of the nucleus
whose start advances with `i`, so neighbours share 6 of their 8 inputs and
distant cells share none. Every nuclear cell still projects to exactly 4 olivary
cells. This is not cosmetic — see [Why the olive was perfectly
synchronized](#why-the-olive-was-perfectly-synchronized).

**IO–IO coupling stops being all-to-all.** What matters physiologically is a
cell's *total* coupling conductance, not the per-junction value. All-to-all on 40
cells means 39 partners and ~12× the leak conductance, which would clamp the
entire olive to a single potential. The topology is local instead (`nearest_k`,
4 partners) with `gap_g` set so the total per cell is 0.0572 mS/cm² — the same
0.95× leak the 4-cell build had.

The wiring **algorithms** are ported too, not just the counts: `connectPCtoNC`'s
topographic block plus randomized overlap, `connectNCtoIO`, `connectIOtoPC`'s
contiguous climbing-fiber blocks, and `connectIOtoIO`. Two are used in a
generalized form because the scale-up takes them outside the regime CbmSim's
version is defined on (both explained above). `sim/connectivity.py` documents the
two places CbmSim's own version does not close cleanly — a retry loop that can
write a duplicate synapse, and a mop-up that dumps leftovers onto the last
nuclear cell — and what is done instead so the realized counts match the
configured ones exactly.

**The one scaled-down number** is the granule input. CbmSim gives each Purkinje
cell 32,768 of a million shared granule cells; this MVP gives it a private pool
of 500 Poisson fibers. That is the spec's own simplification — private pools keep
each Purkinje cell's coincidence detection independent, which is what H2
measures — and it is the only place the connectivity departs from the reference.
It was swept rather than assumed; see [The PF pool
simplification](#the-pf-pool-simplification-measured).

**These numbers satisfy the white paper's ordering for free.** 4 < 8 < 32 gives
`N_IO ≪ N_DCN ≪ N_PKJ` (40 < 80 < 320). The counts are mutually consistent
(320 PKJ × 3 targets = 960 = 80 DCN × 12 inputs), so nothing has to be assumed
that the paper does not state.

### What this costs: the closed-loop constraint

Spec §2.2(2) — every PKJ a climbing fiber contacts helps regulate that same IO —
**holds for all 320 Purkinje cells**, and holds structurally rather than
trivially: the two topographic maps line up, so a Purkinje cell's own nuclear
target always falls inside the block of the nucleus that its climbing fiber's
olivary cell reads back from. (On the unscaled build this was true only because
DCN→IO was complete, i.e. every PKJ influenced every IO.)

§2.2(1) — "an IO may only be modulated by DCN modulated by exactly the PKJ its
own CF contacts" — still does not hold strictly: each olivary cell is influenced
by ~88 Purkinje cells, of which only its own 8 CF targets are sanctioned. That
is **27% of the microzone rather than 100%**, so the violation is one of degree
rather than of kind, and it shrinks further as the nucleus grows relative to
`n_dcn_per_io`. The loop is also no longer un-openable: because
`n_dcn_per_io < n_dcn`, `enforce_closed_loop=False` genuinely rotates which block
of the nucleus each olivary cell reads, misrouting the feedback without cutting
DCN→IO outright. At CbmSim's own ratios there was nothing to rotate.

---

## Hypotheses under test

**H1 — triple-negative equilibrium feedback.** The closed loop PKJ ⊣ DCN ⊣ IO,
with the climbing fiber driving LTD at active PF→PKJ synapses, should hold IO
firing and PF→PKJ weights at a stable equilibrium.
→ **Holds**, and survives the ablation that falsifies it.

**H2 — LTD/LTP window asymmetry must balance at equilibrium.** Because the LTD
window (~100 ms) is much shorter than the average inter-CF interval at
equilibrium (~1000 ms at 1 Hz), LTD magnitude must be ~9× LTP magnitude for
weights to balance on average, shifting as `ratio ≈ (1000 − window) / window`.
→ **Validated**: 8 settings × 10 seeds, every one within 0.02 Hz of prediction.

**H3 (stretch) — three-window plasticity.** An LTD → null → LTP variant should
let δ+/δ− sit closer together and slow the weights' random-walk drift, at the
cost of occasionally blocking a needed LTP event.
→ **Validated as a trade**: the null window does slow the walk, monotonically
across 5 levels × 10 seeds, at a measurable cost in rate control.

---

## Results

![H1](results/fig1_h1_equilibrium.png)
![H2 and H3](results/fig2_plasticity.png)
![coupling](results/fig3_coupling.png)
![robustness](results/fig4_robustness.png)

> These four figures are generated by `experiments/figures.py` from the sweep
> sidecars and saved raw trajectories — re-run it any time without re-simulating.
> `results/` is gitignored, so run `experiments/sweep.py` then `figures.py` to
> produce them.

All numbers below are **mean ± SD over independent seeds**, from a 400-run sweep
(`experiments/sweep.py`, 249 min on 60 cores, zero failures). Every run wrote a
JSON sidecar carrying its full resolved config, a hash of the source that
produced it, and every scalar it is judged on; `results/SWEEP_RESULTS.md` is
regenerated from those by `experiments/analyze_sweeps.py`. Nothing here is n=1.

### H1: the loop settles at the plasticity's balance point

600 s runs, drift measured over the second half:

| heterogeneity | IO / CF rate | mean weight | weight drift | n |
|---|---|---|---|---|
| none (cv 0) | **1.005 ± 0.003 Hz** | 0.369 ± 0.017 | −0.00000 ± 0.00000 | 4 |
| cv 0.10 | **1.008 ± 0.003 Hz** | 0.370 ± 0.020 | −0.00000 ± 0.00000 | 4 |

Drift is zero to five decimals: a settled equilibrium, not a slow trend.
Settling takes a few hundred seconds, so **report equilibria from runs of 300 s
or more** — a 90 s run is still measuring the transient and reports a drift of
about −0.00085 units/s that is not real.

A single 300 s `run_network.py` on the shipped defaults reproduces this
independently of the sweep: **IO 1.00 Hz** (per-cell 0.97–1.03), PKJ 56.12 Hz,
DCN 17.23 Hz, mean weight 0.382, drift −0.00007 units/s, not saturated,
CF synchrony 0.074. The PKJ, DCN and synchrony figures match the sweep's to the
digits reported.

### H1, the falsification: cutting DCN→IO

The ablation the equilibrium claim actually rests on. Only the nucleo-olivary
projection is removed — wiring, plasticity, climbing-fiber teaching and gap
junctions are all untouched (`ablate_dcn_io`, 180 s, 12 seeds each):

| loop | IO / CF rate | mean weight | PKJ | DCN |
|---|---|---|---|---|
| closed | **1.006 ± 0.009 Hz** | 0.364 ± 0.012 | 56.2 | 17.2 |
| **cut** | **2.97 ± 0.00 Hz** | **0.0013** (floor) | 46.3 | 25.6 |

Removing only the feedback limb abolishes the equilibrium completely. CF rate
runs up to 2.97 Hz — the isolated cell's intrinsic rate is 3.00 Hz, so the olive
goes to free-running — and the weights, no longer balanced against anything, are
driven onto the floor. Identical result with and without heterogeneity. This is
the control that distinguishes a real feedback equilibrium from a coincidental
operating point, and the loop passes it.

### H1, the stronger form: the equilibrium is an attractor

An operating point is not an equilibrium unless the system *returns* to it. Runs
started from five different initial weights (180 s, 5 seeds each):

| `w_init` | IO / CF rate | final weight | drift |
|---|---|---|---|
| 0.1 | 0.999 ± 0.008 | 0.362 ± 0.019 | **+0.0004** |
| 0.3 | 1.006 ± 0.019 | 0.365 ± 0.023 | **+0.0001** |
| 0.5 | 1.005 ± 0.012 | 0.364 ± 0.017 | −0.0002 |
| 0.7 | 1.000 ± 0.004 | 0.365 ± 0.013 | −0.0006 |
| 0.9 | 1.005 ± 0.006 | 0.362 ± 0.016 | **−0.0011** |

Every starting point converges on the same weight (0.362–0.365) and the same
rate (1.00 Hz), and **the drift sign flips with which side it approaches from** —
positive from below, negative from above, steepest from furthest away. That is
the signature of a stable fixed point, not a coincidence of initial conditions.

### H2: the plasticity window sets the equilibrium rate

H1 and H2 are one claim: the loop settles where the plasticity balances, at an
inter-CF interval of `window × (1 + δ−/δ+)`. Eight settings, 120 s, **10 seeds
each**, predicted against measured:

| LTD window | δ−/δ+ | predicted | measured | error |
|---|---|---|---|---|
| 250 ms | 9 | 0.400 Hz | **0.400 ± 0.017** | +0.000 |
| 200 ms | 9 | 0.500 Hz | **0.508 ± 0.022** | +0.008 |
| 100 ms | 19 | 0.500 Hz | **0.504 ± 0.011** | +0.004 |
| 150 ms | 9 | 0.667 Hz | **0.673 ± 0.017** | +0.006 |
| 100 ms | 14 | 0.667 Hz | **0.671 ± 0.017** | +0.005 |
| 100 ms | 9 | 1.000 Hz | **1.008 ± 0.021** | +0.008 |
| 100 ms | 4 | 2.000 Hz | **1.982 ± 0.016** | −0.018 |
| 50 ms | 9 | 2.000 Hz | **1.980 ± 0.018** | −0.020 |

Every condition lands within 0.02 Hz of prediction, and all three *pairs* that
predict the same rate from different knobs agree with each other: 100/19 with
200/9 at 0.5 Hz, 100/14 with 150/9 at 0.667 Hz, 100/4 with 50/9 at 2 Hz. The
loop tracks the product `window × (1 + ratio)` and is indifferent to how that
product is reached, which is what H1 and H2 together predict and what a
coincidence would not produce.

The usable range extends to at least 2 Hz: both 2 Hz conditions settle at
w ≈ 0.57 with `weight_saturated` false in all 20 runs. That is a property of
this scale — the equilibrium sits at w ≈ 0.37, leaving headroom above it.

### H3: three-window plasticity

An LTD → null → LTP variant, tested on a spiking network for the first time
(120 s, 10 seeds each):

| null window | IO / CF rate | mean weight | cross-synapse SD |
|---|---|---|---|
| 0 ms (two-window) | 1.008 ± 0.021 | 0.358 ± 0.016 | **0.1514 ± 0.0016** |
| 25 ms | 0.978 ± 0.022 | 0.365 ± 0.016 | 0.1493 ± 0.0026 |
| 50 ms | 0.950 ± 0.022 | 0.363 ± 0.013 | 0.1473 ± 0.0016 |
| 100 ms | 0.908 ± 0.022 | 0.353 ± 0.019 | 0.1464 ± 0.0028 |
| 200 ms | 0.838 ± 0.013 | 0.348 ± 0.011 | **0.1415 ± 0.0028** |

H3's claim is that a null window should slow the weights' random walk. It does:
cross-synapse spread falls monotonically across all five levels, by ~7% from
0 ms to 200 ms. The effect is real but modest, and it is not free — the null
window blocks LTP events the loop needs, so the equilibrium rate drops steadily
(1.008 → 0.838 Hz) as the window widens. **A null window is a trade, not an
improvement**: it buys weight stability and pays in rate control.

### The PF pool simplification, measured

Each Purkinje cell gets a private pool of Poisson fibers instead of CbmSim's
32,768 drawn from a shared granule population. That is the model's biggest
simplification and it is load-bearing for H2, so it was swept (90 s, 5 seeds):

| fibers per PKJ | IO / CF rate | mean weight | saturated? |
|---|---|---|---|
| 125 | **0.254 ± 0.022 Hz** | **0.995** | **yes — at ceiling** |
| 250 | 0.979 ± 0.035 | 0.735 ± 0.012 | no |
| **500** (shipped) | 0.984 ± 0.035 | 0.368 ± 0.026 | no |
| 1000 | 1.000 ± 0.012 | 0.185 ± 0.009 | no |
| 2000 | 1.050 ± 0.022 | 0.088 ± 0.012 | no |

**The equilibrium rate is invariant to pool size over a 4× range** (250–1000
fibers all give ~1 Hz) even though the weight it is reached at varies 4×, exactly
as it must: total excitatory drive is `n_pf × w × gain`, so `w` compensates for
`n_pf` and the loop still finds its rate.

The model does break outside that range. At 125 fibers the loop **cannot** reach
1 Hz: even with every synapse pinned at `w_max` there is not enough excitatory
drive, so the weights saturate and CF rate collapses to 0.25 Hz. At 2000 the
weight (0.088) approaches the floor and the rate begins to creep up. So 500 is a
defensible choice rather than an arbitrary one — but **the result to quote is the
rate, which is robust, not the weight, which is a function of pool size.**

### Numerical validity: the 1 ms outer loop is fine

The IO's 0.1 ms substep is pinned against a 4× finer one by
`tests/test_io_channels.py`. The outer loop — which the plasticity windows,
synaptic decay and the whole feedback limb run on — is checked here:

| `dt_ms` | IO / CF rate | mean weight | PKJ |
|---|---|---|---|
| 1.0 | 0.981 ± 0.040 | 0.366 ± 0.030 | 56.1 |
| 0.5 | 1.003 ± 0.022 | 0.391 ± 0.011 | 56.9 |
| 0.25 | 0.996 ± 0.009 | 0.421 ± 0.011 | 57.6 |

**CF rate is dt-invariant** across a 4× refinement — the quantity H1 and H2 are
about does not move. The weight it settles at drifts mildly (0.366 → 0.421, about
13%), tracking a 2.7% rise in PKJ rate, so **absolute weights should be read as
dt-dependent to ~15%, while rates should not.**

### Why the olive was perfectly synchronized

On the unscaled 4-cell build, CF synchrony read **1.000 at every coupling
strength, including `gap_g = 0`**. That was not the gap junctions working well.
The four cells were producing *bit-identical* voltage traces:

```
max |V[1] - V[0]| over the whole run = 0.000e+00 mV
max |V[2] - V[0]| over the whole run = 0.000e+00 mV
max |V[3] - V[0]| over the whole run = 0.000e+00 mV
```

Zero to the last bit, not "very close". Three things had to be true at once, and
each arrived from a separate, individually reasonable decision:

1. **Identical input.** CbmSim's `connectNCtoIO` is complete bipartite, so all
   four cells received inhibition from the same eight nuclear cells.
2. **No noise.** `noise_sigma = 0`, so nothing broke the tie stochastically.
3. **Identical parameters.** Every olivary cell shared one `IOChannelParams`.

A deterministic system with identical parameters and identical input has exactly
one trajectory. The four "cells" were one cell integrated four times. Synchrony
read 1.000 because the correlation of a signal with itself is 1.000, and the gap
junctions had nothing to act on — current flows between coupled cells only when
their potentials *differ*, and here the difference was identically zero, so
`I_gap` was zero at every timestep regardless of `gap_g`.

**The 10× scale-up fixes this, and not by accident.** Condition (1) is the one
that breaks: holding `n_dcn_per_io` at 8 while the nucleus grows to 80 makes the
nucleo-olivary projection topographic instead of complete, so no two olivary
cells share an inhibition pattern (neighbours share 6 of 8, distant cells share
0). The cells diverge on their own, and the gap junctions have real differences
to work against. `tests/test_network_graph.py` locks this in: it asserts all
`n_io` inhibition patterns are distinct, so the degeneracy cannot come back
unnoticed.

Conditions (2) and (3) are untouched — the olive is still deterministic, still
uniformly parameterised, and still has no applied current, bias, or jitter.
`io_heterogeneity_cv` exists to relax (3) deliberately, and it earns its place
for a different reason (below).

### Electrical coupling: what it does, and what sets it

`sim/io_coupling.py`. A gap junction is an ohmic resistor between two somata, so
the current into cell *i* is

```
I_gap(i) = sum_j  g_gap[i,j] * (V_j - V_i)
```

with `g_gap` symmetric and zero on the diagonal. This is the rule CbmSim applies
to its own IO array — `vCoupleIO[i] += coupleRiRjRatioIO * (vIO[j] - vIO[i])`
summed over each cell's partners in `MZone::updateIOOut` — so `all_to_all` is the
direct analogue of CbmSim's default and `gap_g` plays the role of
`coupleRiRjRatioIO`. Two deliberate differences:

- CbmSim adds the coupling as a voltage increment in arbitrary units on top of a
  leaky-integrate-and-fire IO. Here it is a real conductance in mS/cm² entering
  the conductance-based cell's current balance, so it **shunts as well as pulls**
  — which is what a gap junction physically does.
- CbmSim computes the term from the *previous* timestep's voltages. Here it is
  evaluated inside the 0.1 ms sub-timestep, at the same instant as every other
  current. The Ca²⁺ spike upstroke moves V by tens of mV in a millisecond, so a
  1 ms lag would materially mis-state the current flowing during exactly the
  event the coupling is supposed to synchronize.

The conductance splits exactly into `(g_gap @ V) − (row sums) * V`, so it folds
into the same exponential-Euler update as every other conductance — no separate
explicit term, no stability penalty. That is *why* the cells are integrated as a
population: cell *i*'s update needs every other cell's V at the same instant.

Four topologies are built (`gap_topology`): `all_to_all` (CbmSim's default),
`ring` (2 partners), `nearest_k` (2k partners), and `small_world` (nearest_k with
a fraction of junctions rewired to random partners). **The shipped topology is
`nearest_k` with k=2**, because all-to-all does not survive the scale-up: on 40
cells it means 39 partners and ~12× leak, which would clamp the whole olive to
one potential. Gap junctions in the olive are local contacts within a glomerulus,
so a local topology is also the more faithful choice at this size.

`experiments/run_gap_sweep.py` measures both what coupling buys and what it
costs, in the assembled closed loop. "Vm r" is the mean pairwise correlation of
subthreshold membrane potential with the Ca²⁺ spikes masked out — the direct
read-out of electrical coupling, and far more sensitive than a spike measure at
~1 Hz firing.

**Coupling strength** (90 s, 6 seeds, `nearest_k`, 4 partners):

| `gap_g` | × leak | Vm r | CF sync | IO Hz | rate spread |
|---|---|---|---|---|---|
| 0.000 | 0.0 | +0.023 ± 0.004 | +0.014 | 0.998 ± 0.005 | 0.080 Hz |
| 0.00715 | 0.5 | +0.129 ± 0.013 | +0.053 | 0.996 ± 0.015 | 0.065 Hz |
| **0.0143** | **1.0** | **+0.540 ± 0.035** | **+0.074** | 0.988 ± 0.034 | 0.033 Hz |
| 0.0286 | 1.9 | +0.769 ± 0.010 | +0.170 | 0.989 ± 0.019 | 0.001 Hz |
| 0.0572 | 3.8 | +0.836 ± 0.004 | +0.269 | 0.999 ± 0.019 | 0.001 Hz |

CF rate is flat at ~1 Hz down the whole column while Vm correlation moves 36-fold:
**coupling changes when the olive fires together, not how often.** The rate is set
by the plasticity balance point, independently of synchrony.

**Topology beats strength, at identical cost.** Every row below has the *same*
total coupling conductance per cell (0.0572 mS/cm², 0.95× leak); only the wiring
differs (90 s, 4 seeds):

| topology | partners | graph diameter | Vm r | **CF sync** | d=1 | d=5 | d=20 |
|---|---|---|---|---|---|---|---|
| ring | 2 | 20 | +0.550 | **+0.049** | +0.898 | +0.636 | +0.408 |
| nearest_k | 4 | 10 | +0.532 | **+0.072** | +0.902 | +0.615 | +0.389 |
| small_world (15% rewired) | 4 | 7 | +0.541 | **+0.140** | +0.863 | +0.609 | +0.423 |
| all_to_all | 39 | 1 | +0.761 | **+0.448** | +0.866 | +0.746 | **+0.756** |

Spike synchrony varies **9-fold across topologies that cost exactly the same
conductance.** The last three columns say why: under every local topology the
correlation decays steeply with distance around the olive (0.90 at neighbours,
0.39 at the far side), while under all-to-all it is **flat** (0.87 → 0.76). Global
synchrony is limited by *path length*, not by conductance — the local junctions
are already near saturation at d=1, so raising `gap_g` has almost nothing left to
give there. Rewiring just 15% of the junctions cuts the diameter from 10 to 7 and
nearly doubles CF synchrony for free.

**If you want a more synchronous olive, shorten the paths; do not turn up the
conductance.**

**Heterogeneity does not affect synchrony at all.** Crossing cell-to-cell
variability with coupling (4 levels × 5 strengths × 6 seeds = 120 runs), Vm
correlation and CF synchrony are unchanged by heterogeneity at every coupling
strength — at `gap_g` = 0.0143 they read +0.540 / +0.547 / +0.529 / +0.447 for
cv = 0 / 0.05 / 0.10 / 0.20. What heterogeneity *does* control is the dispersion
of firing rates across cells, and coupling then removes it:

| cv | rate spread, uncoupled | rate spread at 1.9× leak |
|---|---|---|
| 0.00 | 0.080 Hz | 0.001 Hz |
| 0.05 | 0.111 Hz | 0.000 Hz |
| 0.10 | 0.168 Hz | 0.000 Hz |
| 0.20 | **0.283 Hz** | **0.003 Hz** |

So the two knobs do different jobs: **heterogeneity spreads the population's
rates, and gap junctions pull them back into a common rate.** That is a real and
physiologically sensible function for olivary coupling — enforcing a shared
rhythm across cells that would otherwise run at their own — and it is invisible
in any measurement that only looks at correlation.

---

## The IO neuron

The olive is the one cell type here that is not an integrate-and-fire
abstraction. H1 is a claim about a self-paced oscillator being *held* at a rate
by inhibition, and a cell with an imposed drive would assume the answer. So the
IO fires because of its own ionic currents (`sim/io_channels.py`):

| current | gating | role |
|---|---|---|
| `I_CaL` | `k³·l`, low-threshold ("T-type") | Inactivated at rest, de-inactivates on hyperpolarization, then produces the regenerative low-threshold Ca²⁺ spike that carries the CF burst. **This is the current that generates the rhythm.** |
| `I_CaH` | `r²`, high-threshold | Opens only on the Ca²⁺ spike; the Ca²⁺ source that loads `[Ca]i`. |
| `I_KCa` | gated by `[Ca]i`, not voltage | Repolarizes the Ca²⁺ spike and produces the long AHP that paces the oscillation and de-inactivates `I_CaL` for the next cycle. |
| `I_h` | `q`, activated by hyperpolarization | Depolarizing sag out of the AHP trough; sets oscillation frequency. |
| `I_leak` | — | Passive. |
| `I_GABA` | — | DCN inhibition, as a real Cl⁻ conductance (shunting *and* hyperpolarizing) rather than a subtracted scalar. |
| `I_gap` | — | IO–IO electrical coupling, as above. |

Gating kinetics — the voltage and `[Ca]` dependence, i.e. the actual physiology —
come from the standard olivary model lineage (Schweighofer, Doya & Kawato 1999,
*J Neurophysiol* 82:804; De Gruijl et al. 2012, *PLoS Comput Biol* 8:e1002814)
and are hard-coded, not tunable. Maximal conductances are re-fit in
`config.IOChannelParams`, because those papers are 3-compartment models and this
is a single-compartment reduction: kinetics transfer between the two,
conductance *densities* do not.

**What is deliberately not modeled:** the fast Na⁺/delayed-rectifier K⁺ spikes
that ride on top of the Ca²⁺ spike. The downstream circuit consumes only CF
*event times*, and resolving Na⁺ spikes would force a ~40× smaller timestep for
no change to anything measured here. A CF event is therefore detected as the
low-threshold Ca²⁺ spike itself, which is what actually triggers the somatic
burst in the real cell.

**IO firing is driven entirely by the cell's own currents.** `i_app = 0` — no
applied current, no bias, no tonic depolarizing drive — and `noise_sigma = 0` by
default, so no jitter either. Membrane noise remains available
(`IOChannelParams.noise_sigma`; 2.0 µA/cm² gives ~3.5 mV of Vm jitter) and is
what the model was originally fitted with. It is zero-mean, so it adds no
depolarization; what it does is decide *which* oscillation peaks reach threshold,
which smooths how steeply CF rate falls off with inhibition. The loop no longer
needs it, because the inhibition's own granularity supplies that smoothing (see
[the feedback limb](#the-feedback-limb-exists-because-inhibition-is-grainy)).

### The single-cell checks

`experiments/run_io_characterization.py` regenerates these; none involve the rest
of the circuit.

**Spontaneous rhythm**, `i_app = 0`, noiseless: **3.00 Hz**. V swings from about
−75 mV between spikes to above +75 mV at the Ca²⁺ spike peak, with `[Ca]i`
transients to ~2.7 µM.

**Channel ablation** (the modeller's pharmacology). Noiseless so each
contribution is unambiguous; the noisy column is there for contrast, because
noise lets a blocked cell still fire occasionally:

| condition | noiseless | with `noise_sigma = 2.0` | what it shows |
|---|---|---|---|
| control | **3.00 Hz** | 3.50 Hz | — |
| `I_CaL` blocked | **0.00 Hz** (silent, flat at −62.6 mV) | 0.75 Hz | the T-type current, not the leak, is the pacemaker |
| `I_KCa` blocked | 0.00 Hz | 0.00 Hz | nothing repolarizes the Ca²⁺ spike, so the cell latches into depolarization block at +110 mV |
| `I_h` blocked | 2.75 Hz | 3.38 Hz | the rhythm survives but slows — `I_h` sets frequency |

**Rebound burst** — an 800 ms hyperpolarizing step to −81 mV, then release,
produces a Ca²⁺ spike **7 ms later**, far sooner than the free-running period.
With `I_CaL` blocked the same step produces nothing. That contrast is the T-type
de-inactivation signature, and it is the actual test.

**Transfer curve** — CF rate falls monotonically with DCN inhibition, from 3.00 Hz
uninhibited to silence. Monotonicity is what H1's equilibrium argument rests on,
and `tests/test_io_channels.py` asserts it.

Integration is exponential Euler (`x ← x∞ + (x − x∞)·exp(−dt/τ)`,
unconditionally stable) on a 0.1 ms sub-timestep inside the 1 ms outer loop.
Halving that sub-timestep three times over (to 0.0125 ms) changes the mean
inter-spike interval by 0.33%, and the last halving by 0.04% — converged;
`tests/test_io_channels.py` asserts under 2% against a 4× finer step.

### The feedback limb exists because inhibition is grainy

This is the single most load-bearing detail in the model, and it is easy to miss.

With the default noiseless olive and the DCN conductance **held constant**, the
response to inhibition is all-or-nothing: **3.00 Hz at g = 0, and silence at
every g ≥ 0.10 mS/cm².** There is no graded range at all. A negative-feedback
loop cannot regulate against a cliff.

But only 8 nuclear cells feed each olivary cell, at ~15 Hz, into a 50 ms
conductance. What the cell actually sees therefore swings around its mean, and it
fires in the troughs. Delivered that way the same averages produce a smooth,
monotone, usable curve:

| DCN rate | 0 Hz | 6 Hz | 9 Hz | 12 Hz | **15 Hz** | 18 Hz | 21 Hz | 24 Hz | 30 Hz |
|---|---|---|---|---|---|---|---|---|---|
| mean `g_GABA` | 0.000 | 0.059 | 0.088 | 0.117 | **0.147** | 0.177 | 0.207 | 0.238 | 0.297 |
| CF rate, held constant | 3.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0.00 | 0.00 | 0.00 | 0.00 |
| CF rate, **spike-driven** | 3.00 | 2.48 | 2.09 | 1.59 | **1.05** | 0.51 | 0.21 | 0.06 | 0.01 |

The bold column is the loop's operating point. **That bottom row is the entire
negative-feedback limb, and it is a property of the synapse's granularity, not
of the mean it delivers.** `results/io_transfer_curve.png` plots both curves on
one axis. (Adding membrane noise back makes the held-constant curve gradeable
too, so noise and synaptic granularity are two routes to the same requirement;
the model relies on the second.)

Two consequences worth knowing before you retune anything:

- **There is an upper bound on nucleo-olivary convergence.** More converging
  cells deliver the same mean conductance in more, smaller events, which is
  *smoother* — and smoother pushes the cell back toward the held-constant cliff.
  This is a real prediction of the model: past some convergence the loop cannot
  regulate itself without another noise source.
- **`tau_dcn_io_ms` is deliberately short for the same reason.** Lengthening it
  averages the few DCN inputs into a steadier conductance and has the same effect
  as raising convergence.

**PKJ/DCN noise is load-bearing too, for a parallel reason.** A noiseless
integrate-and-fire cell has a hard rheobase, so its rate collapses from tens of
Hz to silence over a very narrow band of inhibitory conductance. The loop needs
DCN's rate to vary *smoothly* with PKJ drive, and the OU membrane noise
(`LIFParams.noise_sigma_mv`, 3 mV) is what linearizes the f-I curve around
threshold to give it that. `tests/test_neurons.py` pins this. Note this is a
**separate knob** from `IOChannelParams.noise_sigma`, which is off by default:
the olive has its own ionic machinery to generate firing, the integrate-and-fire
cells do not.

---

## Command reference

Every script takes `--help`. All of them accept `--out-dir` (default `results`)
and, where they simulate, `--seed` (default 0).

**Budgeting.** At the shipped scale one simulated second costs about **2.5 s of
wall-clock on one core** — a 300 s run took 784 s. Nearly all of that is the
olive's ionic sub-stepping, which is vectorized across cells, so `--n-io 4` is
much cheaper than 40 despite the same per-cell physics. Nothing here is
multi-threaded except `sweep.py`, which parallelizes across conditions.

### `experiments/run_network_snapshot.py` — the wiring, no simulation

| flag | default | meaning |
|---|---|---|
| `--n-io N` | 40 | Olivary cells. `n_dcn` follows as `2*n_io`, which the PKJ→DCN counts require; the whole microzone scales with it. |
| `--gap-topology T` | `nearest_k` | `all_to_all` \| `ring` \| `nearest_k` \| `small_world`. Changes the panel-3 web. |
| `--gap-g G` | 0.0143 | Conductance of **one** junction, mS/cm². `0` draws no junctions. |
| `--n-pf-shown N` | 8 | PF units drawn per Purkinje cell. All 500 is neither drawable nor a readable matrix; every other cell is shown in full. |
| `--open-loop` | off | `enforce_closed_loop=False`: rotate which block of the nucleus each olivary cell reads, misrouting the feedback without cutting it. |
| `--seed N` | 0 | Seeds only the drawn PF cloud's jitter, **not** the wiring (that is `connectivity_seed`). |

### `experiments/run_baseline.py` — H1 on one loop

| flag | default | meaning |
|---|---|---|
| `--duration-s S` | 60 | Simulated seconds after burn-in (~2.5 s wall-clock each). **Use 300+ before quoting a drift** — 60 s is still the transient. |
| `--gap-g G` | 0.0143 | One junction's conductance; `0` uncouples the olive. |
| `--n-tracked-synapses N` | 15 | Individual synapses plotted against the mean; `0` disables the lower panel. |

### `experiments/run_network.py` — the microzone, fully recorded

Everything `run_baseline.py` has, plus:

| flag | default | meaning |
|---|---|---|
| `--burn-in-s S` | 8 | Seconds run with plasticity and recording off, to settle the membrane state. |
| `--n-io N` | 40 | As in the snapshot: the whole microzone scales with it. |
| `--gap-topology T` | `nearest_k` | Compare topologies at **matched total conductance**, not at matched `gap_g`. |
| `--raster-window-s S` | 10 | How much of the run's tail the rasters show; the whole run is a solid block. |

### `experiments/run_io_characterization.py` — the olivary cell alone

| flag | default | meaning |
|---|---|---|
| `--transfer-duration-s S` | 8 | Simulated seconds per point on the two rate/conductance curves. Each point pools 12 identical uncoupled cells, so it is 12× that many cell-seconds. 26 points — this is where the runtime goes. |

### `experiments/run_calibration.py` — re-derive the three gains

| flag | default | meaning |
|---|---|---|
| `--fit-secs S` | 8 | Simulated seconds per bisection probe in stages 1–2 (16 steps each). |
| `--curve-secs S` | 15 | Simulated seconds per plotted curve point. |
| `--io-fit-secs S` | 25 | Simulated seconds per stage-3 probe. CF events are rare, so this stage needs far longer runs to resolve a rate. |
| `--skip-io` | off | Fit stages 1–2 only. Stage 3 dominates the runtime and the other two gains do not depend on it. |

It prints each fitted value next to what `config.py` currently holds, so drift
between them is visible at a glance. **It does not edit `config.py`.**

### `experiments/run_gap_sweep.py` — coupling strength vs. the loop

| flag | default | meaning |
|---|---|---|
| `--gap-g G [G ...]` | `0 0.008 0.02 0.03 0.05 0.08` | Single-junction conductances to sweep. One full network run each — the list length is the cost. |
| `--duration-s S` | 40 | Simulated seconds per point. Enough for the correlation measures, not for a settled weight. |
| `--trace-window-s S` | 8 | Vm window at the end of each run the subthreshold correlation is measured over. |
| `--topology T` | `nearest_k` | Held fixed across the sweep. Each topology gives a different partner count, so the same `gap_g` is a different **total** conductance. |

### `experiments/run_window_sweep.py` — H2, direct

| flag | default | meaning |
|---|---|---|
| `--windows MS [MS ...]` | `50 100 200` | LTD window lengths; one fit per window. |
| `--n-ratios N` | 6 | Candidate ratios per window, spread 0.2×–2.2× the analytical prediction. **One simulation each.** |
| `--duration-s S` | 30 | Simulated seconds per candidate. Short on purpose — the drift *sign* is what gets bracketed. |
| `--delta-plus D` | 0.001 | LTP increment, held fixed while `delta_minus` varies. |
| `--n-io N` | 4 | Olivary cells. 4 is CbmSim's own size and keeps the default sweep to ~30 min; 40 is the shipped network and ~10× that. |

### `experiments/run_three_window.py` — H3, direct

| flag | default | meaning |
|---|---|---|
| `--ltd-window-ms MS` | 100 | LTD window, shared by both arms. |
| `--null-window-ms MS` | 100 | Null window appended after it in the 3-window arm. Wider ⇒ steadier weights, lower CF rate. |
| `--delta-plus D` | 0.001 | LTP increment, held fixed while `delta_minus` is fitted. |
| `--search-duration-s S` | 30 | Simulated seconds per candidate ratio during the bracket search; 7 candidates × 2 arms. |
| `--eval-duration-s S` | 90 | Simulated seconds for the two final runs the spread is measured on. **Do not shorten this.** The weights start uniform and spread over time, so a short run compares transients rather than equilibria and can come out backwards. |
| `--n-io N` | 4 | As above. |
| `--n-tracked-synapses N` | 15 | `0` skips the figures entirely. |

### `experiments/sweep.py` — the replicated sweep

```bash
python experiments/sweep.py --list                  # the matrix and its run counts
python experiments/sweep.py --dry-run               # what is still outstanding
python experiments/sweep.py open_loop --workers 32  # one experiment
python experiments/sweep.py all --workers 60        # everything: 400 runs
```

| flag | default | meaning |
|---|---|---|
| `experiment` | `all` | Name, `all`, or a comma-separated list. |
| `--workers N` | cores − 4, capped at 60 | Parallel processes, one condition each. Each is single-threaded by design; letting 60 workers each spawn a BLAS thread pool oversubscribes the machine. |
| `--list` | — | Print the matrix and exit. |
| `--dry-run` | — | Report outstanding conditions and exit. |

The nine experiments, and what each one is for:

| experiment | runs | question |
|---|---|---|
| `settling` | 8 | How long does the loop take to settle? (600 s runs) |
| `open_loop` | 48 | The H1 falsification: cut DCN→IO and nothing else. |
| `initial_conditions` | 25 | Is the equilibrium an *attractor*? Five starting weights. |
| `dt_convergence` | 12 | Does the 1 ms outer loop change the answer? |
| `h2_window` | 80 | H2, predicted vs. measured, 8 settings × 10 seeds. |
| `h3_three_window` | 50 | H3, five null-window widths × 10 seeds. |
| `heterogeneity_coupling` | 120 | Heterogeneity × coupling, fully crossed. |
| `topology` | 32 | Topology at matched total conductance. |
| `pf_pool` | 25 | Sensitivity to the one scaled-down number. |

**The sweep is resumable.** A condition whose sidecar already exists is skipped,
so a crash, a reboot or a deliberate interrupt costs only the runs in flight.
Each sidecar carries the full resolved config, a SHA256 of `config.py` + `sim/*`,
the seeds, the wall time, and every scalar the run is judged on — so a result can
be re-read and re-plotted months later without re-running anything.

### `experiments/analyze_sweeps.py` and `experiments/figures.py`

Both read only `results/sweeps/**`; neither re-simulates.

| script | flag | default | meaning |
|---|---|---|---|
| `analyze_sweeps.py` | `experiment` | all | Restrict the report to one experiment. |
| `analyze_sweeps.py` | `--out PATH` | `results/SWEEP_RESULTS.md` | Markdown file to write. |
| `figures.py` | — | — | No flags: every panel is tied to a specific sweep experiment, so there is nothing to select. Exits with a message if the sweep has not been run. |

---

## Configuration reference

Everything lives in `config.py`, in three dataclasses. Each one **rejects
assignment to undeclared fields** — `cfg.duration_ms = 2000` raises rather than
silently creating an attribute the simulation never reads. That matters most for
`sweep.py`, where a typo would otherwise log a swept condition while running the
default.

Anything not listed as a CLI flag above is set by editing `config.py` or by
constructing `SimConfig(...)` directly:

```python
from config import SimConfig
from sim.simulate import Simulation

cfg = SimConfig(duration_s=300.0, seed=0, ltd_window_ms=50.0, delta_minus=0.009)
cfg.io_channels.noise_sigma = 2.0        # nested dataclasses are reached by attribute
log = Simulation(cfg).run()
```

### `SimConfig` — the circuit

**Timing**

| field | default | meaning |
|---|---|---|
| `dt_ms` | 1.0 | Outer timestep. The IO substeps internally at `io_channels.sub_dt_ms`. |
| `duration_s` | 60.0 | Run length if `run()` is not given one. |
| `burn_in_s` | 8.0 | Run before plasticity and logging start, so the loop reaches its dynamic equilibrium. Without it the olive overshoots from rest, inflicting a burst of spurious early LTD that permanently crashes the weights. |

**Populations and wiring** (see [The network](#the-network))

| field | default | meaning |
|---|---|---|
| `n_io` | 40 | Olivary cells. |
| `n_dcn` | 80 | Nuclear cells. |
| `n_pkj_per_io` | 8 | PKJ per climbing fiber; `n_pkj = n_io * n_pkj_per_io`. |
| `n_pkj_per_dcn` | 12 | PKJ converging on one DCN. |
| `n_dcn_per_pkj` | 3 | DCN reached by one PKJ. |
| `n_dcn_per_io` | 8 | DCN converging on one IO. **Below `n_dcn` this makes DCN→IO topographic**, which is what breaks the olive's degeneracy. |
| `n_pf_per_pkj` | 500 | Private Poisson fibers per Purkinje cell — the one scaled-down number. |
| `connectivity_seed` | 0 | Seeds the randomized half of the PKJ→DCN overlap. Independent of `seed`. |
| `enforce_closed_loop` | True | `False` rotates each olivary cell's block of the nucleus by a full block width, so it is inhibited by cells its own CF does not drive — misrouted, not removed. Needs `n_dcn_per_io < n_dcn` to mean anything. |
| `ablate_dcn_io` | False | **The H1 falsification.** Cuts DCN→IO: nuclear cells still spike and are still recorded, their GABA simply never reaches the olive. Everything else is untouched. |

`build_connectivity` raises if the counts do not close
(`n_pkj % n_dcn != 0`, or `n_pkj_per_dcn * n_dcn != n_dcn_per_pkj * n_pkj`), so a
bad combination fails immediately rather than silently rewiring.

**Input and cell baselines**

| field | default | meaning |
|---|---|---|
| `pf_rate_hz` | 20.0 | Per-fiber Poisson rate. |
| `pkj_baseline_hz` | 50.0 | PKJ rate with no synaptic input. |
| `dcn_baseline_hz` | 60.0 | DCN rate before PKJ inhibition — well above its ~15 Hz operating rate, deliberately: a LIF just above rheobase has a near-vertical f-I curve, and the nucleus has to respond *gradually* to PKJ. |
| `pkj`, `dcn` | `LIFParams()` | The Purkinje and nuclear membrane parameter sets (below). |
| `io_channels` | `IOChannelParams()` | The olivary cell's ionic parameter set (below). |

**Synapses**

| field | default | meaning |
|---|---|---|
| `e_exc_mv` | 0.0 | AMPA-like reversal (PF→PKJ). |
| `e_inh_mv` | −75.0 | GABA_A / Cl⁻ reversal (PKJ→DCN, and the CF pause). |
| `tau_pf_pkj_ms` | 5.0 | PF→PKJ conductance decay. |
| `tau_pkj_dcn_ms` | 10.0 | PKJ→DCN conductance decay. |
| `tau_dcn_io_ms` | 50.0 | DCN→IO decay, matching the olivary glomerulus' slow GABA response. **Deliberately short** — see [the feedback limb](#the-feedback-limb-exists-because-inhibition-is-grainy). |
| `pf_pkj_gain` | 0.000137 | Excitatory conductance per unit of (weight × PF spike). Fitted: PKJ 50 Hz at w=0, 70 Hz at w=1. |
| `pkj_dcn_gain` | 0.007717 | Inhibitory conductance per PKJ spike. Fitted: DCN 15 Hz when its 12 PKJ fire at 60 Hz. |
| `dcn_io_gaba_gain` | 0.0974 | GABA_A conductance per DCN spike. Fitted: IO 1 Hz at DCN 15 Hz. |
| `pkj_dcn_fitted_at_n_pkj` | 8 | The convergence `pkj_dcn_gain` was fitted at. |
| `dcn_io_fitted_at_n_dcn` | 2 | The convergence `dcn_io_gaba_gain` was fitted at. |
| `cf_pause_g` | 1.0 | Inhibitory conductance during the post-complex-spike pause — large enough to dominate `g_leak` and clamp the cell near `e_inh_mv`. |
| `cf_pause_ms` | 10.0 | Pause duration after each CF event. |

The two `*_fitted_at_*` fields are not cosmetic. `sim/simulate.py` scales each
convergent gain by `fitted_at / actual`, so the total inhibition a DCN (and an
IO) receives is independent of the population ratios. Without that, changing a
ratio would silently move the loop's operating point as well as its wiring, and
every result would need re-fitting for every topology. Note this equalizes the
**mean** only — it cannot equalize the graininess, which is exactly what the
olive's feedback limb depends on.

**Gap junctions**

| field | default | meaning |
|---|---|---|
| `gap_g` | 0.0143 | Conductance of **one** junction, mS/cm². `0` uncouples. The physiological quantity is `gap_g × partners`; 0.0143 × 4 = 0.0572, about 0.95× leak. |
| `gap_topology` | `nearest_k` | `all_to_all` \| `ring` \| `nearest_k` \| `small_world`. |
| `gap_n_neighbors` | 2 | Neighbours *per side* under `nearest_k`/`small_world`, so `2k` partners. Ignored by the others. |

**Weights and plasticity**

| field | default | meaning |
|---|---|---|
| `w_init` | 0.5 | Initial weight, all synapses. Swept in `initial_conditions`. |
| `w_min`, `w_max` | 0.0, 1.0 | Clip bounds. A run that pins against one is not an equilibrium; `summarize()` reports `weight_saturated`. |
| `ltd_window_ms` | 100.0 | A CF within this window after a PF spike depresses it. |
| `null_window_ms` | 0.0 | Window after the LTD window in which a CF causes no change. `0` = two-window mode; `> 0` is H3. |
| `delta_plus` | 0.001 | LTP increment. |
| `delta_minus` | 0.009 | LTD decrement. The ratio `δ−/δ+` and `ltd_window_ms` together set the equilibrium rate: `1000 / (window × (1 + ratio))`. |

**Heterogeneity**

| field | default | meaning |
|---|---|---|
| `io_heterogeneity_cv` | 0.0 | Coefficient of variation of the per-cell `g_cal`/`g_kca`/`g_h`. Variability **between** cells, not noise **within** one: drawn once at construction, so the population is fixed and the cells stay deterministic. It adds no drive, bias or jitter. |
| `io_heterogeneity_seed` | 0 | Seeds that draw, independently of `seed` and `connectivity_seed`, so one cell population can be re-used across input seeds. |

**Recording**

| field | default | meaning |
|---|---|---|
| `record_every_ms` | 10.0 | Cadence for the slow traces (mean weight, tracked synapses). |
| `trace_window_s` | 3.0 | Length of the full-resolution membrane-potential window, taken at the **end** of the run. |
| `n_pf_recorded` | 40 | PF units whose spike trains are kept for the raster, of 160,000. |
| `n_tracked_synapses` | 15 | Individual PF→PKJ synapses logged alongside the mean. |
| `seed` | `None` | Master RNG seed. PF draws, IO noise, PKJ/DCN noise and the recording subsamples each get their own derived stream, so changing one never shifts the others. |

### `IOChannelParams` — the olivary cell

Gating kinetics are hard-coded in `sim/io_channels.py` from the published models
and are **not** meant to be tuned. What lives here is what a single-compartment
reduction legitimately has to re-fit. Units: V in mV, g in mS/cm², I in µA/cm²,
`[Ca]i` in µM, time in ms.

| field | default | meaning |
|---|---|---|
| `g_cal` | 1.1 | Low-threshold (T-type) Ca²⁺ — the rebound/oscillation driver. |
| `g_cah` | 1.0 | High-threshold Ca²⁺ — the `[Ca]i` source that recruits `I_KCa`. |
| `g_kca` | 15.0 | Ca²⁺-activated K⁺ — spike repolarization and the pacing AHP. |
| `g_h` | 0.30 | h current — depolarizing sag; sets oscillation frequency. |
| `g_leak` | 0.06 | Passive leak. |
| `e_ca`, `e_k`, `e_h`, `e_leak`, `e_gaba` | 120, −75, −43, −60, −75 | Reversal potentials. The −60 mV leak is a departure from the source models' depolarizing leak, which had made the *leak* the pacemaker. |
| `c_m` | 1.0 | Membrane capacitance. |
| `i_app` | 0.0 | Applied current. **Zero: the cell is driven by its own channels alone.** |
| `ca_influx` | 0.003 | µM per (µA/cm²) of `I_CaH` per ms. Scaled 1000× from the source models (with `kca_ca_scale` up 1000× to match, leaving dynamics identical) purely so `[Ca]i` reads in real units. |
| `ca_decay` | 0.075 | First-order extrusion rate, 1/ms (τ ≈ 13 ms). |
| `kca_ca_scale`, `kca_alpha_max`, `kca_beta` | 0.02, 0.01, 0.015 | `I_KCa`'s `[Ca]` dependence. |
| `cal_tau_k_ms` | 1.0 | `I_CaL` activation time constant. |
| `noise_sigma` | **0.0** | SD of the OU membrane-noise current. Zero by default: firing is generated entirely by the cell's own currents. 2.0 gives ~3.5 mV of Vm jitter. |
| `noise_tau_ms` | 5.0 | OU correlation time. |
| `v_spike_mv` | −35.0 | Upward crossing = one Ca²⁺ spike = one CF event. |
| `spike_refractory_ms` | 15.0 | Stops one broad Ca²⁺ spike being counted as several CF events. |
| `v_init_mv`, `ca_init_um` | −60.0, 0.0 | Initial state. |
| `sub_dt_ms` | 0.1 | Internal integration step. Converged: halving three times moves the mean ISI by 0.33%. |
| `display_tau_ms` | 3000.0 | Smoothing for the **logged rate estimate only**; nothing in the dynamics reads it. |

### `LIFParams` — the Purkinje and nuclear cells

One instance each, at `cfg.pkj` and `cfg.dcn`. None of these come from the white
paper; they are conventional cortical values. What was actually fitted for this
circuit is the noise level and the three gains in `SimConfig`.

| field | default | meaning |
|---|---|---|
| `tau_m_ms` | 20.0 | Membrane time constant. `C = 1`, so `g_leak = 1/tau_m`. |
| `v_th_mv` | −50.0 | Spike threshold. |
| `v_reset_mv` | −65.0 | Post-spike reset. |
| `e_leak_mv` | −70.0 | Resting potential with no input. |
| `t_ref_ms` | 2.0 | Absolute refractory period. |
| `v_peak_mv` | 0.0 | **Not dynamics**: the value a spike is drawn at in logged V traces, since an integrate-and-fire cell resets rather than producing an upstroke. |
| `noise_sigma_mv` | 3.0 | OU membrane noise. **Load-bearing** — it is what keeps the f-I curve smooth near rheobase and so keeps the loop's feedback graded. |

---

## What gets recorded

`run_network.py` collects, and plots:

| file | contents |
|---|---|
| `network_rasters.png` | PF / PKJ / DCN / IO spike rasters; PF and PKJ coloured by climbing fiber |
| `network_weights.png` | Mean PF→PKJ weight per climbing-fiber territory, plus tracked individual synapses |
| `network_voltages.png` | Membrane potentials for every cell type, plus the GABA conductance each IO is receiving |
| `network_io_state.png` | IO V and `[Ca²⁺]ᵢ` together |
| `network_3d.png` | The wired network in 3D |
| `connectivity_matrix.png` | The adjacency matrix that wiring induces |

The other scripts add `gap_coupling_sweep.png`, `baseline_{rasters,voltages,weights}.png`,
`calibration_curves.png`, `io_{traces,channel_ablation,rebound,transfer_curve}.png`,
and the two `window_sweep_*.png` / two `three_window_*.png` files. Sweep runs
write one JSON sidecar each under `results/sweeps/`, plus compressed `.npz` raw
arrays for headline conditions.

**Spike trains are exact and complete** — every spike of every cell, at full
timestep resolution — and are the primary record; firing rates are counted from
them rather than read off smoothed traces. Two things are deliberately
subsampled, for readability rather than cost:

- the PF raster keeps `n_pf_recorded` (40) of the 160,000 fibers;
- membrane potentials are recorded at full resolution only over a window at the
  end of the run (`trace_window_s`, 3 s). Sampling V at the slow logging cadence
  aliases every spike away, and sampling it at 1 ms for a whole run is a lot of
  memory for a plot no one can read.

---

## What's implemented, and how

| module | what it holds |
|---|---|
| `config.py` | All parameters and defaults, with a note on each saying whether it is paper-specified, ported from CbmSim, or fitted |
| `sim/poisson_input.py` | PF spike generation: Bernoulli-per-bin per fiber, one private pool per Purkinje cell |
| `sim/neurons.py` | PKJ / DCN conductance-based integrate-and-fire, plus the CF pause |
| `sim/io_channels.py` | The conductance-based olivary cell: `I_CaL` / `I_CaH` / `I_KCa` / `I_h` / leak / GABA / gap |
| `sim/io_coupling.py` | IO↔IO gap junctions: the four topologies and the synchrony metric |
| `sim/connectivity.py` | CbmSim's microzone wiring, ported |
| `sim/plasticity.py` | The coincidence-detection LTD / LTP / null rule |
| `sim/network_graph.py` | The wiring as a positioned, typed graph + its adjacency matrix |
| `sim/network_viz.py` | The 3D snapshot and connectivity-matrix figures |
| `sim/simulate.py` | The main time-stepped loop and the recording layer |
| `sim/analysis.py` | Spike-based metrics + raster / trace / weight plots |

A few implementation choices worth knowing:

- **PKJ / DCN** (`sim/neurons.py`) integrate
  `C dV/dt = −g_L(V−E_L) − g_exc(V−E_exc) − g_inh(V−E_inh) + I_tonic` by
  exponential Euler, with single-exponential synaptic conductances. Baselines are
  configured as *rates*: `tonic_drive_for_rate` inverts the LIF f-I curve, so
  `pkj_baseline_hz = 50` really means 50 Hz. The CF pause is modeled as a large
  inhibitory **conductance**, so that (like the real post-complex-spike pause) it
  both hyperpolarizes the cell and shunts the PF excitation arriving during it.
- **Plasticity** (`sim/plasticity.py`) is retrospective: a PF spike is resolved
  `ltd_steps + null_steps` later, once it is known whether a CF landed in its LTD
  window, its null window, or neither. It is implemented as a circular buffer plus
  *cumulative* CF counters rather than a per-spike event queue — whether a CF fell
  in a window is a subtraction of two counters, O(1) per timestep regardless of
  how many synapses spiked, and exactly equivalent to scheduling each spike
  individually. At this scale a queue would have to schedule and retire about
  3.2 million events per simulated second (160,000 synapses at 20 Hz).
- **The main loop** (`sim/simulate.py`) runs PF → PKJ → DCN → IO → CF/plasticity,
  each population reading its upstream source's *just-updated* spikes — a
  semi-implicit ordering choice that propagates a step's effect around the whole
  loop in one step of latency instead of four. Populations are flat and each
  projection is one matrix multiply against a connection table.
- **The network graph** (`sim/network_graph.py`) is built from the same
  `Connectivity` and gap-junction matrix the simulation runs on, so the picture is
  of the network that actually ran. `tests/test_network_graph.py` asserts the
  graph's edge sets equal the simulation's own routing matrices.

### Where the runtime goes

Nearly all of it is the olive's ionic integration, which sub-steps at 0.1 ms
inside the 1 ms outer loop. That inner loop is **vectorized across cells**, so
forty olivary neurons cost about the same as one — which is why the scripts that
drive a *single* cell for a long time (characterization, calibration) are the
slowest things here despite simulating the least.

---

## Assumptions the paper does not specify

The white paper specifies population ratios and plasticity defaults (both
implemented as given), but *not* a membrane model or coupling gains. Those were
fitted, and `experiments/run_calibration.py` is how — it measures each stage in
isolation, driven by a surrogate of its input, and reports the gain that hits the
target, so the constants in `config.py` can be re-derived after any change rather
than being magic numbers.

| gain | fitted so that | value |
|---|---|---|
| `pf_pkj_gain` | PKJ runs 50 Hz at weight 0 and 70 Hz at weight 1 | 0.000137 |
| `pkj_dcn_gain` | DCN sits at 15 Hz when its 12 PKJ fire at 60 Hz | 0.007717 |
| `dcn_io_gaba_gain` | IO fires 1 Hz at DCN = 15 Hz | 0.0974 |

Each stage's curve passes through its target. Re-running `run_calibration.py`
against the shipped config reproduces all three values to the digits stored
(0.000137 / 0.007717 / 0.097437) and gives:

| stage | measured |
|---|---|
| PKJ vs. weight | 49.6 Hz at w=0, 60.1 at w=0.5, **70.1 at w=1** |
| DCN vs. PKJ | 26.9 / 22.8 / 18.8 / **15.1** / 11.7 / 8.7 / 6.2 Hz at PKJ 45→75 Hz — slope −0.69 Hz/Hz |
| IO vs. DCN | 2.61 / 1.99 / **1.01** / 0.27 / 0.04 / 0.01 Hz at DCN 5→30 Hz |

The IO stage is the one worth looking at: smooth and monotone across the whole
range the loop visits, which is exactly what H1's equilibrium argument needs and
what a held conductance would not give (see above).

`dcn_io_gaba_gain` has been re-fitted twice, and the sequence is informative:
**0.68** with membrane noise and 2 converging DCN, **0.273** once the noise went
to zero, **0.0974** once the convergence rose to 8. Each step needs *less*
inhibition per synapse, for the same underlying reason — how readily the cell
escapes inhibition depends on how *grainy* that inhibition is, and both changes
made it smoother. `run_calibration.py` reads the live config and mirrors the real
convergence, so re-running it after any topology or noise change re-derives the
right value rather than going stale.

---

## Testing

```bash
python -m unittest discover -s tests      # 97 tests, ~8 min
```

| file | what it pins |
|---|---|
| `tests/test_plasticity.py` | Window logic and boundaries, both window schemes |
| `tests/test_neurons.py` | The f-I inversion, the membrane model, the CF pause, and that PKJ/DCN noise really does linearize the f-I curve |
| `tests/test_io_channels.py` | Gating functions, substep convergence against a 4× finer step, cell behaviour (rhythm, rebound, monotone response to inhibition), gap junctions, and CF routing |
| `tests/test_network_graph.py` | That the drawn graph equals the simulated wiring — right projections, right directions, right routing — and that all `n_io` inhibition patterns are distinct |
| `tests/test_perturbations.py` | That heterogeneity makes cells *differ* without making them *noisy*, and that ablation removes the feedback limb and nothing else |

The suite is slow because several tests run short closed-loop simulations rather
than mocking them — the properties being asserted are emergent, and a mock could
not exhibit them.

---

## Non-goals and explicit simplifications

- **No mossy fibers, granule cells, or Golgi cells.** PF spikes are generated
  directly as Poisson processes, in private per-Purkinje-cell pools.
- **No spatial or topographic organization beyond CbmSim's connection tables.**
  The 3D positions in `network_graph.py` are for drawing; nothing in the dynamics
  reads them, and no connection depends on distance.
- **The IO is single-compartment** — no separate soma/dendrite/axon, and no fast
  Na⁺/K⁺ spikes. Gap junctions are somatic, whereas real olivary coupling is
  dendritic and inside the glomerulus.
- **CF bursts are single timestamped events**, not variable-spike-count bursts:
  the Ca²⁺ spike is resolved, the Na⁺ spikes riding on it are not.
- **PKJ and DCN have no distinct cell classes**, no spike-frequency adaptation,
  and no plasticity of their own. Only PF→PKJ is plastic.
- **§2.2(1) of the spec is satisfied only approximately** — 27% of the microzone
  rather than 100%. See [What this costs](#what-this-costs-the-closed-loop-constraint).
