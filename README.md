# io_equilibrium_sim

A small, CPU-only Python simulation of a simplified olivo-cerebellar
microcircuit, built to test hypotheses from an internal Mauk Lab white paper on
Inferior Olive (IO) function. Architecturally inspired by
[CbmSim](https://github.com/mmauk/CbmSim) (Mauk Lab, UT Austin) but **not** a
port of it — no CUDA/C++, and no granule-cell/mossy-fiber front end (see
"Non-goals" below).

Full spec: `io_equilibrium_sim_MVP_spec.md` (project owner's original spec; this
README summarizes what was actually built and any assumptions made where the
spec left implementation details open).

## The network

The microzone is CbmSim's, with its connection numbers taken directly from
`src/cbm_state/connectivityparams.cpp` rather than re-derived, and its
populations scaled up 10x:

```
       40 IO, 80 DCN, 320 PKJ,  160,000 PF units

  PF ---(+, plastic)--> PKJ ---(-)--> DCN ---(-)--> IO ---(CF)--> PKJ
                                                     ^                |
                                                     |                |
                                          IO <--> IO gap junctions    |
                                                                      |
                              (CF also gates LTD at the PF synapses) -+
```

**Populations are scaled, per-cell convergence is not.** Every number in the
lower block below is CbmSim's own, unscaled: each Purkinje cell still receives
one climbing fiber and contacts 3 nuclear cells, each nuclear cell still
integrates 12 Purkinje cells, each olivary cell still integrates 8 nuclear cells.
Because convergence sets each cell's synaptic load, the operating point carries
over unchanged and **no synaptic gain needed refitting** for the scale-up.

| quantity | CbmSim | here |
|---|---|---|
| olivary cells | `num_io` = 4 | **40** (x10) |
| nuclear cells | `num_nc` = 8 | **80** (x10) |
| Purkinje cells | `num_pc` = 32 | **320** (x10) |
| PKJ per climbing fiber | `num_p_io_from_io_to_pc` = 8 | **8** |
| PKJ converging on one DCN | `num_p_nc_from_pc_to_nc` = 12 | **12** |
| DCN reached by one PKJ | `num_p_pc_from_pc_to_nc` = 3 | **3** |
| DCN converging on one IO | `num_p_io_from_nc_to_io` = 8 | **8** of 80 — see below |
| IO reached by one DCN | `num_p_nc_from_nc_to_io` = 4 | **4** |
| IO-IO coupling partners | `num_p_io_in_io_to_io` = 3 | **4** (local, not all-to-all) |
| PF onto one PKJ | `num_p_pc_from_gr_to_pc` = 32,768 | **500** — scaled down |

Two entries change meaning at this scale, and both changes are forced:

**DCN->IO stops being complete.** At CbmSim's size, 8 nuclear cells converging on
each olivary cell *is* the whole nucleus — `connectNCtoIO` is complete bipartite.
Holding that convergence at 8 while the nucleus grows to 80 makes the projection
topographic: olivary cell `i` reads a contiguous block of the nucleus whose start
advances with `i`, so neighbouring cells share 6 of their 8 inputs and distant
cells share none. Every nuclear cell still projects to exactly 4 olivary cells.
This is not cosmetic — see "Why the olive was perfectly synchronized" below.

**IO-IO coupling stops being all-to-all.** What matters physiologically is a
cell's *total* coupling conductance, not the per-junction value. All-to-all on 40
cells means 39 partners and ~12x the leak conductance, which would clamp the
entire olive to a single potential. The topology is local instead
(`nearest_k`, 4 partners) with `gap_g` set so the total per cell is 0.0572
mS/cm^2 — the same 0.95x leak the 4-cell build had.

The wiring algorithms are ported too, not just the counts: `connectPCtoNC`'s
topographic block plus randomized overlap, `connectNCtoIO`, `connectIOtoPC`'s
contiguous climbing-fiber blocks, and `connectIOtoIO`. Two of them are used in
their generalized form, because the scale-up takes them outside the regime where
CbmSim's version is defined: `connectNCtoIO` becomes topographic rather than
complete once `n_dcn_per_io < n_dcn`, and the coupling is local rather than
all-to-all (both explained above). See `sim/connectivity.py`, which documents
the two places CbmSim's own version does not close cleanly (a retry loop that can
write a duplicate synapse, and a mop-up that dumps leftovers onto the last
nuclear cell) and what is done instead so the realized counts match the
configured ones exactly.

**The one scaled-down number** is the granule input. CbmSim gives each Purkinje
cell 32,768 of a million shared granule cells; this MVP gives it a private pool
of 500 Poisson fibers. That is the spec's own simplification — private pools keep
each Purkinje cell's coincidence detection independent, which is what H2
measures — and it is the only place the connectivity departs from the reference.

**These numbers satisfy the white paper's ordering for free.** 4 < 8 < 32 gives
`N_IO ≪ N_DCN ≪ N_PKJ` (40 < 80 < 320); every nuclear cell diverges to 4 olivary
cells, and every olivary cell is inhibited by 8 of the 80. The counts are
mutually consistent (320 PKJ × 3 targets = 960 = 80 DCN × 12 inputs), so nothing
has to be assumed that the paper does not state — which was not true of the ratios inferred earlier.

**What this costs is the closed-loop constraint — less than it used to.**
Spec §2.2(2) — every PKJ a climbing fiber contacts helps regulate that same IO —
**holds for all 320 Purkinje cells**, and now holds structurally rather than
trivially. The two topographic maps line up: a Purkinje cell's own nuclear
target always falls inside the block of the nucleus that its climbing fiber's
olivary cell reads back from. (On the unscaled build this was true only because
DCN→IO was complete, i.e. every PKJ influenced every IO.)

§2.2(1) — "an IO may only be modulated by DCN modulated by exactly the PKJ its
own CF contacts" — still does not hold strictly: each olivary cell is influenced
by ~88 Purkinje cells, of which only its own 8 CF targets are sanctioned. But
that is **27% of the microzone rather than 100%**, so the violation is now one of
degree rather than of kind, and it shrinks further as the nucleus grows relative
to `n_dcn_per_io`. The loop is also no longer un-openable: because
`n_dcn_per_io < n_dcn`, `enforce_closed_loop=False` can genuinely rotate which
block of the nucleus each olivary cell reads, misrouting the feedback without
cutting DCN→IO outright. At CbmSim's own 4-cell ratios there was nothing to
rotate.

## Hypotheses under test

- **H1 — triple-negative equilibrium feedback.** The closed loop
  PKJ ⊣ DCN ⊣ IO, with IO's climbing-fiber (CF) output driving LTD at active
  PF→PKJ synapses, should hold IO firing and PF→PKJ weights at a stable
  equilibrium. **Holds** under the spiking network: see "Results" below.
- **H2 — LTD/LTP window asymmetry must balance at equilibrium.** Because the
  LTD window (~100 ms) is much shorter than the average inter-CF interval at
  equilibrium (~1000 ms at 1 Hz), LTD magnitude must be ~9x LTP magnitude for
  weights to balance on average, shifting as
  `ratio ≈ (1000 - window_ms) / window_ms` when the window changes.
  **Validated** at the current scale: 8 settings x 10 seeds, every one within
  0.02 Hz of prediction (see Results).
- **H3 (stretch) — three-window plasticity.** A LTD → null → LTP variant should
  let δ+/δ− be closer together and slow the weights' random-walk drift, at the
  cost of occasionally blocking a needed LTP event.
  **Validated**: the null window does slow the weights' random walk, monotonically
  across 5 levels x 10 seeds, at a measurable cost in rate control (see Results).

## Install & quick start

```bash
pip install -r requirements.txt

python -m unittest discover -s tests            # unit tests

python experiments/run_network_snapshot.py      # the wiring, in 3D (no simulation; instant)
python experiments/run_network.py               # the microzone, fully recorded
python experiments/run_gap_sweep.py             # what the IO-IO coupling does to the loop
python experiments/run_baseline.py              # H1 on a single loop
python experiments/run_calibration.py           # re-derive the synaptic gains
python experiments/run_io_characterization.py   # IO channel physiology, single cell
python experiments/run_window_sweep.py          # H2, single-condition sweep
python experiments/sweep.py all --workers 60    # the full 400-run sweep
python experiments/analyze_sweeps.py            # regenerate SWEEP_RESULTS.md
```

Each script accepts `--help`. Plots land in `results/` by default.

Runtimes on this machine (single-threaded CPU, no GPU):

| command | wall time |
|---|---|
| `run_network_snapshot.py` | instant — it runs no simulation |
| `run_network.py` (60 s simulated) | ~100 s |
| `run_baseline.py` (60 s simulated) | ~100 s |
| `python -m unittest discover -s tests` | ~7 min |
| `run_gap_sweep.py`, `run_calibration.py`, `run_io_characterization.py` | several minutes each |

Nearly all of it is the IO's ionic integration, which sub-steps at 0.1 ms inside
the 1 ms outer loop. That inner loop is vectorized across cells, so eight
olivary neurons cost about the same as one, so the scripts that drive a *single*
cell for a long time (the characterization and calibration protocols) are the
slowest things here.

## What gets recorded

`run_network.py` collects, and plots:

| file | contents |
|---|---|
| `network_rasters.png` | PF / PKJ / DCN / IO spike rasters; PF and PKJ coloured by climbing fiber |
| `network_weights.png` | mean PF→PKJ weight per climbing-fiber territory, and tracked synapses |
| `network_voltages.png` | membrane potentials for every cell type, plus the GABA conductance each IO is receiving |
| `network_io_state.png` | IO V and [Ca²⁺]ᵢ together |
| `network_3d.png` | the wired network in 3D |
| `connectivity_matrix.png` | the adjacency matrix that wiring induces |

The other scripts add: `gap_coupling_sweep.png` (`run_gap_sweep.py`);
`baseline_rasters/_voltages/_weights.png` (`run_baseline.py`);
`calibration_curves.png` (`run_calibration.py`); and `io_traces.png`,
`io_channel_ablation.png`, `io_rebound.png`, `io_transfer_curve.png`
(`run_io_characterization.py`). `run_window_sweep.py` and `run_three_window.py`
write theirs too. Sweep runs instead write one JSON sidecar each under
results/sweeps/, plus compressed .npz raw arrays for headline conditions.

Spike trains are **exact and complete** — every spike of every cell, at full
timestep resolution — and are the primary record; firing rates are counted from
them rather than read off smoothed traces. Two things are deliberately
subsampled, for readability rather than cost: the PF raster keeps
`n_pf_recorded` (40) of the 160,000 fibers, and membrane potentials are recorded
at full resolution only over a window at the end of the run
(`trace_window_s`, 3 s) — sampling V at the slow logging cadence aliases every
spike away, and sampling it at 1 ms for a whole run is a lot of memory for a
plot no one can read.

## Results

![H1](results/fig1_h1_equilibrium.png)
![H2 and H3](results/fig2_plasticity.png)
![coupling](results/fig3_coupling.png)
![robustness](results/fig4_robustness.png)

> The four figures above are generated by `experiments/figures.py` from the sweep
> sidecars and the saved raw trajectories; re-run it any time without re-simulating.
> Single-run figures (`run_network.py`, `run_baseline.py`, `run_gap_sweep.py`) are
> regenerated by their own scripts.

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

Drift is zero to five decimals: this is a settled equilibrium, not a slow trend.
**The 90 s runs this README used to quote were not converged** — they showed
−0.00085 units/s, and the loop was still descending. Settling takes a few
hundred seconds; report equilibria from runs of 300 s or more.

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
driven onto the floor. The result is identical with and without heterogeneity.
This is the control that distinguishes a real feedback equilibrium from a
coincidental operating point, and the loop passes it.

### H1, the stronger form: the equilibrium is an attractor

An operating point is not an equilibrium unless the system *returns* to it.
Runs started from five different initial weights (180 s, 5 seeds each):

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

### Numerical validity: the 1 ms outer loop is fine

The IO substep was already checked against a 4× finer one; the outer loop that
the plasticity windows, synaptic decay and the whole feedback limb run on was
not. Halving and quartering it:

| `dt_ms` | IO / CF rate | mean weight | PKJ |
|---|---|---|---|
| 1.0 | 0.981 ± 0.040 | 0.366 ± 0.030 | 56.1 |
| 0.5 | 1.003 ± 0.022 | 0.391 ± 0.011 | 56.9 |
| 0.25 | 0.996 ± 0.009 | 0.421 ± 0.011 | 57.6 |

**CF rate is dt-invariant** across a 4× refinement — the quantity H1 and H2 are
about does not move. The weight it settles at does drift mildly (0.366 → 0.421,
about 13%), tracking a 2.7% rise in PKJ rate, so absolute weights should be read
as dt-dependent to ~15% while rates should not.

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
was 1.000 because the correlation of a signal with itself is 1.000, and the gap
junctions had nothing to act on -- current flows between coupled cells only when
their potentials *differ*, and here the difference was identically zero, so
`I_gap` was zero at every timestep regardless of `gap_g`.

**The 10x scale-up fixes this, and not by accident.** Condition (1) is the one
that breaks: holding `n_dcn_per_io` at CbmSim's 8 while the nucleus grows to 80
makes the nucleo-olivary projection topographic instead of complete, so no two
olivary cells share an inhibition pattern (neighbours share 6 of 8, distant cells
share 0). The cells now diverge on their own, and the gap junctions have real
differences to work against. `tests/test_network_graph.py` locks this in: it
asserts all `n_io` inhibition patterns are distinct, so the degeneracy cannot
come back unnoticed.

Conditions (2) and (3) are untouched -- the olive is still deterministic, still
uniformly parameterised, and still has no applied current, bias, or jitter.

### H2: the plasticity window sets the equilibrium rate

H1 and H2 are one claim: the loop settles where the plasticity balances, at an
inter-CF interval of `window x (1 + δ−/δ+)`. Eight settings, 120 s, **10 seeds
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
loop tracks the product `window x (1 + ratio)` and is indifferent to how that
product is reached, which is what H1 and H2 together predict and what a
coincidence would not produce.

**The usable range now extends to at least 2 Hz.** An earlier version of this
README reported compression above ~1.2 Hz, with the 2 Hz targets landing at
1.85 Hz and weights pinned near the `w_max` ceiling. That was a property of the
8-IO build, whose equilibrium sat at w ≈ 0.6. At this scale the equilibrium sits
at w ≈ 0.37, leaving enough headroom that both 2 Hz conditions settle at w ≈ 0.57
with `weight_saturated` false in all 20 runs.

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
window blocks LTP events that the loop needs, so the equilibrium rate drops
steadily (1.008 → 0.838 Hz) as the window widens. A null window is therefore a
trade, not an improvement: it buys weight stability and pays in rate control.

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
`n_pf` and the loop still finds its rate. The shipped 500 sits near the middle of
that range.

The model does break outside it. At 125 fibers the loop **cannot** reach 1 Hz:
even with every synapse pinned at `w_max` there is not enough excitatory drive,
so the weights saturate and CF rate collapses to 0.25 Hz. At 2000 the weight
(0.088) is approaching the floor and the rate begins to creep up. So 500 is a
defensible choice rather than an arbitrary one, but the result to quote is the
*rate*, which is robust, not the *weight*, which is a function of pool size.

## What's implemented, and how

- **PKJ / DCN** (`sim/neurons.py`): conductance-based leaky integrate-and-fire,
  `C dV/dt = -g_L(V-E_L) - g_exc(V-E_exc) - g_inh(V-E_inh) + I_tonic`, with
  single-exponential synaptic conductances and exponential-Euler integration.
  Baseline rates stay the config knobs they were: `tonic_drive_for_rate` inverts
  the LIF f-I curve, so `pkj_baseline_hz = 50` still means 50 Hz.
  PKJ additionally gets the CF-triggered pause, modeled as a large inhibitory
  *conductance* so that (like the real post-complex-spike pause) it both
  hyperpolarizes the cell and shunts the PF excitation arriving during it.
- **IO** (`sim/io_channels.py`): conductance-based and Hodgkin-Huxley style —
  see "The IO neuron" below. Held as an `IOPopulation` integrated in lockstep,
  because gap junctions couple the cells within a sub-timestep.
- **IO↔IO gap junctions** (`sim/io_coupling.py`): see "Electrical coupling".
- **PF input** (`sim/poisson_input.py`): independent Bernoulli-per-bin
  approximation of a Poisson process per PF unit, one private pool per PKJ cell.
- **Connectivity** (`sim/connectivity.py`): CbmSim's microzone wiring, ported —
  explicit connection tables built by the same algorithms, with the counts taken
  from `connectivityparams.cpp`. See "The network" above for what is copied and
  the one thing (granule fan-in) that is scaled down.
- **Plasticity** (`sim/plasticity.py`): PF→PKJ synapses undergo LTD if a CF
  event lands within `ltd_window_ms` after the PF spike, LTP otherwise (or "no
  change" in the null window, for the 3-window/H3 variant). Implemented as an
  O(1)-per-timestep circular buffer plus cumulative CF counters rather than a
  per-spike event queue — see the module docstring for why that is equivalent.
  Each Purkinje cell resolves against the one climbing fiber that contacts it
  (`cf_source_of_pkj`). Window boundaries are inclusive on the near edge in the
  sense pinned by `tests/test_plasticity.py`.
- **Main loop** (`sim/simulate.py`): PF → PKJ → DCN → IO → CF/plasticity, each
  population reading its upstream source's *just-updated* spikes — a
  semi-implicit ordering choice that propagates a step's effect around the whole
  loop in one step of latency instead of four. Populations are flat and each
  projection is one matrix multiply against the connection table.
- **Network graph** (`sim/network_graph.py`, `sim/network_viz.py`): the wiring as
  an explicit graph of typed, positioned nodes and typed edges, with the
  adjacency matrix it induces. Built from the same `Connectivity` and
  gap-junction matrix the simulation runs on, so the picture is of the network
  that actually ran; `tests/test_network_graph.py` asserts the graph's edge sets
  equal the simulation's own routing matrices.

### Electrical coupling

`sim/io_coupling.py`. A gap junction is an ohmic resistor between two somata, so
the current into cell *i* is

```
I_gap(i) = sum_j  g_gap[i,j] * (V_j - V_i)
```

with `g_gap` symmetric and zero on the diagonal. This is the rule CbmSim applies
to its own IO array — `vCoupleIO[i] += coupleRiRjRatioIO * (vIO[j] - vIO[i])`
summed over each cell's partners in `MZone::updateIOOut`, with `connectIOtoIO`
wiring every cell to every other — so `all_to_all` is the direct analogue of
CbmSim's default and `gap_g` plays the role of `coupleRiRjRatioIO`. Two
deliberate differences:

- CbmSim adds the coupling as a voltage increment in arbitrary units on top of a
  leaky-integrate-and-fire IO. Here it is a real conductance in mS/cm² entering
  the conductance-based cell's current balance, so it shunts as well as pulls —
  which is what a gap junction physically does.
- CbmSim computes the term in `updateIOOut`, i.e. from the *previous* timestep's
  voltages. Here it is evaluated inside the 0.1 ms sub-timestep, at the same
  instant as every other current. The Ca²⁺ spike upstroke moves V by tens of mV
  in a millisecond, so a 1 ms lag would materially mis-state the current flowing
  during exactly the event the coupling is supposed to synchronize.

The conductance splits exactly into `(g_gap @ V) - (row sums) * V`, so it folds
into the same exponential-Euler update as every other conductance — no separate
explicit term, no stability penalty. That is *why* the cells are integrated as a
population: cell *i*'s update needs every other cell's V at the same instant.

Three topologies are built (`gap_topology`): `all_to_all` (CbmSim's default),
`ring` (2 partners), and `nearest_k` (2k partners each). **The shipped topology
is `nearest_k` with k=2, not all-to-all**, because all-to-all does not survive
the 10x scale-up: on 40 cells it means 39 partners per cell and a total coupling
conductance of ~12x leak, which would clamp the whole olive to one potential.
Gap junctions in the olive are local contacts within a glomerulus, so a local
topology is also the more faithful choice at this size. `gap_g = 0.0143` across
4 partners gives 0.057 mS/cm², about 0.95x the leak conductance — the same
per-cell load the 4-cell all-to-all build had, which is the quantity that
actually matters physiologically.

**What it does, and what it costs.** `experiments/run_gap_sweep.py` measures
both, in the assembled closed loop. "Vm r" is the mean pairwise correlation of
subthreshold membrane potential with the Ca2+ spikes masked out — the direct
read-out of electrical coupling, and much more sensitive than a spike measure at
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

| topology | Vm r | **CF sync** | d=1 | d=5 | d=20 |
|---|---|---|---|---|---|
| ring (2 partners) | +0.550 | **+0.049** | +0.898 | +0.636 | +0.408 |
| nearest_k (4) | +0.532 | **+0.072** | +0.902 | +0.615 | +0.389 |
| small_world (4, 15% rewired) | +0.541 | **+0.140** | +0.863 | +0.609 | +0.423 |
| all_to_all (39) | +0.761 | **+0.448** | +0.866 | +0.746 | **+0.756** |

Spike synchrony varies **9-fold** across topologies that cost exactly the same
conductance. The last three columns say why: under every local topology the
correlation decays steeply with distance around the olive (0.90 at neighbours,
0.39 at the far side), while under all-to-all it is **flat** (0.87 → 0.76).
Global synchrony is limited by *path length*, not by conductance — a plain ring
of 40 cells is ~10 hops across, and the local junctions are already near
saturation at d=1, so raising `gap_g` has almost nothing left to give there.
Rewiring just 15% of the junctions to random partners cuts the graph diameter
from 10 to 7 and nearly doubles CF synchrony for free. If you want a more
synchronous olive, shorten the paths; do not turn up the conductance.

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

So the two knobs do different jobs: heterogeneity spreads the population's rates,
and gap junctions pull them back into a common rate. That is a real and
physiologically sensible function for olivary coupling — enforcing a shared rhythm
across cells that would otherwise run at their own — and it is invisible in any
measurement that only looks at correlation.

> **Note on an earlier recommendation.** This README previously suggested adding
> heterogeneity to fix the "synchrony = 1.000 at zero coupling" degeneracy. The
> data says that was the wrong remedy: the topographic DCN→IO projection fixed it
> completely on its own (uncoupled Vm r = +0.023 at cv = 0), and heterogeneity
> adds nothing to decorrelation. It earns its place for a different reason, above.

### The IO neuron

The IO fires because of its own ionic currents, implemented in
`sim/io_channels.py`:

| current | gating | role |
|---|---|---|
| `I_CaL` | `k³·l`, low-threshold ("T-type") | Inactivated at rest, de-inactivates on hyperpolarization, then produces the regenerative low-threshold Ca²⁺ spike that carries the CF burst. **This is the current that generates the rhythm.** |
| `I_CaH` | `r²`, high-threshold | Opens only on the Ca²⁺ spike; the Ca²⁺ source that loads `[Ca]i`. |
| `I_KCa` | gated by `[Ca]i`, not voltage | Repolarizes the Ca²⁺ spike and produces the long AHP that paces the oscillation and de-inactivates `I_CaL` for the next cycle. |
| `I_h` | `q`, activated by hyperpolarization | Depolarizing sag out of the AHP trough; sets oscillation frequency. |
| `I_leak` | — | Passive. |
| `I_GABA` | — | DCN inhibition, as a real Cl⁻ conductance (shunting *and* hyperpolarizing) rather than a subtracted scalar. |
| `I_gap` | — | IO-IO electrical coupling, as above. |

Gating kinetics — the voltage and `[Ca]` dependence, i.e. the actual physiology —
are taken from the standard olivary model lineage (Schweighofer, Doya & Kawato
1999, *J Neurophysiol* 82:804; De Gruijl et al. 2012, *PLoS Comput Biol*
8:e1002814) and are hard-coded, not tunable. Maximal conductances are re-fit in
`config.IOChannelParams`, because those papers are 3-compartment models and this
is a single-compartment reduction: kinetics transfer between the two,
conductance *densities* do not.

**What is deliberately not modeled:** the fast Na⁺/delayed-rectifier K⁺ spikes
that ride on top of the Ca²⁺ spike. The downstream circuit consumes only CF
*event times*, and resolving Na⁺ spikes would force a ~40x smaller timestep for
no change to anything measured here. A CF event is therefore detected as the
low-threshold Ca²⁺ spike itself, which is what actually triggers the somatic
burst in the real cell. One visible consequence: with `I_KCa` blocked nothing
repolarizes the Ca²⁺ spike, so the model latches into depolarization block
rather than firing broad spikes.

**IO firing is driven entirely by the cell's own currents.** `i_app = 0` — there
is no applied current, no bias, no tonic depolarizing drive — and since
`noise_sigma = 0` by default there is no jitter either. The cell free-runs at
**3.00 Hz** on `I_CaL`/`I_KCa`/`I_h` alone; block `I_CaL` and it falls silent at
−62.6 mV, which is what pins the T-type current rather than the leak as the
generator.

Membrane noise remains available (`IOChannelParams.noise_sigma`, 2.0 µA/cm² gives
~3.5 mV of Vm jitter) and is what the model was originally fitted with. It is
zero-mean, so it adds no depolarization; what it does is decide *which*
oscillation peaks reach threshold. Turning it off has one measurable consequence
for the loop: a noiseless cell is less excitable at a given conductance, so the
loop compensates by holding the weights higher. Same 60 s closed-loop run:

| `noise_sigma` | IO rate | ISI CV | DCN | equilibrium weight |
|---|---|---|---|---|
| 2.0 | 1.000 Hz | 0.67 | 12.9 Hz | 0.593 |
| 0.0, *same gain* | 0.997 Hz | 0.57 | 9.2 Hz | 0.836 |
| **0.0, re-fitted gain** (default) | **1.010 Hz** | 0.53 | 12.9 Hz | **0.567** |

H1 holds in all three — the loop finds 1 Hz regardless. The middle row is what
happens if you turn the noise off without re-fitting: the loop still equilibrates,
but has to hold the weights near `w_max` to do it. Re-fitting
`dcn_io_gaba_gain` (0.68 → 0.273, see "Assumptions") restores the mid-range
operating point.

One side effect worth noting: CF synchrony under gap-junction coupling rises from
0.005 to 0.075 at the same `gap_g` when the noise is removed. Independent
per-cell membrane noise was actively decorrelating the olive; without it the
junctions have more purchase.

`run_io_characterization.py` regenerates the four single-cell checks this model
was validated against, none of which involve the rest of the circuit:

- **Spontaneous rhythm** with `i_app = 0` — 3.5 Hz, Vm −72 ± 3.5 mV between
  spikes, peaks above 0 mV, `[Ca]i` transients to ~3 µM.
- **Channel ablation** (noiseless, so each contribution is unambiguous):
  control 3.00 Hz; `I_CaL` blocked **0.00 Hz** (silent at −63 mV); `I_KCa`
  blocked 0.00 Hz (depolarization block); `I_h` blocked 2.75 Hz (rhythm survives
  but slows).
- **Rebound burst** — an 800 ms hyperpolarizing step to −81 mV, then release,
  produces a Ca²⁺ spike **7 ms later**, far sooner than the free-running period.
  With `I_CaL` blocked the same step produces nothing. This is the T-type
  de-inactivation signature.
- **Transfer curve** — CF rate falls monotonically from 3.5 Hz uninhibited to
  silence. Monotonicity is what H1's equilibrium argument rests on, and
  `tests/test_io_channels.py` asserts it.

Integration is exponential Euler (`x ← x∞ + (x − x∞)·exp(−dt/τ)`,
unconditionally stable) on a 0.1 ms sub-timestep inside the 1 ms outer loop.
Halving that sub-timestep three times over (to 0.0125 ms) changes the mean
inter-spike interval by 0.33%, and the last halving by 0.04% — converged;
`tests/test_io_channels.py` asserts under 2% against a 4x finer step.

## Assumptions not specified by the paper

The white paper specifies population ratios and plasticity defaults (both
implemented as given — see `config.py`), but *not* a membrane model or coupling
gains. Those were fitted, and `experiments/run_calibration.py` is how — it
measures each stage in isolation, driven by a surrogate of its input, and
reports the gain that hits the target, so the constants in `config.py` can be
re-derived after any change rather than being magic numbers:

| gain | fitted so that | value |
|---|---|---|
| `pf_pkj_gain` | PKJ runs 50 Hz at weight 0 and 70 Hz at weight 1 | 0.000137 |
| `pkj_dcn_gain` | DCN sits at 15 Hz when its 12 PKJ fire at 60 Hz | (re-fitted) |
| `dcn_io_gaba_gain` | IO fires 1 Hz at DCN = 15 Hz | 0.1158 |

The PKJ and DCN spans are the ones the superseded rate model produced, so a
weight still means what it used to and the loop still has room to move in both
directions from `w_init = 0.5`.

`dcn_io_gaba_gain` depends on `IOChannelParams.noise_sigma`: it was 0.68 when the
olive carried 2.0 µA/cm² of membrane noise, and is 0.273 with noise off, because
a noiseless cell needs less inhibition to sit at 1 Hz. `run_calibration.py` reads
the live config, so re-running it after changing the noise re-derives the right
value rather than going stale.

`dcn_io_gaba_gain` has been re-fitted twice, and the sequence is informative:
0.68 with membrane noise and 2 converging DCN, 0.273 once the noise went to zero,
0.1158 once the paper's ratios raised the convergence to 6. Each step needs *less*
inhibition per synapse, for the same underlying reason — how readily the cell
escapes inhibition depends on how grainy that inhibition is, and both changes made
it smoother. `run_calibration.py` reads the live config and mirrors the real
convergence, so re-running it after any topology or noise change re-derives the
right value rather than going stale.

Each stage's curve passes through its target: PKJ 49.6 Hz at weight 0 and 70.1 Hz
at weight 1, DCN 15.2 Hz at PKJ 60 Hz, IO 0.99 Hz at DCN 15 Hz. The IO stage is
the one worth looking at — 5/10/15/20/25/30 Hz of DCN gives
2.52/1.86/0.99/0.32/0.10/0.01 Hz of CF output, smooth and monotone across the
range the loop visits. It is steeper than at two converging DCN; that is the
graininess cost quantified below.

Two findings worth knowing if you retune any of this:

**PKJ/DCN noise is load-bearing, not decoration.** A noiseless integrate-and-fire
cell has a hard rheobase, so its rate collapses from tens of Hz to silence over a
very narrow band of inhibitory conductance. The loop's negative feedback needs
DCN's rate to vary *smoothly* with PKJ drive, and the OU membrane noise
(`LIFParams.noise_sigma_mv`, 3 mV) is what linearizes the f-I curve around
threshold to give it that. `tests/test_neurons.py` pins this. Note this is a
*separate* knob from the olive's `IOChannelParams.noise_sigma`, which is off by
default: the IO has its own ionic machinery to generate firing, the
integrate-and-fire cells do not.

**The loop's feedback limb exists only because inhibition arrives as discrete
events.** With the default noiseless olive, holding the DCN conductance
*constant* gives an all-or-nothing response: 3.00 Hz at `g = 0`, and silence at
every `g >= 0.10` mS/cm². There is no graded range at all — steady inhibition
either leaves the rhythm untouched or abolishes it.

But only a handful of nuclear cells feed each IO, at ~15 Hz, into a 50 ms
conductance. What the cell actually sees therefore swings around its mean, and it
fires in the troughs. Delivered that way the same averages produce a smooth
monotone curve — and how smooth depends on how many cells converge:

| mean g | 0.00 | 0.14 | 0.28 | 0.42 | 0.55 |
|---|---|---|---|---|---|
| CF rate (Hz), 2 DCN converging | 2.96 | 2.34 | 1.66 | 0.96 | 0.53 |
| CF rate (Hz), **6 DCN** (the shipped ratio) | 2.96 | 1.98 | 0.71 | 0.13 | 0.00 |

That is the entire negative-feedback limb, and it is a property of the synapse's
*granularity*, not of the mean it delivers. `results/io_transfer_curve.png` plots
both curves on the same axis. (Adding membrane noise back makes the held-constant
curve gradeable too — 1.35 Hz at 0.40, dead by 0.60 — so noise and synaptic
granularity are two routes to the same requirement; the model now relies on the
second.)

**How much graininess does the loop need?** Restoring the paper's ratios raised
the DCN converging on each olive from 2 to 6, which delivers the same mean
conductance in three times as many, three times smaller, events. Measured at
matched mean conductance, noiseless, that systematically narrows the limb:

| DCN per IO | CV(g) at g≈0.42 | CF rate at g≈0.42 | g giving 1 Hz | silent beyond |
|---|---|---|---|---|
| held constant | 0.00 | 0 Hz | ~0.05 | **0.10** |
| **6** (default) | 0.10 | 0.13 Hz | **0.23** | ~0.55 |
| 2 | 0.17 | 0.96 Hz | 0.42 | ~0.90 |
| 1 | 0.23 | 1.57 Hz | 0.55 | > 0.90 |

So the loop still works at six converging nuclear cells — the curve is smooth and
monotone (2.96 → 1.98 → 0.71 → 0.13 Hz) and the usable range is still about five
times wider than a held conductance — but it is roughly half as wide as at two,
and the operating point moves to a lower mean conductance. This is the cost of
the spec's population ordering, and it is a real prediction: the olive's feedback
limb needs its inhibition to arrive in *few enough* events to stay grainy, so
there is an upper bound on nucleo-olivary convergence beyond which the loop
cannot regulate itself without some other noise source.

That is also why `tau_dcn_io_ms` is deliberately kept short. Lengthening it
averages the few DCN inputs into a steadier conductance, which pushes the cell
back toward the held-constant limit. Measured with membrane noise at a fixed mean
conductance of ~0.445, CF rate falls as the fluctuation is smoothed away:

| DCN→IO τ | CV of g | CF rate at mean g ≈ 0.445 |
|---|---|---|
| constant | 0.00 | 0.95 Hz |
| 150 ms | 0.08 | 1.39 Hz |
| 100 ms | 0.10 | 1.57 Hz |
| 50 ms (default) | 0.15 | 2.05 Hz |

`config.SimConfig.burn_in_s` (default 8 s) runs the loop to its dynamic
equilibrium *before* plasticity or logging start. Without it every cell starts
from rest and IO transiently overshoots toward its fully-unopposed rate before
DCN/PKJ activity ramps up to inhibit it — inflicting a burst of spurious early
LTD that permanently crashes the weights. That is a startup artifact, not part
of the modeled dynamics, hence the burn-in.

## Module layout

```
config.py                    # all parameters + defaults, paper-specified vs. fitted
sim/
  poisson_input.py           # PF Poisson spike generation
  neurons.py                 # PKJ / DCN conductance-based integrate-and-fire
  io_channels.py             # conductance-based IO: I_CaL / I_CaH / I_KCa / I_h
  io_coupling.py             # IO-IO gap junctions: topologies + synchrony metric
  connectivity.py            # CbmSim's microzone wiring, ported
  plasticity.py              # coincidence-detection LTD/LTP/null rule
  network_graph.py           # the wiring as a positioned, typed graph + adjacency matrix
  network_viz.py             # 3D snapshot and connectivity-matrix figures
  simulate.py                # main time-stepped loop and the recording layer
  analysis.py                # spike-based metrics + raster/trace/weight plots
experiments/
  run_network.py             # the microzone, fully recorded
  run_network_snapshot.py    # the wiring in 3D, no simulation
  run_gap_sweep.py           # coupling strength vs. synchrony and vs. the loop's equilibrium
  run_baseline.py            # H1 on a single loop
  run_calibration.py         # stage-by-stage gain fitting
  run_io_characterization.py # IO channel physiology, single cell
  run_window_sweep.py        # H2, single-condition sweep
  run_three_window.py        # H3, single run
  sweep.py                   # parallel condition sweeps, one JSON sidecar per run
  analyze_sweeps.py          # aggregate sidecars into replicated tables
  figures.py                 # the four analysis figures, incl. raw trajectories
tests/
  test_plasticity.py         # window logic, both window schemes
  test_neurons.py            # f-I inversion, membrane, CF pause
  test_io_channels.py        # gating, integration, cell behaviour, gap junctions, wiring
  test_network_graph.py      # the drawn graph matches the simulated wiring
  test_perturbations.py      # heterogeneity and DCN->IO ablation do what they claim
```

## Non-goals / explicit simplifications

- No mossy fibers, granule cells, or Golgi cells — PF spikes are generated
  directly as Poisson processes.
- No spatial or topographic organization beyond CbmSim's connection tables. The 3D
  positions in `network_graph.py` are for drawing; nothing in the dynamics reads
  them, and no connection depends on distance.
- The IO is single-compartment: no separate soma/dendrite/axon, and no fast
  Na⁺/K⁺ spikes (see "The IO neuron"). Gap junctions are somatic, whereas real
  olivary coupling is dendritic and inside the glomerulus.
- CF bursts are modeled as single timestamped events, not variable-spike-count
  bursts — the Ca²⁺ spike is resolved, the Na⁺ spikes riding on it are not.
- PKJ and DCN have no distinct cell classes, no spike-frequency adaptation, and
  no synaptic plasticity of their own; only PF→PKJ is plastic.
# io_equilibrium_sim
