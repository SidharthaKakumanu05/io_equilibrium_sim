"""Coincidence-detection LTD/LTP(/null) plasticity for PF -> PKJ synapses.

The rule, which is the white paper's: a PF spike is DEPRESSED if a
climbing-fiber event follows it within `ltd_window_ms`, and POTENTIATED if none
does. The three-window variant (H3) inserts a `null_window_ms` after the LTD
window in which a CF event produces no change at all, so a near-miss neither
depresses nor potentiates.

Every Purkinje cell in the microzone shares one weight matrix and one instance
of this class. `cf_source_of_pkj` gives the single climbing fiber that contacts
each cell (CbmSim's `connectIOtoPC` guarantees exactly one), so each row
resolves against its own olivary neuron rather than against a population
average.

The implementation is retrospective: a PF spike is not resolved when it
arrives, but `ltd_steps + null_steps` later, once it is known whether a CF event
landed in its LTD window, its null window, or neither. Rather than queueing an
event per spike -- 160,000 synapses at 20 Hz means scheduling and retiring about
3.2 million events per simulated second -- it keeps a circular buffer of past
PF-spike arrays plus a circular buffer of *cumulative* CF counts per source.
Whether a CF fell in a window is then a subtraction of two counters, O(1) per
timestep regardless of how many synapses spiked, and exactly equivalent to
scheduling each spike individually: the count difference IS the number of CF
events in that interval.

The LTD window is delays 1..ltd_steps inclusive, and both ends matter:

  * A CF at exactly `ltd_window_ms` after the spike still depresses; the very
    next timestep potentiates instead.
  * A CF in the SAME timestep as the spike (delay 0) potentiates, not depresses.
    `step` folds this step's CF events into the cumulative counters before
    recording them, so a simultaneous CF reads as already-past rather than as
    falling inside the window. That is the right sense for a rule about a CF
    *following* a PF spike, and tests/test_plasticity.py pins it.

tests/test_plasticity.py pins every boundary in both window schemes.

`weight_dependence` controls the step SIZE, not the rule. At its default 0.0 the
rule is additive -- every LTD step is exactly delta_minus and every LTP step
exactly delta_plus, whatever the synapse's present weight, with the [w_min,
w_max] range enforced by clipping afterwards. That form gives an individual
synapse no fixed point: within one Purkinje cell every synapse sees the same CF
verdict, so the feedback loop's restoring force is common to all 500 of them and
cancels out of their differences, leaving the differences a driftless random walk
that fills [w_min, w_max]. At weight_dependence = 1.0 the LTD step is scaled by
(w - w_min) and the LTP step by (w_max - w), so each synapse is pulled toward its
own balance point and the within-cell spread becomes stationary. Values between
the two interpolate.
"""
import numpy as np
from .native import discrete_update


class Plasticity:
    def __init__(self, n_pkj, n_pf, dt_ms, ltd_window_ms, delta_plus, delta_minus,
                 null_window_ms=0.0, w_min=0.0, w_max=1.0, cf_source_of_pkj=None,
                 weight_dependence=0.0, mode="additive", low_weight=0.25,
                 high_weight=0.55, p_ltd=0.9, p_ltp=0.1, transition_seed=None):
        if mode not in ("additive", "binary", "abbott_cascade", "mauk_cascade"):
            raise ValueError("Unknown plasticity mode: " + str(mode))
        self.mode = mode
        self.shape = (n_pkj, n_pf)
        self.low_weight, self.high_weight = float(low_weight), float(high_weight)
        self.p_ltd, self.p_ltp = float(p_ltd), float(p_ltp)
        if mode != "additive":
            if weight_dependence != 0:
                raise ValueError("Discrete plasticity cannot use weight_dependence")
            if not (np.isfinite([low_weight, high_weight, p_ltd, p_ltp]).all()
                    and w_min <= low_weight < high_weight <= w_max
                    and 0 <= p_ltd <= 1 and 0 <= p_ltp <= 1):
                raise ValueError("Invalid discrete weight levels or transition probabilities")
            self.states = np.full(self.shape, 3, dtype=np.uint8)
            self.transition_rng = np.random.default_rng(transition_seed)
            self.switch_counts = np.zeros(self.shape, dtype=np.int64)
            self.n_state_transitions = 0
            self.n_weight_switches = 0
            self.n_ltd_switches = 0
            self.n_ltp_switches = 0
            self.n_null_events = 0
        self.dt_ms = dt_ms                                          # timestep, ms
        self.ltd_steps = max(1, round(ltd_window_ms / dt_ms))       # LTD window length, in steps
        self.null_steps = max(0, round(null_window_ms / dt_ms))     # null window length, in steps (0 = disabled)
        self.total_steps = self.ltd_steps + self.null_steps         # full resolution delay, in steps
        self.delta_plus = delta_plus                                 # LTP increment
        self.delta_minus = delta_minus                               # LTD decrement
        self.w_min = w_min                                           # weight clip lower bound
        self.w_max = w_max                                           # weight clip upper bound
        # How strongly each step's SIZE depends on the synapse's present weight. 0.0 reproduces the
        # additive rule exactly (every step is +-delta regardless of w, bounded only by the clip);
        # 1.0 makes LTD proportional to (w - w_min) and LTP to (w_max - w), which is the soft-bound
        # form. The WHEN of the rule -- LTD if a CF follows inside the window, LTP otherwise -- is
        # identical either way; only the magnitude changes. See the class docstring.
        self.weight_dependence = float(weight_dependence)
        self._w_span = float(w_max - w_min)

        # Which CF each Purkinje cell listens to. Default: one shared source, which is what the
        # unit tests drive; the assembled network always passes the real cf_of_pkj map.
        if cf_source_of_pkj is None:
            self.cf_source_of_pkj = np.zeros(n_pkj, dtype=int)
        else:
            self.cf_source_of_pkj = np.asarray(cf_source_of_pkj, dtype=int)
            if self.cf_source_of_pkj.shape != (n_pkj,):
                raise ValueError(f"cf_source_of_pkj must have shape ({n_pkj},)")
        self.n_cf_sources = int(self.cf_source_of_pkj.max()) + 1

        self._buf_len = self.total_steps + 1                        # ring buffer depth: total_steps of history + "now"
        self._pf_ring = np.zeros((self._buf_len, n_pkj, n_pf), dtype=bool)      # circular buffer of past PF-spike arrays
        self._cf_ring = np.zeros((self._buf_len, self.n_cf_sources), dtype=np.int64)  # cumulative CF counts, per source
        self._write_ptr = -1                                         # index of the most recently written slot
        self._filled = 0                                             # how many slots have been written so far
        self._cf_cumulative = np.zeros(self.n_cf_sources, dtype=np.int64)  # running total of CF events ever seen

        self.n_ltd_events = 0                                        # diagnostic: total synapse-LTD events applied
        self.n_ltp_events = 0                                        # diagnostic: total synapse-LTP events applied

    def step(self, pf_spikes, cf_event, weights):
        """`cf_event` is a bool per CF source (a bare bool is accepted when
        there is only one). `weights` is modified in place."""
        cf = np.atleast_1d(np.asarray(cf_event, dtype=bool))
        self._cf_cumulative += cf                                   # count this step's CF events before recording them

        self._write_ptr = (self._write_ptr + 1) % self._buf_len      # advance the circular write pointer
        self._pf_ring[self._write_ptr] = pf_spikes                   # record this step's PF spikes
        self._cf_ring[self._write_ptr] = self._cf_cumulative         # record the running CF counts as of this step
        self._filled = min(self._filled + 1, self._buf_len)          # track how much history has accumulated

        if self._filled < self._buf_len:
            return                                                   # buffer not yet full: nothing old enough to resolve

        oldest_ptr = (self._write_ptr + 1) % self._buf_len              # the slot exactly total_steps steps ago
        boundary_ptr = (self._write_ptr - self.null_steps) % self._buf_len  # the slot exactly null_steps steps ago (LTD/null boundary)

        spikes_then = self._pf_ring[oldest_ptr]                       # (n_pkj, n_pf): which synapses spiked at that instant
        src = self.cf_source_of_pkj
        count_at_spike = self._cf_ring[oldest_ptr][src]               # per PKJ: CF count at spike time
        count_at_boundary = self._cf_ring[boundary_ptr][src]          # per PKJ: CF count at the LTD/null boundary
        count_now = self._cf_ring[self._write_ptr][src]               # per PKJ: CF count right now

        had_ltd = count_at_boundary > count_at_spike                 # a CF landed within this cell's LTD window
        had_null = ~had_ltd & (count_now > count_at_boundary)         # a CF landed within the null window instead

        # PF activity is sparse (2% of inputs per default timestep). Gather the
        # fired coordinates once, then apply each cell's verdict to that list,
        # rather than scanning two full-size boolean masks for reads and writes.
        # np.flatnonzero preserves row-major event order; tuple indexing also writes
        # through correctly when a caller supplies a non-contiguous weight view.
        rows, cols = np.divmod(np.flatnonzero(spikes_then), spikes_then.shape[1])
        ltd = had_ltd[rows]
        ltp = ~had_ltd[rows] & ~had_null[rows]
        ltd_idx = (rows[ltd], cols[ltd])
        ltp_idx = (rows[ltp], cols[ltp])

        if self.mode == "additive":
            self._apply_additive(weights, ltd_idx, ltp_idx)
        else:
            indices = rows * self.shape[1] + cols
            directions = ltp.astype(np.int8) - ltd.astype(np.int8)
            self._apply_discrete(weights, indices, directions)

    def _apply_additive(self, weights, ltd_idx, ltp_idx):
        wd = self.weight_dependence
        if wd == 0.0:
            weights[ltd_idx] -= self.delta_minus
            weights[ltp_idx] += self.delta_plus
        elif True:
            # Identical arithmetic to the branch below, evaluated only at the
            # synapses that actually move. (w - w_min)/span is elementwise, so
            # gathering first and dividing after is bit-for-bit the same value,
            # and it avoids building two full-matrix temporaries every timestep.
            w_ltd = weights[ltd_idx]
            w_ltp = weights[ltp_idx]
            weights[ltd_idx] = w_ltd - self.delta_minus * (
                (1.0 - wd) + wd * ((w_ltd - self.w_min) / self._w_span))
            weights[ltp_idx] = w_ltp + self.delta_plus * (
                (1.0 - wd) + wd * ((self.w_max - w_ltp) / self._w_span))
        else:
            # Scale each step by how far the synapse is from the bound it is moving toward:
            # (1 - wd) of the step stays additive, wd of it is proportional to the remaining
            # distance. At wd = 1 a synapse at w_min cannot be depressed further and one at
            # w_max cannot be potentiated further, so the bounds are approached asymptotically
            # rather than by clipping -- which is what gives an individual synapse a fixed point.
            frac_down = (weights - self.w_min) / self._w_span
            frac_up = (self.w_max - weights) / self._w_span
            weights[ltd_idx] -= self.delta_minus * ((1.0 - wd) + wd * frac_down[ltd_idx])
            weights[ltp_idx] += self.delta_plus * ((1.0 - wd) + wd * frac_up[ltp_idx])
        self.n_ltd_events += len(ltd_idx[0])
        self.n_ltp_events += len(ltp_idx[0])
        # synapses that spiked and landed in the null window are deliberately left untouched

        # Only the synapses just written can be out of bounds: every other weight was
        # already inside [w_min, w_max] when it was last modified, and clip is the
        # identity on an in-range float. Clipping just the touched indices is therefore
        # bit-for-bit identical to clipping the whole matrix, and avoids streaming
        # 2.4 MB through memory on every one of the run's millions of timesteps.
        # NB: weights[idx] is a fancy-index COPY, so the clipped values must be
        # assigned back; an out= into weights[idx] would write to a temporary.
        for idx in (ltd_idx, ltp_idx):
            if len(idx[0]):
                weights[idx] = np.clip(weights[idx], self.w_min, self.w_max)


    def initialize_weights(self, weights, seed=None):
        """Initialize matched random low/high weights and shallow states 3/4.

        Uses a separate initialization stream, never the transition stream. Each
        synapse independently has probability 1/2 of starting high. Intended once,
        before learning; no independently shuffled weight/state assignments.
        """
        if weights.shape != self.shape:
            raise ValueError("Weight shape mismatch")
        if not (self.w_min <= self.low_weight < self.high_weight <= self.w_max):
            raise ValueError("Two-level initialization requires levels within weight bounds")
        if self.mode != "additive" and (self.n_ltd_events or self.n_ltp_events or self.n_null_events):
            raise ValueError("Cannot reinitialize after learning events")
        high = np.random.default_rng(seed).integers(0, 2, size=self.shape, dtype=np.uint8)
        weights[:] = np.where(high, self.high_weight, self.low_weight)
        if self.mode != "additive":
            self.states[:] = high + 3

    def validate_state(self, weights):
        """Check the fixed-level model invariant without consuming random numbers."""
        if weights.shape != self.shape or not np.isfinite(weights).all():
            raise ValueError("Invalid weight shape or finite values")
        if self.mode == "additive":
            return
        if (self.states.shape != self.shape or self.states.dtype != np.uint8
                or np.any(self.states > 7)
                or (self.mode == "binary" and np.any((self.states != 3) & (self.states != 4)))):
            raise ValueError("Invalid discrete states")
        expected = np.where(self.states < 4, self.low_weight, self.high_weight)
        if not np.array_equal(weights, expected):
            raise ValueError("Discrete states and expressed weights disagree")
        if (np.any(self.switch_counts < 0) or self.switch_counts.shape != self.shape
                or int(self.switch_counts.sum()) != self.n_weight_switches
                or self.n_ltd_switches + self.n_ltp_switches != self.n_weight_switches
                or not 0 <= self.n_weight_switches <= self.n_state_transitions <= self.n_ltd_events + self.n_ltp_events
                or not 0 <= self.n_ltd_switches <= self.n_ltd_events
                or not 0 <= self.n_ltp_switches <= self.n_ltp_events or self.n_null_events < 0):
            raise ValueError("Invalid discrete transition counters")

    def apply_resolved(self, weights, flat_indices, directions, uniforms=None):
        """Apply one simultaneous set of already-resolved PF learning events.

        Unique C-order flat synapse indices; directions -1=LTD, 0=null, +1=LTP.
        Optional uniforms have one entry per index in [0,1] (1 always fails).
        Supplied uniforms bypass the transition RNG. Otherwise one draw per
        non-null event, including endpoint events, in supplied index order.
        Call successive batches chronologically; repeated indices are rejected.
        This primitive does not advance biological time or the eligibility ring.
        """
        idx = np.asarray(flat_indices)
        direction = np.asarray(directions)
        if (idx.ndim != 1 or (len(idx) and idx.dtype.kind not in "iu")
                or direction.shape != idx.shape or np.any(~np.isin(direction, [-1, 0, 1]))
                or np.any(idx < 0) or np.any(idx >= np.prod(self.shape))
                or len(np.unique(idx)) != len(idx)):
            raise ValueError("Require unique valid flat indices and aligned -1/0/+1 directions")
        idx = idx.astype(np.intp, copy=False)
        if uniforms is not None:
            uniforms = np.asarray(uniforms, dtype=float)
            if uniforms.shape != idx.shape or not np.isfinite(uniforms).all() or np.any((uniforms < 0) | (uniforms > 1)):
                raise ValueError("Require aligned finite uniforms in [0,1]")
        self.validate_state(weights)
        if self.mode == "additive":
            rows, cols = np.divmod(idx, self.shape[1])
            self._apply_additive(weights, (rows[direction == -1], cols[direction == -1]),
                                 (rows[direction == 1], cols[direction == 1]))
        else:
            self._apply_discrete(weights, idx, direction, uniforms)

    def _apply_discrete(self, weights, indices, directions, uniforms=None):
        # Literal CbmSim d921c856 kernels.cu 498-720; double precision here,
        # preserving this simulator's conventions and strict random < threshold.
        self.n_ltd_events += int(np.count_nonzero(directions == -1))
        self.n_ltp_events += int(np.count_nonzero(directions == 1))
        self.n_null_events += int(np.count_nonzero(directions == 0))
        active = directions != 0
        ids = indices[active]
        if not len(ids):
            return
        direction = directions[active]
        u = self.transition_rng.random(len(ids)) if uniforms is None else uniforms[active]
        if discrete_update is not None and weights.dtype == np.float64:
            # Only arithmetic/indexing is fused; eligibility, draw count/order,
            # null handling and all state ownership remain on the reference path.
            changes, switches, ltd_switches, ltp_switches = discrete_update(
                weights, self.states, self.switch_counts, ids,
                direction.astype(np.int8, copy=False), u,
                {"binary": 0, "abbott_cascade": 1, "mauk_cascade": 2}[self.mode],
                self.p_ltd, self.p_ltp, self.low_weight, self.high_weight)
            self.n_state_transitions += changes
            self.n_weight_switches += switches
            self.n_ltd_switches += ltd_switches
            self.n_ltp_switches += ltp_switches
            return
        rows, cols = np.divmod(ids, self.shape[1])
        before = self.states[rows, cols]
        ltd = direction == -1
        if self.mode == "binary":
            target = np.where(ltd, 3, 4).astype(np.uint8)
            probability = np.where(ltd, self.p_ltd, self.p_ltp)
        else:
            down = np.array([0, 0, 1, 2, 3, 3, 3, 3] if self.mode == "abbott_cascade"
                            else [0, 0, 1, 2, 3, 4, 5, 6], dtype=np.uint8)
            up = np.array([4, 4, 4, 4, 5, 6, 7, 7] if self.mode == "abbott_cascade"
                          else [1, 2, 3, 4, 5, 6, 7, 7], dtype=np.uint8)
            pd = np.array([0, .125, .25, .5, 1, .5, .25, .125]) * self.p_ltd
            pu = np.array([.125, .25, .5, 1, .5, .25, .125, 0]) * self.p_ltp
            target = np.where(ltd, down[before], up[before])
            probability = np.where(ltd, pd[before], pu[before])
        after = np.where(u < probability, target, before)
        changed = after != before
        switched = (after < 4) != (before < 4)
        self.states[rows, cols] = after
        # Tuple indices preserve writes even for non-contiguous weight views.
        weights[rows[switched], cols[switched]] = np.where(after[switched] < 4, self.low_weight, self.high_weight)
        self.switch_counts[rows[switched], cols[switched]] += 1
        self.n_state_transitions += int(changed.sum())
        self.n_weight_switches += int(switched.sum())
        self.n_ltd_switches += int(np.count_nonzero(switched & ltd))
        self.n_ltp_switches += int(np.count_nonzero(switched & ~ltd))

    def cascade_counters(self):
        """Null events, state transitions, weight switches, LTD switches, LTP switches."""
        return np.array([self.n_null_events, self.n_state_transitions, self.n_weight_switches,
                         self.n_ltd_switches, self.n_ltp_switches], dtype=np.int64)
