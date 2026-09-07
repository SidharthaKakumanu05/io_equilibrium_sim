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
"""
import numpy as np


class Plasticity:
    def __init__(self, n_pkj, n_pf, dt_ms, ltd_window_ms, delta_plus, delta_minus,
                 null_window_ms=0.0, w_min=0.0, w_max=1.0, cf_source_of_pkj=None):
        self.dt_ms = dt_ms                                          # timestep, ms
        self.ltd_steps = max(1, round(ltd_window_ms / dt_ms))       # LTD window length, in steps
        self.null_steps = max(0, round(null_window_ms / dt_ms))     # null window length, in steps (0 = disabled)
        self.total_steps = self.ltd_steps + self.null_steps         # full resolution delay, in steps
        self.delta_plus = delta_plus                                 # LTP increment
        self.delta_minus = delta_minus                               # LTD decrement
        self.w_min = w_min                                           # weight clip lower bound
        self.w_max = w_max                                           # weight clip upper bound

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

        had_ltd = (count_at_boundary > count_at_spike)[:, None]       # a CF landed within this cell's LTD window
        had_null = ~had_ltd & (count_now > count_at_boundary)[:, None]  # a CF landed within the null window instead

        ltd_mask = spikes_then & had_ltd                              # broadcast the per-cell verdict over its PF inputs
        ltp_mask = spikes_then & ~had_ltd & ~had_null                 # neither window fired -> potentiate

        weights[ltd_mask] -= self.delta_minus
        weights[ltp_mask] += self.delta_plus
        self.n_ltd_events += int(ltd_mask.sum())
        self.n_ltp_events += int(ltp_mask.sum())
        # synapses that spiked and landed in the null window are deliberately left untouched

        np.clip(weights, self.w_min, self.w_max, out=weights)        # enforce bounds after every update
