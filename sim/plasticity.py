"""Coincidence-detection LTD/LTP(/null) plasticity for PF -> PKJ synapses.

Every Purkinje cell in the network shares one weight matrix and one instance of
this class; `cf_source_of_pkj` says which climbing fiber each row listens to, so
a network of several closed-loop groups resolves each group's synapses against
that group's own IO neuron.

The rule is retrospective: a PF spike is not resolved when it arrives, but
`ltd_steps + null_steps` later, once it is known whether a CF event landed in
its LTD window, its null window, or neither. Rather than queueing an event per
spike, the implementation keeps a circular buffer of past PF-spike arrays and a
circular buffer of *cumulative* CF counts per source. Whether a CF fell in a
window is then a subtraction of two counters -- O(1) per timestep regardless of
how many synapses spiked -- and is exactly equivalent to scheduling each spike
individually, because the count difference is precisely the number of CF events
in that interval.

Window boundaries are inclusive on the near edge: a CF at exactly
`ltd_window_ms` after a spike still counts as LTD, and the very next timestep
does not. tests/test_plasticity.py pins that boundary in both window schemes.
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

        # Which CF each Purkinje cell listens to. Default: one shared source, i.e. a single group.
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
