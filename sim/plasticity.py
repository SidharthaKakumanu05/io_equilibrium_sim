"""The learning rule at the PF -> PKJ synapses.

THE RULE. Take any PF spike onto a Purkinje cell. Look at whether that Purkinje
cell's climbing fibre fires during the next ltd_window_ms (100 ms):

    CF fires within the window   -> that synapse is weakened by delta_minus  (LTD)
    CF does not fire             -> that synapse is strengthened by delta_plus (LTP)

Optional three-window version: if null_window_ms > 0, a CF that arrives just
after the LTD window, within the null window, causes no change at all.

WHY THE LOOP SETTLES AT ~1 Hz. If CF events come every T ms, a fraction
window/T of PF spikes get LTD and the rest get LTP. Weights stop changing when
    (window / T) x delta_minus = (1 - window / T) x delta_plus,
i.e. T = window x (1 + delta_minus / delta_plus) = 100 x (1 + 9) = 1000 ms.

HOW IT IS COMPUTED. A PF spike's fate is only known once its window has passed.
So the rule works with a delay: each step it looks at the PF spikes from exactly
(ltd window + null window) steps ago, and asks how many CF events have happened
since then. To answer that cheaply it keeps two ring buffers (fixed-length
histories that overwrite their oldest entry):

    _pf_ring   the PF spike array from each recent step
    _cf_ring   a running total of CF events per climbing fibre, as of each step

"Did a CF arrive between then and now?" is then just
"total now > total then". Each step therefore costs the same no matter how
many synapses spiked.

Timing edge cases: a CF exactly ltd_window_ms after the PF spike counts as LTD.
A CF in the SAME step as the PF spike does NOT. The rule is about a CF
following a PF spike.
"""
import numpy as np


class Plasticity:
    """The learning rule for all PF -> PKJ synapses. Call step() once per time
    step; it changes the weight matrix in place."""

    def __init__(self, n_pkj, n_pf, dt_ms, ltd_window_ms, delta_plus, delta_minus,
                 null_window_ms=0.0, w_min=0.0, w_max=1.0, cf_source_of_pkj=None):
        self.dt_ms = dt_ms
        self.ltd_steps = max(1, round(ltd_window_ms / dt_ms))       # LTD window, in steps (100)
        self.null_steps = max(0, round(null_window_ms / dt_ms))     # null window, in steps (0 = off)
        self.total_steps = self.ltd_steps + self.null_steps         # how long to wait before judging a PF spike
        self.delta_plus = delta_plus                                 # LTP step
        self.delta_minus = delta_minus                               # LTD step
        self.w_min = w_min                                           # weights are clipped to [w_min, w_max]
        self.w_max = w_max

        # Which climbing fibre each Purkinje cell learns from. The simulation always passes this in;
        # without it, every cell shares one climbing fibre.
        if cf_source_of_pkj is None:
            self.cf_source_of_pkj = np.zeros(n_pkj, dtype=int)
        else:
            self.cf_source_of_pkj = np.asarray(cf_source_of_pkj, dtype=int)
            if self.cf_source_of_pkj.shape != (n_pkj,):
                raise ValueError(f"cf_source_of_pkj must have shape ({n_pkj},)")
        self.n_cf_sources = int(self.cf_source_of_pkj.max()) + 1    # number of climbing fibres

        # --- the ring buffers (see the module docstring) ---
        self._buf_len = self.total_steps + 1                        # history length: total_steps back, plus now
        self._pf_ring = np.zeros((self._buf_len, n_pkj, n_pf), dtype=bool)      # past PF spike arrays
        self._cf_ring = np.zeros((self._buf_len, self.n_cf_sources), dtype=np.int64)  # past running CF totals
        self._write_ptr = -1                                         # slot written most recently
        self._filled = 0                                             # slots written so far (until the buffer is full)
        self._cf_cumulative = np.zeros(self.n_cf_sources, dtype=np.int64)  # running CF total per climbing fibre

        self.n_ltd_events = 0                                        # counters for inspection only
        self.n_ltp_events = 0

    def step(self, pf_spikes, cf_event, weights):
        """Record this step, then apply LTD/LTP to the PF spikes that are
        exactly total_steps old.

        pf_spikes: (n_pkj, n_pf) True/False, this step's PF spikes
        cf_event:  True/False per climbing fibre, this step's CF events
        weights:   (n_pkj, n_pf), changed in place
        """
        cf = np.atleast_1d(np.asarray(cf_event, dtype=bool))
        # Add this step's CF events to the totals BEFORE storing them. A CF in the same step as a
        # PF spike is then already included in "the total at spike time", so it doesn't count
        # as following the spike.
        self._cf_cumulative += cf

        # --- 1. store this step in the ring buffers ---
        self._write_ptr = (self._write_ptr + 1) % self._buf_len      # move to the next slot (wrapping around)
        self._pf_ring[self._write_ptr] = pf_spikes
        self._cf_ring[self._write_ptr] = self._cf_cumulative
        self._filled = min(self._filled + 1, self._buf_len)

        if self._filled < self._buf_len:
            return                                                   # first total_steps steps: nothing old enough yet

        # --- 2. find the three moments we need ---
        oldest_ptr = (self._write_ptr + 1) % self._buf_len              # when the PF spikes being judged happened
        boundary_ptr = (self._write_ptr - self.null_steps) % self._buf_len  # end of their LTD window (= now if no null window)

        spikes_then = self._pf_ring[oldest_ptr]                       # (n_pkj, n_pf): the PF spikes being judged
        src = self.cf_source_of_pkj
        count_at_spike = self._cf_ring[oldest_ptr][src]               # each PKJ's CF total when the PF spiked
        count_at_boundary = self._cf_ring[boundary_ptr][src]          # ...at the end of the LTD window
        count_now = self._cf_ring[self._write_ptr][src]               # ...now

        # --- 3. one verdict per Purkinje cell, applied to all its PFs that spiked then ---
        had_ltd = (count_at_boundary > count_at_spike)[:, None]       # CF arrived inside the LTD window
        had_null = ~had_ltd & (count_now > count_at_boundary)[:, None]  # CF arrived in the null window instead

        ltd_mask = spikes_then & had_ltd                              # synapses to weaken
        ltp_mask = spikes_then & ~had_ltd & ~had_null                 # synapses to strengthen (no CF at all)

        weights[ltd_mask] -= self.delta_minus
        weights[ltp_mask] += self.delta_plus
        self.n_ltd_events += int(ltd_mask.sum())
        self.n_ltp_events += int(ltp_mask.sum())
        # (synapses whose CF fell in the null window are left unchanged)

        np.clip(weights, self.w_min, self.w_max, out=weights)        # keep weights within [w_min, w_max]
