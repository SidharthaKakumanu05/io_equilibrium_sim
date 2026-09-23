"""Streaming spike recording, for runs too long to hold in memory.

`sim/simulate.py`'s SpikeRecorder keeps every spike in two Python lists and
splits them at the end. That is the right shape for the 60-300 s runs the rest
of the project makes, and it stops working somewhere well short of an hour: at
the shipped scale 320 Purkinje cells at 60 Hz emit 19,200 spikes per simulated
second, so a 5-hour run is ~346 million PKJ spikes. As Python floats and ints in
a list that is tens of gigabytes, and none of it is needed at once.

This module writes the same information straight to disk in a form that can be
read back in slices. Two flat binary files per population:

    <name>.t.i32   step index of each spike, int32, ascending
    <name>.c.i16   cell index of each spike, int16, same order

6 bytes per spike, so those 346 million PKJ spikes are 2.1 GB rather than tens.
Times are stored as STEP INDICES, not milliseconds: at dt = 1 ms a 5-hour run is
18 million steps, which is exact in int32 where the float millisecond it stands
for would be carrying fractional zeros for nothing. Cell indices are int16
because no population here approaches 32,767 cells; `SpikeStream` asserts that
rather than trusting it.

Because the writer appends in time order the time file is sorted, which is what
makes the reader cheap: `SpikeStore.window` is two searchsorted calls into a
memory-mapped array, so cutting a 10-second raster out of the middle of a
5-hour run touches only the spikes in that window. Nothing here loads a whole
population's spikes unless you ask it to.
"""
import json
from pathlib import Path

import numpy as np

T_DTYPE = np.int32       # step index; 18e6 steps for a 5 h run at dt = 1 ms, well inside int32
C_DTYPE = np.int16       # cell index within its population


class SpikeStream:
    """Append-only writer for one population's spikes.

    Buffers into preallocated arrays and flushes every `chunk` spikes, so the
    per-step cost is a flatnonzero and a slice assignment -- no Python-object
    boxing per spike, which is what made the in-memory recorder expensive at
    length as much as the memory did.
    """

    def __init__(self, name, n_cells, out_dir, chunk=1 << 20):
        if n_cells > np.iinfo(C_DTYPE).max:
            raise ValueError(f"{name}: {n_cells} cells exceeds the int16 cell index")
        self.name, self.n_cells = name, n_cells
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._tf = open(out_dir / f"{name}.t.i32", "wb")
        self._cf = open(out_dir / f"{name}.c.i16", "wb")
        self._tbuf = np.empty(chunk, dtype=T_DTYPE)
        self._cbuf = np.empty(chunk, dtype=C_DTYPE)
        self._n = 0                                   # spikes currently in the buffer
        self.n_spikes = 0                             # spikes written over the whole run

    def record(self, step, spiked):
        idx = np.flatnonzero(spiked)
        k = len(idx)
        if not k:
            return
        if self._n + k > len(self._tbuf):
            self.flush()
            if k > len(self._tbuf):                   # a single step larger than the buffer: write it straight out
                self._tf.write(np.full(k, step, dtype=T_DTYPE).tobytes())
                self._cf.write(idx.astype(C_DTYPE).tobytes())
                self.n_spikes += k
                return
        self._tbuf[self._n:self._n + k] = step
        self._cbuf[self._n:self._n + k] = idx
        self._n += k
        self.n_spikes += k

    def flush(self):
        if self._n:
            self._tf.write(self._tbuf[:self._n].tobytes())
            self._cf.write(self._cbuf[:self._n].tobytes())
            self._n = 0
        self._tf.flush()
        self._cf.flush()

    def close(self):
        self.flush()
        self._tf.close()
        self._cf.close()


class SpikeStore:
    """Read side of a SpikeStream. Memory-maps the two files; never copies more
    than the slice asked for."""

    def __init__(self, name, n_cells, out_dir, dt_ms=1.0):
        self.name, self.n_cells, self.dt_ms = name, n_cells, dt_ms
        out_dir = Path(out_dir)
        self.t = np.memmap(out_dir / f"{name}.t.i32", dtype=T_DTYPE, mode="r")
        self.c = np.memmap(out_dir / f"{name}.c.i16", dtype=C_DTYPE, mode="r")
        if len(self.t) != len(self.c):
            raise ValueError(f"{name}: {len(self.t)} times vs {len(self.c)} cells -- truncated write?")

    def __len__(self):
        return len(self.t)

    def window(self, t0_s, t1_s):
        """(times_ms, cells) for spikes in [t0_s, t1_s). Two searchsorted calls
        into the sorted time file, then one copy of just that slice."""
        s0 = int(round(t0_s * 1000.0 / self.dt_ms))
        s1 = int(round(t1_s * 1000.0 / self.dt_ms))
        i0, i1 = np.searchsorted(self.t, [s0, s1])
        return np.asarray(self.t[i0:i1], dtype=float) * self.dt_ms, np.asarray(self.c[i0:i1])

    def trains(self, t0_s, t1_s):
        """Per-cell spike-time arrays (ms) over a window -- the shape the
        existing plotting code in sim/analysis.py expects."""
        t, c = self.window(t0_s, t1_s)
        order = np.argsort(c, kind="stable")
        t, c = t[order], c[order]
        bounds = np.searchsorted(c, np.arange(self.n_cells + 1))
        return [t[bounds[i]:bounds[i + 1]] for i in range(self.n_cells)]

    def rate_trace(self, bin_s, duration_s):
        """Population rate (Hz per cell) in bins of `bin_s`, over the whole run.

        Reads the time file in slabs rather than whole: a 346-million-spike
        int32 array is 1.4 GB, and binning it needs no more than one slab at a
        time."""
        steps_per_bin = max(1, int(round(bin_s * 1000.0 / self.dt_ms)))
        n_bins = int(np.ceil(duration_s * 1000.0 / self.dt_ms / steps_per_bin))
        counts = np.zeros(n_bins, dtype=np.int64)
        slab = 1 << 24
        for i in range(0, len(self.t), slab):
            b = np.asarray(self.t[i:i + slab], dtype=np.int64) // steps_per_bin
            counts += np.bincount(b, minlength=n_bins)[:n_bins]
        centres = (np.arange(n_bins) + 0.5) * bin_s
        return centres, counts / (bin_s * self.n_cells)

    def per_cell_counts(self, t0_s, t1_s):
        """Spike count per cell over a window, as an (n_cells,) array."""
        _, c = self.window(t0_s, t1_s)
        return np.bincount(c, minlength=self.n_cells)

    def isi_cv(self, t0_s, t1_s):
        """(mean ISI ms, CV) per cell over a window. Same definition as
        sim/analysis.isi_stats, computed off the store instead of a train list."""
        means, cvs = [], []
        for s in self.trains(t0_s, t1_s):
            if len(s) < 3:
                means.append(np.nan); cvs.append(np.nan); continue
            isi = np.diff(s)
            means.append(float(isi.mean())); cvs.append(float(isi.std() / isi.mean()))
        return np.asarray(means), np.asarray(cvs)


class RunBundle:
    """Everything one long run wrote, addressed by name.

    A bundle is a directory, not a pickle: the spike files stay memory-mapped
    and the arrays that ARE small enough to load (the slow weight traces, the
    membrane-potential windows, the final weight matrix) live in .npz files
    beside them. That means a figure script can open a 5-hour run and plot one
    10-second raster without reading 2 GB.
    """

    def __init__(self, path):
        self.path = Path(path)
        with open(self.path / "meta.json") as f:
            self.meta = json.load(f)
        dt, m = self.meta["dt_ms"], self.meta
        self.io = SpikeStore("io", m["n_io"], self.path, dt)
        self.pkj = SpikeStore("pkj", m["n_pkj"], self.path, dt)
        self.dcn = SpikeStore("dcn", m["n_dcn"], self.path, dt)
        self.pf = SpikeStore("pf", m["n_pf_recorded"], self.path, dt)
        self._slow = None
        self._final = None

    @property
    def duration_s(self):
        return self.meta["duration_s"]

    @property
    def slow(self):
        """Lazily loaded weight traces: t_s, mean_weight, pkj_mean_weight,
        sample_weights, tracked_synapses."""
        if self._slow is None:
            self._slow = np.load(self.path / "slow.npz")
        return self._slow

    @property
    def final(self):
        """Lazily loaded end-of-run state: final_weights, gap_matrix,
        io_of_pkj, group_of_dcn, pf_recorded."""
        if self._final is None:
            self._final = np.load(self.path / "final.npz")
        return self._final

    def trace(self, i):
        """One membrane-potential window, by index into meta['trace_windows_s']."""
        return np.load(self.path / f"trace_{i}.npz")

    @property
    def n_traces(self):
        return len(self.meta["trace_windows_s"])
