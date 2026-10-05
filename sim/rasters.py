"""Spike rasters saved in CbmSim's binary format.

This matches Control::newRasters in CbmSim (src/control.cpp), so files from
this simulation can be read with the same tools as files from CbmSim.

THE FORMAT. One file per cell type. Each file is a flat list of signed integers
(int16 for most cells; int32 for granule cells / PFs, which have too many
indices for int16), written trial by trial:

    -2, trial                       start of a trial (trial counts from 0)
    -1, ts, cell, cell, ...         one time step: the indices of the cells
    -1, ts, cell, ...               that spiked on step ts (ts counts from 0
    ...                             within the trial; a step with no spikes is
    -2, trial                       just "-1, ts")
    -1, ts, ...

Every time step of every trial gets a "-1, ts" entry. Numbers are stored
little-endian, as CbmSim writes them on x86.

File names follow CbmSim: <base name> + the extension below.

  this sim   CbmSim   extension  dtype
  IO         IO       .ior       int16
  PKJ        PC       .pcr       int16
  DCN        NC       .ncr       int16
  PF         GR       .grr       int32
"""
import numpy as np

FORMATS = {                                  # cell type -> (file extension, integer type)
    "io":  (".ior", np.dtype("<i2")),
    "pkj": (".pcr", np.dtype("<i2")),
    "dcn": (".ncr", np.dtype("<i2")),
    "pf":  (".grr", np.dtype("<i4")),
}
NEW_TRIAL, NEW_STEP = -2, -1                 # the two markers


class CbmRasterWriter:
    """Writes one cell type's raster file while the simulation runs.

    Call start_trial() at the start of each trial, record() on every time step,
    and end_trial() at the end; each trial is appended to the file in one write,
    as in CbmSim. The file is emptied when the writer is created.
    """

    def __init__(self, path, dtype):
        self.path = path
        self.dtype = np.dtype(dtype)
        self._chunks = []
        open(path, "wb").close()             # start from an empty file

    def start_trial(self, trial):
        self._chunks = [np.array([NEW_TRIAL, trial], dtype=self.dtype)]

    def record(self, ts, spiked):
        """ts: time step within the trial. spiked: True/False per cell, or the
        indices of the cells that spiked."""
        spiked = np.asarray(spiked)
        idx = np.flatnonzero(spiked) if spiked.dtype == bool else spiked
        self._chunks.append(np.array([NEW_STEP, ts], dtype=self.dtype))
        if len(idx):
            self._chunks.append(idx.astype(self.dtype))

    def end_trial(self):
        with open(self.path, "ab") as f:
            f.write(np.concatenate(self._chunks).tobytes())
        self._chunks = []


def read_cbm_raster(path, dtype=None):
    """Read a CbmSim-format raster file.

    Returns three arrays with one entry per spike: (trial, ts, cell). For
    example, the PKJ cells that fired on step 120 of trial 3 are
    cell[(trial == 3) & (ts == 120)]. dtype is worked out from the file
    extension if not given (.grr = int32, everything else int16).
    """
    if dtype is None:
        dtype = np.dtype("<i4") if str(path).endswith(".grr") else np.dtype("<i2")
    data = np.fromfile(path, dtype=dtype).astype(np.int64)
    pos = np.arange(len(data))

    is_trial_marker, is_step_marker = data == NEW_TRIAL, data == NEW_STEP
    is_header = np.zeros(len(data), dtype=bool)                  # the number right after a marker
    is_header[1:] = is_trial_marker[:-1] | is_step_marker[:-1]
    is_cell = ~is_trial_marker & ~is_step_marker & ~is_header

    # For every entry, find the most recent trial marker and step marker before it,
    # then read the trial / ts value stored right after that marker.
    last_trial = np.maximum.accumulate(np.where(is_trial_marker, pos, -1))
    last_step = np.maximum.accumulate(np.where(is_step_marker, pos, -1))
    trial = data[np.minimum(last_trial + 1, len(data) - 1)]
    ts = data[np.minimum(last_step + 1, len(data) - 1)]
    return trial[is_cell], ts[is_cell], data[is_cell]
