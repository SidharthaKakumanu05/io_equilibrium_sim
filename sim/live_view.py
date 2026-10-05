"""Live view: watch the network in a web browser while the simulation runs.

run.py starts this when LIVE_VIEW = 1 and prints an address such as
http://localhost:8765. Open it in a browser (VS Code forwards the port from the
server to your machine). The page shows:

  * the network in 3D (the same layout as sim/network_graph.py): cells flash
    when they fire, PF -> PKJ synapses are coloured by their current weight, and
    each CF event lights up the climbing fibre down to its 8 Purkinje cells
  * a scrolling spike raster and olive membrane potentials
  * CF rate, PKJ / DCN rates and mean weight, building up over the whole run

HOW IT WORKS. A small web server runs on a background thread inside the
simulation process; it uses only the Python standard library.

  1. Simulation.run calls on_step() every time step. on_step() collects that
     step's spikes, and every frame_ms of simulated time (50 ms) bundles them
     into one "frame", together with olive voltages and the current weights.
  2. Each frame goes to every open browser tab as a Server-Sent Event, a
     one-way stream of messages over an ordinary HTTP connection (/events).
  3. The page (sim/live_view.html) draws everything itself. A tab opened
     mid-run first receives an "init" message: the network layout, the
     history so far, and the last few seconds of frames.

The browser can only watch; it cannot change the simulation. If the page
crashes or is closed, the run carries on. The server only listens on
localhost (127.0.0.1).
"""
import json
import socket
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from sim.network_graph import build_network_graph

PAGE = Path(__file__).with_name("live_view.html")
EDGE_CODES = {"pf_pkj": 0, "io_pkj": 1, "pkj_dcn": 2, "dcn_io": 3, "io_io": 4}   # sent to the page as numbers
NODE_CODES = {"PF": 0, "PKJ": 1, "DCN": 2, "IO": 3}


class LiveView:
    """Collects frames from a running Simulation and serves them to browsers.

    Usage (this is what run.py does):
        view = LiveView(port=8765)
        url = view.start(sim, header)        # before sim.run()
        sim.run(..., on_step=view.on_step)
        view.finish()
    """

    def __init__(self, port=8765, frame_ms=50.0, n_pf_shown=2, n_io_traces=5, recent_ms=3000.0):
        self.first_port = int(port)
        self.frame_ms = float(frame_ms)       # simulated time covered by one frame
        self.n_pf_shown = int(n_pf_shown)     # PFs drawn per Purkinje cell (all 500 would be unreadable)
        self.n_io_traces = int(n_io_traces)   # olive cells whose voltage is sent every step
        self.recent_ms = float(recent_ms)     # how much recent detail a newly opened tab receives

        self._cond = threading.Condition()    # guards everything below; notified on every new message
        self._messages = deque()              # (sequence number, SSE text) of recent messages
        self._seq = 0                         # sequence number of the newest message
        self._history = []                    # one small summary per frame, for the run-long charts
        self._static = None                   # the "init" payload pieces that never change
        self._status = "starting"
        self._closed = False

    # --- setup -------------------------------------------------------------

    def start(self, sim, header, predicted_hz=None):
        """Build the static description of the network and start the server.
        `header` is a dict of labels shown at the top of the page; predicted_hz
        is drawn as a reference line on the CF-rate chart. Returns the URL to open."""
        cfg = sim.cfg
        self._sim = sim
        self._frame_steps = max(1, int(round(self.frame_ms / cfg.dt_ms)))
        self._k = min(self.n_pf_shown, cfg.n_pf_per_pkj)
        self._trace_cells = np.unique(np.linspace(0, cfg.n_io - 1, min(self.n_io_traces, cfg.n_io)).astype(int))
        self._pkj_per_io = np.bincount(sim.io_of_pkj, minlength=cfg.n_io).astype(float)
        self._reset_frame()
        self._wall_start = None

        # The network layout and wiring, from the same tables the simulation runs on.
        graph = build_network_graph(cfg, sim.conn, sim.gap_matrix, sim.weights,
                                    n_pf_per_pkj_shown=self._k, seed=0)
        self._static = {
            "run_id": f"{time.time():.3f}",           # lets an open tab notice that a new run started
            "header": header, "predicted_hz": predicted_hz,
            "dt_ms": cfg.dt_ms, "frame_ms": self._frame_steps * cfg.dt_ms,
            "n_trials": cfg.n_trials, "trial_ms": cfg.trial_ms,
            "n_io": cfg.n_io, "n_dcn": sim.conn.n_dcn, "n_pkj": sim.conn.n_pkj, "n_pf_shown": self._k,
            "trace_cells": self._trace_cells.tolist(),
            "v_spike_mv": cfg.io_channels.v_spike_mv,
            "node_kind": [NODE_CODES[k] for k in graph.kind],
            "node_pos": np.round(graph.position, 3).ravel().tolist(),        # x, y, z per node
            "edge_src": graph.edge_src.tolist(), "edge_dst": graph.edge_dst.tolist(),
            "edge_kind": [EDGE_CODES[k] for k in graph.edge_kind],
        }

        self._server = self._bind()
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return f"http://localhost:{self._server.server_address[1]}"

    def _bind(self):
        """Start the HTTP server on the first free port from first_port upward,
        so several runs at once each get their own page."""
        view = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = PAGE.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/events":
                    view._stream(self)
                else:
                    self.send_error(404)

            def log_message(self, *args):          # keep the terminal quiet
                pass

        for port in range(self.first_port, self.first_port + 50):
            try:
                server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
                server.daemon_threads = True
                return server
            except OSError:
                continue
        raise OSError(f"no free port in {self.first_port}-{self.first_port + 49} for the live view")

    # --- called by the simulation ------------------------------------------

    def status(self, text):
        """Show a status word on the page, e.g. "burn-in"."""
        self._status = text
        self._publish("status", {"text": text})

    def on_step(self, sim, trial, ts, t_ms, pf_spikes, cf_events, pkj_spiked, dcn_spiked):
        """Called by Simulation.run after every step of every trial."""
        if self._wall_start is None:
            self._wall_start = time.time()
            self.status("running")
        f, off = self._frame, self._frame["n"]
        for name, spiked in (("io", cf_events), ("pkj", pkj_spiked), ("dcn", dcn_spiked),
                             ("pf", pf_spikes[:, :self._k])):          # PF: the drawn subsample only
            idx = np.flatnonzero(spiked)                                # PF index = PKJ row x k + fibre
            if len(idx):
                f[name].append(np.column_stack([np.full(len(idx), off), idx]))
        f["iov"].append(sim.io.V[self._trace_cells].copy())
        f["n"] += 1
        f["trial"], f["t_end"] = trial, t_ms
        if f["n"] == self._frame_steps:
            self._emit_frame()

    def finish(self):
        """Send any partial frame and tell the page the run is over."""
        if self._frame["n"]:
            self._emit_frame()
        self.status("finished")
        self._publish("done", {})
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    # --- frames ------------------------------------------------------------

    def _reset_frame(self):
        self._frame = {"n": 0, "io": [], "pkj": [], "dcn": [], "pf": [], "iov": [], "trial": 0, "t_end": 0.0}

    def _emit_frame(self):
        f, sim = self._frame, self._sim
        n = f["n"]
        t_start = f["t_end"] - n * sim.cfg.dt_ms                     # time just before the frame's first step

        def flat(chunks):                                             # [[off, cell], ...] -> [off, cell, off, cell, ...]
            return np.concatenate(chunks).ravel().tolist() if chunks else []

        row_mean = sim.weights.mean(axis=1)
        w_terr = np.bincount(sim.io_of_pkj, row_mean, minlength=len(self._pkj_per_io)) / self._pkj_per_io
        counts = {name: sum(len(c) for c in f[name]) for name in ("io", "pkj", "dcn")}
        summary = {"t": round(f["t_end"], 3), "n": n, "io": counts["io"], "pkj": counts["pkj"],
                   "dcn": counts["dcn"], "w": round(float(row_mean.mean()), 5),
                   "wt": np.round(w_terr, 4).tolist()}
        frame = dict(summary,
                     t0=round(t_start, 3), trial=f["trial"],
                     wall=round(time.time() - self._wall_start, 3),
                     sp_io=flat(f["io"]), sp_pkj=flat(f["pkj"]), sp_dcn=flat(f["dcn"]), sp_pf=flat(f["pf"]),
                     iov=np.round(np.array(f["iov"]), 1).ravel().tolist(),
                     pfw=np.round(sim.weights[:, :self._k], 3).ravel().tolist())
        with self._cond:
            self._history.append(summary)
        self._publish("frame", frame)
        self._reset_frame()

    # --- delivery ----------------------------------------------------------

    def _publish(self, event, payload):
        """Queue one message for every open tab, keeping only recent ones."""
        text = f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"
        keep = int(self.recent_ms / self.frame_ms) + 10
        with self._cond:
            self._seq += 1
            self._messages.append((self._seq, event, text))
            while len(self._messages) > keep:
                self._messages.popleft()
            self._cond.notify_all()

    def _init_message(self):
        """What a newly opened tab gets first: layout, history so far, recent frames.
        Called with the lock held."""
        stride = max(1, len(self._history) // 2000)                   # cap the history at ~2000 points
        init = dict(self._static, status=self._status, history=self._history[::stride])
        recent = "".join(text for _, event, text in self._messages if event == "frame")
        return f"event: init\ndata: {json.dumps(init, separators=(',', ':'))}\n\n" + recent

    def _stream(self, handler):
        """Serve one /events connection until the run ends or the tab closes."""
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        with self._cond:
            first = self._init_message()
            cursor = self._seq
        try:
            handler.wfile.write(first.encode())
            handler.wfile.flush()
            while True:
                with self._cond:
                    self._cond.wait_for(lambda: self._seq > cursor or self._closed, timeout=15.0)
                    new = [(s, text) for s, _, text in self._messages if s > cursor]
                    closed = self._closed
                if new:
                    cursor = new[-1][0]
                    handler.wfile.write("".join(text for _, text in new).encode())
                else:
                    handler.wfile.write(b": still here\n\n")             # keeps idle connections open
                handler.wfile.flush()
                if closed and not new:
                    return
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            return                                                     # the tab was closed
