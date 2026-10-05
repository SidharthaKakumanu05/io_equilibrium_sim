#!/usr/bin/env python3
"""Set up and run simulations from a web page.

    python3 launcher.py               # then open the link it prints
    python3 launcher.py --port 8700   # choose the port (default 8700)

The page has every setting from run.py's SETTINGS block, a Run and a Stop
button, the live view of the running network (sim/live_view.html), and a
console showing the run's output. Leave the launcher running and start as many
runs from the page as you like, one at a time.

HOW IT WORKS
  * Settings come straight from run.py: the launcher reads run.py's SETTINGS
    block every time the page loads, and shows each setting with its comment as
    help text. A setting you add to run.py appears on the page automatically.
  * Run starts `python3 run.py NAME=VALUE ...` as a separate process, with only
    the values you changed, exactly as if you typed it in a terminal. The
    command is shown in the console, so any run can be repeated from the
    command line. If the run crashes, the launcher carries on.
  * The run's live view (sim/live_view.py) is relayed through this server, so
    only this one port needs to be forwarded to your browser.
  * The link contains a random token, as Jupyter's does. Starting or stopping a
    run needs it, so other users on this machine cannot start runs as you even
    though they can reach localhost. Open the page with the printed link.
"""
import argparse
import ast
import json
import re
import secrets
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
RUN_PY = HERE / "run.py"
PAGE = HERE / "sim" / "live_view.html"
HIDDEN = {"LIVE_VIEW", "LIVE_PORT"}     # set by the launcher itself, so not shown on the page
CHILD_LIVE_PORT = 8801                  # where a run's own live view listens (relayed, not opened directly)
SAFE_TEXT = re.compile(r"^[A-Za-z0-9_.,/ +-]*$")   # allowed characters in text settings (e.g. OUTPUT_DIR)
MAX_LOG_LINES = 5000


# ---------------------------------------------------------------------------
#  Reading the settings out of run.py
# ---------------------------------------------------------------------------

def read_settings():
    """Parse run.py's SETTINGS block into a list of sections.

    Each section is {"title", "advanced", "items"}, and each item is either a
    note {"note": text} (a comment line in run.py) or a setting
    {"name", "value", "kind", "help"}. `kind` decides the input on the page:
    "toggle" (a 0/1 switch), "int", "float", "optional" (blank = None) or "text".
    """
    src = RUN_PY.read_text()
    lines = src.split("\n")
    stop = next(i for i, line in enumerate(lines) if "Machinery below" in line)
    assigns = {node.lineno - 1: node for node in ast.parse(src).body
               if isinstance(node, ast.Assign) and len(node.targets) == 1
               and isinstance(node.targets[0], ast.Name) and node.targets[0].id.isupper()
               and node.lineno - 1 < stop}

    sections, last_was_note = [], False

    def section(title, advanced=False):
        sections.append({"title": title, "advanced": advanced, "items": []})

    i = 0
    while i < stop:
        line, text = lines[i], lines[i].strip()
        if i in assigns:                                         # a setting
            node = assigns[i]
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                i += 1
                continue
            rest = line[node.end_col_offset:]
            help_text = rest.split("#", 1)[1].strip() if "#" in rest else ""
            j = i + 1
            while j < stop and lines[j].startswith(" ") and lines[j].strip().startswith("#"):
                help_text += " " + lines[j].strip()[1:].strip()     # indented comment lines continue it
                j += 1
            name = node.targets[0].id
            if not sections:
                section("General")
            if name not in HIDDEN:
                sections[-1]["items"].append({"name": name, "value": value, "help": " ".join(help_text.split()),
                                              "kind": _kind(name, value, help_text)})
            last_was_note, i = False, j
            continue
        header = re.match(r"#\s*-{2,}\s*(.*?)\s*-*$", text) if line.startswith("#") else None
        if header and header.group(1):                           # "# --- title ---" starts a section
            section(header.group(1))
            last_was_note = False
        elif text.startswith("#  ADVANCED"):
            section("Advanced: the calibrated operating point", advanced=True)
            last_was_note = False
        elif (line.startswith("#") and not set(text) <= set("#= ") and "SETTINGS --" not in text
              and sections):                                     # any other comment: a note
            note = text.lstrip("#").strip()
            items = sections[-1]["items"]
            if last_was_note and items and "note" in items[-1]:
                items[-1]["note"] += " " + note
            else:
                items.append({"note": note})
            last_was_note = True
        else:
            last_was_note = False
        i += 1
    return [s for s in sections if any("name" in item for item in s["items"])]


def _kind(name, value, help_text):
    if value is None:
        return "optional"
    if isinstance(value, bool) or (isinstance(value, int) and value in (0, 1)
                                   and (name.startswith(("PLOT_", "SAVE_")) or re.search(r"\b[01] =", help_text))):
        return "toggle"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "text"


def command_args(values):
    """Check the values sent by the page and turn the ones that differ from
    run.py into NAME=VALUE arguments. Raises ValueError with a readable message."""
    settings = {item["name"]: item for s in read_settings() for item in s["items"] if "name" in item}
    args = []
    for name, raw in values.items():
        if name not in settings:
            raise ValueError(f"unknown setting {name}")
        item = settings[name]
        kind, default = item["kind"], item["value"]
        try:
            if kind == "toggle":
                value = 1 if raw in (True, 1, "1", "true") else 0
            elif kind == "int":
                value = int(raw)
                if value != float(raw):
                    raise ValueError
            elif kind == "float":
                value = float(raw)
            elif kind == "optional":
                value = None if raw in (None, "") else (int(raw) if float(raw) == int(float(raw)) else float(raw))
            else:
                value = str(raw)
                if not SAFE_TEXT.match(value):
                    raise ValueError
        except (TypeError, ValueError):
            raise ValueError(f"{name}: {raw!r} is not a valid {kind} value")
        if value != default:
            args.append(f"{name}={value!r}")                     # repr: run.py reads it back with literal_eval
    return args


# ---------------------------------------------------------------------------
#  The run: one child process at a time
# ---------------------------------------------------------------------------

class Runner:
    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.state = "ready"            # ready | running | finished | stopped | failed
        self.port = None                # the running sim's live-view port, once it has printed it
        self.lines, self.first_line = [], 0     # console output; first_line = number of lines dropped
        self.command = ""
        self._stopping = False

    def start(self, args):
        with self.lock:
            if self.state == "running":
                raise RuntimeError("a run is already going; stop it first")
            cmd = [sys.executable, "-u", str(RUN_PY), *args, "LIVE_VIEW=1", f"LIVE_PORT={CHILD_LIVE_PORT}"]
            self.command = "python3 run.py " + " ".join(args)
            self._log(f"$ {self.command}")
            self.proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, bufsize=1)
            self.state, self.port, self._stopping = "running", None, False
        threading.Thread(target=self._follow, args=(self.proc,), daemon=True).start()

    def stop(self):
        with self.lock:
            if self.state != "running" or self.proc is None:
                return
            self._stopping = True
            self.proc.terminate()
        threading.Timer(5.0, lambda p=self.proc: p.poll() is None and p.kill()).start()

    def _follow(self, proc):
        """Copy the run's output into the console, and notice its live-view port."""
        for line in proc.stdout:
            line = line.rstrip("\n")
            m = re.search(r"live view\s+http://localhost:(\d+)", line)
            with self.lock:
                if m:
                    self.port = int(m.group(1))
                    line = "  live view      shown on this page"     # its own port is relayed, not opened
                self._log(line)
        code = proc.wait()
        with self.lock:
            self.state = "stopped" if self._stopping else ("finished" if code == 0 else "failed")
            self.port = None
            self._log(f"[run {self.state}" + (f", exit code {code}]" if self.state == "failed" else "]"))

    def _log(self, line):                    # called with the lock held
        self.lines.append(line)
        if len(self.lines) > MAX_LOG_LINES:
            drop = len(self.lines) - MAX_LOG_LINES
            del self.lines[:drop]
            self.first_line += drop

    def log_since(self, n):
        with self.lock:
            start = max(n, self.first_line)
            return {"lines": self.lines[start - self.first_line:], "next": self.first_line + len(self.lines),
                    "state": self.state, "command": self.command}


# ---------------------------------------------------------------------------
#  The web server
# ---------------------------------------------------------------------------

def make_handler(runner, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/api/settings":
                self._json(200, {"sections": read_settings()})
            elif url.path == "/api/log":
                since = int(parse_qs(url.query).get("since", ["0"])[0])
                self._json(200, runner.log_since(since))
            elif url.path == "/events":
                relay_events(self, runner)
            else:
                self.send_error(404)

        def do_POST(self):
            if not secrets.compare_digest(self.headers.get("X-Token", ""), token):
                return self._json(403, {"error": "missing or wrong token: open the page with the link "
                                                 "printed by launcher.py"})
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/run":
                try:
                    runner.start(command_args(body.get("values", {})))
                except (ValueError, RuntimeError) as exc:
                    return self._json(400, {"error": str(exc)})
                self._json(200, {"ok": True, "command": runner.command})
            elif self.path == "/api/stop":
                runner.stop()
                self._json(200, {"ok": True})
            else:
                self.send_error(404)

    return Handler


def relay_events(handler, runner):
    """The page's /events stream. Between runs it sends the launcher's state;
    while a run is going it relays that run's own live-view stream."""
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.end_headers()
    sent_state, last_write = None, 0.0
    try:
        while True:
            with runner.lock:
                state, port = runner.state, runner.port
            if state == "running" and port:
                try:
                    upstream = urllib.request.urlopen(f"http://127.0.0.1:{port}/events", timeout=30)
                except OSError:
                    time.sleep(0.5)                              # the run's server is not up yet
                    continue
                with upstream:
                    while True:
                        chunk = upstream.read1(65536)
                        if not chunk:
                            return                               # run over; the page reconnects
                        handler.wfile.write(chunk)
                        handler.wfile.flush()
            if state != sent_state:
                handler.wfile.write(f"event: status\ndata: {json.dumps({'text': state})}\n\n".encode())
                sent_state, last_write = state, time.time()
            elif time.time() - last_write > 15:
                handler.wfile.write(b": still here\n\n")
                last_write = time.time()
            handler.wfile.flush()
            time.sleep(0.5)
    except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
        return


def main():
    parser = argparse.ArgumentParser(description="Run io_equilibrium_sim from a web page.")
    parser.add_argument("--port", type=int, default=8700)
    port = parser.parse_args().port

    runner, token = Runner(), secrets.token_urlsafe(16)
    for p in range(port, port + 50):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), make_handler(runner, token))
            break
        except OSError:
            continue
    else:
        raise SystemExit(f"no free port in {port}-{port + 49}")
    server.daemon_threads = True
    url = f"http://localhost:{server.server_address[1]}/?token={token}"
    print(f"\nio_equilibrium_sim launcher\n\n  open:  {url}\n\n"
          "  (VS Code forwards the port; use this full link, the token lets the page start runs)\n"
          "  Ctrl-C here stops the launcher and any run it started.\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()
        server.server_close()
        print("launcher stopped.")


if __name__ == "__main__":
    main()
