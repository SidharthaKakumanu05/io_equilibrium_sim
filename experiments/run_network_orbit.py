#!/usr/bin/env python3
"""The whole network as a 3D model you can actually turn: an orbiting GIF, and
a contact sheet of fixed viewpoints.

experiments/run_network_snapshot.py already draws the network in 3D, but from
one viewpoint, and a single projection of a 3,000-node graph is ambiguous --
which layer a node sits in and whether two edges cross or meet are exactly the
things one angle cannot tell you. Rotating it resolves both.

Nothing here simulates. It builds the same Connectivity and gap-junction matrix
sim/simulate.py would build from the config and draws that, so it is a picture
of the network that runs. Optionally it colours the PF edges by the LIVE weights
from a finished run bundle (--bundle), which turns the same model into a picture
of what plasticity did to the wiring.

  network_orbit.gif     one full turn, --frames frames
  network_views.png     a grid of fixed (elevation, azimuth) views
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SimConfig
from sim.connectivity import build_connectivity
from sim.io_coupling import build_gap_junction_matrix, coupling_summary
from sim.network_graph import build_network_graph
from sim.network_viz import _draw_edges, _draw_nodes, _mpl, _style_axes

KINDS = ("PF", "PKJ", "DCN", "IO")
EDGES = ("pf_pkj", "io_pkj", "pkj_dcn", "dcn_io", "io_io")


def _render(graph, elev, azim, title, figsize=(7.5, 7.0), legend=False, dpi=110):
    """One view of the whole network, returned as an RGB array."""
    plt = _mpl()
    from mpl_toolkits.mplot3d import art3d
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    _draw_edges(ax, graph, EDGES, art3d)
    _draw_nodes(ax, graph, KINDS, legend=legend)
    _style_axes(ax, graph, title, aspect=(1, 1, 0.75))
    ax.view_init(elev=elev, azim=azim)
    if legend:
        ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.canvas.draw()
    # buffer_rgba rather than tostring_rgb: the latter was removed in matplotlib 3.10.
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def orbit_gif(graph, path, frames=72, elev=18, duration_ms=80):
    """One full turn of azimuth, written as an animated GIF.

    Frames are rendered one at a time and handed straight to Pillow rather than
    kept as figures: 72 matplotlib figures holding a 4,240-edge collection each
    is a lot of memory for something that is about to become 72 small images."""
    from PIL import Image
    imgs = []
    for k in range(frames):
        azim = -180.0 + 360.0 * k / frames
        imgs.append(Image.fromarray(_render(graph, elev, azim,
                                            f"{graph.summary().split(';')[0]}",
                                            legend=(k == 0))))
        if (k + 1) % 12 == 0:
            print(f"  ...{k + 1}/{frames} frames", flush=True)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], loop=0,
                 duration=duration_ms, optimize=True)
    return path


def view_grid(graph, path, views=None):
    """A contact sheet of fixed viewpoints, for when a still is wanted.

    The defaults are chosen to show the three things the single snapshot view
    cannot: straight down the axis (the territorial fan), edge-on (the layer
    stack), and from below (the olive and its gap-junction web)."""
    plt = _mpl()
    if views is None:
        views = [(90, -90, "from above -- territorial fan"),
                 (45, -60, "three-quarter"),
                 (18, -62, "default snapshot view"),
                 (2, 0, "edge-on -- the layer stack"),
                 (2, -90, "edge-on, quarter turn"),
                 (-35, -60, "from below -- the olive")]
    n = len(views)
    cols = 3
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(6.0 * cols, 5.4 * rows))
    imgs = [_render(graph, e, a, t, figsize=(6.0, 5.4), legend=(i == 0))
            for i, (e, a, t) in enumerate(views)]
    for i, img in enumerate(imgs):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.axis("off")
    fig.suptitle(f"The whole network from six viewpoints -- {graph.summary()}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="results/long")
    p.add_argument("--frames", type=int, default=72, help="frames in one full turn (default: 72)")
    p.add_argument("--elev", type=float, default=18.0, help="elevation the orbit is flown at")
    p.add_argument("--n-pf-shown", type=int, default=8,
                   help="PF units drawn per Purkinje cell; all 500 is not drawable (default: 8)")
    p.add_argument("--bundle", default=None,
                   help="a finished run bundle, to colour PF edges by the live final weights "
                        "instead of w_init")
    p.add_argument("--no-gif", action="store_true", help="contact sheet only")
    args = p.parse_args()

    cfg = SimConfig()
    conn = build_connectivity(
        n_io=cfg.n_io, n_dcn=cfg.n_dcn, n_pkj_per_io=cfg.n_pkj_per_io,
        n_pkj_per_dcn=cfg.n_pkj_per_dcn, n_dcn_per_pkj=cfg.n_dcn_per_pkj,
        n_dcn_per_io=cfg.n_dcn_per_io, enforce_closed_loop=cfg.enforce_closed_loop,
        seed=cfg.connectivity_seed)
    gap = build_gap_junction_matrix(cfg.n_io, cfg.gap_g, cfg.gap_topology, cfg.gap_n_neighbors)

    weights = None
    if args.bundle:
        from sim.recording import RunBundle
        weights = np.asarray(RunBundle(args.bundle).final["final_weights"])
        print(f"[orbit] PF edges coloured by the final weights of {args.bundle} "
              f"(mean {weights.mean():.4f})")
    graph = build_network_graph(cfg, conn, gap, weights=weights,
                                n_pf_per_pkj_shown=args.n_pf_shown, seed=0)
    print(f"[orbit] {conn.describe()}")
    print(f"[orbit] gap junctions: {coupling_summary(gap, cfg.io_channels.g_leak)}")
    print(f"[orbit] graph: {graph.summary()}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[orbit] rendering the contact sheet...", flush=True)
    view_grid(graph, out / "network_views.png")
    if not args.no_gif:
        print(f"[orbit] rendering {args.frames} orbit frames...", flush=True)
        orbit_gif(graph, out / "network_orbit.gif", frames=args.frames, elev=args.elev)
    print(f"[orbit] wrote {out}/network_views.png"
          f"{'' if args.no_gif else f' and {out}/network_orbit.gif'}")


if __name__ == "__main__":
    main()
