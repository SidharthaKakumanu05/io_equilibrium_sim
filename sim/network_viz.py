"""Drawing a NetworkGraph: the wired network in 3D, and its connectivity matrix.

Both figures are snapshots of the graph sim/network_graph.py builds from the
same Connectivity and gap-junction matrix the simulation runs on, so they show
the network that actually ran rather than a redrawing of the intent.
"""
import numpy as np

NODE_STYLE = {                          # colour, marker size, z-order for each cell type
    "PF":  {"color": "#b9c2cc", "size": 3,   "z": 1, "label": "PF (parallel fiber)"},
    "PKJ": {"color": "#2b6cb0", "size": 26,  "z": 3, "label": "PKJ (Purkinje)"},
    "DCN": {"color": "#dd8a1a", "size": 70,  "z": 4, "label": "DCN (deep nuclear)"},
    "IO":  {"color": "#c0392b", "size": 150, "z": 5, "label": "IO (inferior olive)"},
}

EDGE_STYLE = {                          # colour, width, alpha for each projection
    "pf_pkj":  {"color": "#95a5a6", "lw": 0.25, "alpha": 0.30},
    "pkj_dcn": {"color": "#2b6cb0", "lw": 0.55, "alpha": 0.45},
    "dcn_io":  {"color": "#dd8a1a", "lw": 1.30, "alpha": 0.85},
    "io_pkj":  {"color": "#c0392b", "lw": 0.60, "alpha": 0.55},
    "io_io":   {"color": "#8e44ad", "lw": 1.60, "alpha": 0.90},
}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _draw_edges(ax, graph, kinds, art3d):
    """One Line3DCollection per projection -- thousands of individual plot()
    calls would take longer to draw than the simulation takes to run."""
    for kind in kinds:
        sel = graph.edge_kind == kind
        if not sel.any():
            continue
        segments = np.stack([graph.position[graph.edge_src[sel]], graph.position[graph.edge_dst[sel]]], axis=1)
        style = EDGE_STYLE[kind]
        ax.add_collection3d(art3d.Line3DCollection(
            segments, colors=style["color"], linewidths=style["lw"], alpha=style["alpha"], zorder=2))


def _draw_nodes(ax, graph, kinds, legend=True):
    for kind in kinds:
        ids = graph.node_ids(kind)
        if not len(ids):
            continue
        style = NODE_STYLE[kind]
        p = graph.position[ids]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=style["size"], c=style["color"],
                    depthshade=False, edgecolors="white" if kind != "PF" else "none",
                    linewidths=0.4, zorder=style["z"], label=style["label"] if legend else None)


def _style_axes(ax, graph, title, aspect=None, pad=0.5):
    """Fit the axes to the drawn nodes. `aspect` defaults to the data's own
    proportions (with a floor, so a nearly-planar subgraph does not collapse to
    a line) rather than a fixed cube -- one group occupies a narrow radial
    wedge, and forcing that into a cube renders it as a diagonal streak."""
    p = graph.position
    lo, hi = p.min(axis=0) - pad, p.max(axis=0) + pad
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    if aspect is None:
        span = hi - lo
        aspect = tuple(np.maximum(span / span.max(), 0.30))
    ax.set_box_aspect(aspect)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_zlabel("")
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.03)


def _subgraph_local_frame(graph, group):
    """One group's nodes and the edges between them, with positions re-projected
    onto that group's own frame: x becomes the tangential offset within the
    group, y is flattened, z stays the anatomical height."""
    keep = np.flatnonzero(graph.group == group)
    mask = np.isin(graph.edge_src, keep) & np.isin(graph.edge_dst, keep)
    n_groups = int(graph.meta.get("n_io", graph.group.max() + 1))
    theta = 2.0 * np.pi * group / max(n_groups, 1)
    tangential = np.array([-np.sin(theta), np.cos(theta)])          # unit vector along the group's spread
    xy = graph.position[keep, :2]
    local = np.stack([xy @ tangential, np.zeros(len(keep)), graph.position[keep, 2]], axis=1)
    return type(graph)(kind=graph.kind[keep], group=graph.group[keep], index=graph.index[keep],
                        position=local,
                        edge_src=np.searchsorted(keep, graph.edge_src[mask]),
                        edge_dst=np.searchsorted(keep, graph.edge_dst[mask]),
                        edge_kind=graph.edge_kind[mask], edge_weight=graph.edge_weight[mask],
                        meta=graph.meta)


def plot_network_3d(graph, path, title="Network wiring", elev=18, azim=-62):
    """Three views of the same wired network: the whole thing, one closed-loop
    group in isolation, and the olive's gap-junction web on its own.

    Layers run bottom-to-top in the order signals traverse the loop back to the
    olive -- IO at the base, then DCN, then the Purkinje layer, then the
    parallel fibers -- and the groups fan out radially, so one closed loop reads
    as a vertical column and the electrical coupling reads as the horizontal web
    underneath everything.
    """
    plt = _mpl()
    from mpl_toolkits.mplot3d import art3d

    fig = plt.figure(figsize=(19, 7.5))
    all_kinds = ("PF", "PKJ", "DCN", "IO")

    # --- panel 1: the whole network ---
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    _draw_edges(ax, graph, ("pf_pkj", "io_pkj", "pkj_dcn", "dcn_io", "io_io"), art3d)
    _draw_nodes(ax, graph, all_kinds)
    _style_axes(ax, graph, f"whole network -- {graph.summary().split(';')[0]}", aspect=(1, 1, 0.75))
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)

    # --- panel 2: one group, redrawn in its own local frame so the loop is legible ---
    # In the global layout the layers sit at increasing radius, so a single group is a diagonal
    # wedge -- fine in panel 1, where the cone shows every group converging on the olive, but it
    # renders one group as a streak. Re-projecting onto (tangential offset, 0, height) turns that
    # group into the vertical column the pathway actually is.
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    sub = _subgraph_local_frame(graph, group=0)
    _draw_edges(ax2, sub, ("pf_pkj", "io_pkj", "pkj_dcn", "dcn_io"), art3d)
    _draw_nodes(ax2, sub, all_kinds, legend=False)
    _style_axes(ax2, sub, "one closed-loop group\nPF $\\to$ PKJ $\\dashv$ DCN $\\dashv$ IO $\\to$ CF back to PKJ",
                 aspect=(0.62, 0.18, 1.0))
    ax2.view_init(elev=6, azim=-90)
    x_label = sub.position[:, 0].min() - 0.35
    for kind, name in (("PF", "PF"), ("PKJ", "PKJ"), ("DCN", "DCN"), ("IO", "IO")):
        ids = sub.node_ids(kind)
        ax2.text(x_label, 0.0, float(sub.position[ids, 2].mean()), name,
                  fontsize=10, color=NODE_STYLE[kind]["color"], weight="bold")

    # --- panel 3: the olive's electrical coupling, alone ---
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    io_ids = graph.node_ids("IO")
    gap = graph.edge_kind == "io_io"
    if gap.any():
        segs = np.stack([graph.position[graph.edge_src[gap]], graph.position[graph.edge_dst[gap]]], axis=1)
        widths = 1.0 + 6.0 * graph.edge_weight[gap] / max(graph.edge_weight[gap].max(), 1e-12)
        ax3.add_collection3d(art3d.Line3DCollection(segs, colors=EDGE_STYLE["io_io"]["color"],
                                                     linewidths=widths, alpha=0.85))
    p = graph.position[io_ids]
    # Marker and label density have to follow n_io. At CbmSim's 4 cells a fat marker reads
    # well and every cell can be numbered; at 40 the same marker is wider than the spacing
    # between neighbours, which hides exactly the short local chords a nearest_k topology is
    # made of -- the panel then looks like an unconnected ring.
    n_io = len(io_ids)
    marker = float(np.clip(2600.0 / max(n_io, 1), 45.0, 260.0))
    ax3.scatter(p[:, 0], p[:, 1], p[:, 2], s=marker, c=NODE_STYLE["IO"]["color"],
                 depthshade=False, edgecolors="white", linewidths=0.8, zorder=3)
    label_every = max(1, n_io // 12)
    for n, i in enumerate(io_ids):
        if n % label_every == 0:
            ax3.text(*graph.position[i], f"  {n}", fontsize=8, color="black")
    ax3.set_xlim(p[:, 0].min() - 1, p[:, 0].max() + 1)
    ax3.set_ylim(p[:, 1].min() - 1, p[:, 1].max() + 1)
    ax3.set_zlim(-1, 1)
    ax3.set_box_aspect((1, 1, 0.4))
    ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])
    ax3.grid(False)
    n_gap = int(gap.sum())
    ax3.set_title(f"IO$\\leftrightarrow$IO gap junctions\n{graph.meta.get('gap_topology', '?')}, {n_gap} junctions",
                   fontsize=10)
    ax3.view_init(elev=62, azim=-60)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_connectivity_matrix(graph, path, title="Connectivity matrix"):
    """The adjacency matrix the graph induces, at three zoom levels.

    Nodes are ordered PF, PKJ, DCN, IO, so each projection of the circuit is one
    off-diagonal block and the whole matrix reads as the pathway. Excitatory
    entries are positive (blue), inhibitory negative (red); the gap-junction
    block is symmetric and signless, so it gets its own panel."""
    plt = _mpl()
    import matplotlib.colors as mcolors

    # Chemical synapses only, normalized per projection. Normalized because the four projections carry
    # different units, so one shared scale would render the whole PKJ->DCN block (weights ~0.008) as blank
    # next to the CF block (~1.0). Chemical-only because a gap junction has no sign, and drawing it on a
    # signed excitatory/inhibitory scale would colour it as though it were excitatory; it gets panel 3.
    CHEMICAL = ("pf_pkj", "io_pkj", "pkj_dcn", "dcn_io")
    a = graph.adjacency_matrix(signed=True, kinds=CHEMICAL, normalize_per_kind=True)
    blocks = graph.kind_blocks()
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2),
                              gridspec_kw={"width_ratios": [1.25, 1.25, 0.9]})

    def draw(ax, m, labels, sub_title, tick_every=None):
        scale = np.abs(m).max() or 1.0
        norm = mcolors.TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale)
        im = ax.imshow(m, cmap="RdBu_r", norm=norm, interpolation="nearest", aspect="auto")
        for _, start, stop in labels:
            for edge in (start, stop):
                ax.axhline(edge - 0.5, color="black", linewidth=0.7)
                ax.axvline(edge - 0.5, color="black", linewidth=0.7)
        # Only label blocks wide enough to hold a label: PF is 85% of the axis, so DCN and IO
        # would otherwise print on top of each other at the far edge.
        n = m.shape[0]
        keep = [(k, s, e) for k, s, e in labels if (e - s) / n > 0.06]
        ax.set_xticks([(s + e) / 2.0 - 0.5 for _, s, e in keep])
        ax.set_xticklabels([k for k, _, _ in keep], fontsize=9)
        ax.set_yticks([(s + e) / 2.0 - 0.5 for _, s, e in keep])
        ax.set_yticklabels([k for k, _, _ in keep], fontsize=9)
        ax.set_xlabel("postsynaptic"); ax.set_ylabel("presynaptic")
        ax.set_title(sub_title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.045,
                      label="weight / projection max\n(+ excitatory, - inhibitory)")

    n_chem = int(np.isin(graph.edge_kind, CHEMICAL).sum())
    draw(axes[0], a, blocks, f"chemical synapses: {graph.n_nodes} x {graph.n_nodes}, {n_chem} edges\n"
                              "(PF fills 85% of the axis; see the next panel for the rest)")

    # --- panel 2: drop the PF block, which is 85% of the nodes and one diagonal stripe ---
    start = next(s for k, s, _ in blocks if k == "PKJ")
    sub = a[start:, start:]
    sub_blocks = [(k, s - start, e - start) for k, s, e in blocks if k != "PF"]
    draw(axes[1], sub, sub_blocks,
          f"central circuit, chemical synapses only (PKJ/DCN/IO): {sub.shape[0]} x {sub.shape[0]}")

    # --- panel 3: the gap-junction matrix on its own ---
    gap = graph.adjacency_matrix(signed=False, kinds=("io_io",))
    io_start = next(s for k, s, _ in blocks if k == "IO")
    gap = gap[io_start:, io_start:]
    im = axes[2].imshow(gap, cmap="Purples", interpolation="nearest")
    axes[2].set_xticks(range(len(gap))); axes[2].set_yticks(range(len(gap)))
    axes[2].set_xlabel("IO cell"); axes[2].set_ylabel("IO cell")
    axes[2].set_title(f"IO$\\leftrightarrow$IO gap junctions\n({graph.meta.get('gap_topology', '?')}, symmetric)",
                       fontsize=10)
    fig.colorbar(im, ax=axes[2], fraction=0.045, label="$g_{gap}$ (mS/cm$^2$)")

    note = (f"PF nodes are a {graph.meta.get('n_pf_per_pkj_shown', '?')}-of-"
            f"{graph.meta.get('n_pf_per_pkj_actual', '?')} subsample per Purkinje cell; "
            "every PKJ, DCN and IO cell is shown in full. "
            "Each projection is scaled to its own maximum -- they are in different units.")
    fig.suptitle(f"{title}\n{note}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=150)
    plt.close(fig)
