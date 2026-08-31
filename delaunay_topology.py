#!/usr/bin/env python3
"""
delaunay_topology.py

Interactive analysis of a 2D particle configuration:
  - Loads particle positions from a data file with two columns (x y).
  - Lets you interactively filter out particles beyond a chosen radius
    from the center of mass (slider).
  - Lets you repeatedly "peel" the convex hull (button) — each click
    permanently deletes the current convex-hull particles, so you can
    strip away boundary layers as many times as you like.
  - Computes and plots the Delaunay triangulation.
  - Computes, for every remaining vertex i:
        z_i  = coordination number (# of Delaunay neighbors)
        q_i  = 6 - z_i   if i is an interior vertex
        q_i  = 4 - z_i   if i is on the convex hull
  - Shades triangles touching a defective vertex (q_i != 0) so bound
    dislocation pairs are easy to spot.
  - Shows histograms of z_i and q_i.
  - Prints (and displays) the total topological charge sum_i q_i.
  - "Export panels" button: dumps the CURRENTLY DISPLAYED configuration
    (after whatever radius cut / hull peels you've applied) to disk as
    four separate publication-ready files (triangulation-by-zi,
    triangulation-by-qi, histogram-of-zi, histogram-of-qi) plus a .csv
    of the raw per-particle numbers, so you can polish each figure
    separately for a paper. See make_paper_figures.py for a fully
    tunable, non-interactive template that reuses the same plotting
    functions with custom colormaps/ranges/fonts/sizes/formats.

Usage:
    python3 delaunay_topology.py name.dat

If no filename is given, it defaults to "name.dat" in the current
directory.

Requires: numpy, scipy, matplotlib
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import Delaunay, ConvexHull, QhullError


# ==========================================================================
# Core numerical routines
# ==========================================================================

def load_points(filename):
    """Load a two-column (x y) data file into an (N,2) array."""
    data = np.loadtxt(filename)
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError(f"'{filename}' must have at least two columns (x y).")
    return data[:, :2].astype(float)


def compute_topology(points):
    """
    Given an (N,2) array of points, compute:
      tri       : scipy.spatial.Delaunay object
      zi        : (N,) coordination number of each point
      qi        : (N,) topological charge of each point
      hull_mask : (N,) boolean, True if point is on the convex hull
    """
    n = len(points)
    tri = Delaunay(points)

    neighbors = [set() for _ in range(n)]
    for simplex in tri.simplices:
        for a, b in ((simplex[0], simplex[1]),
                     (simplex[1], simplex[2]),
                     (simplex[2], simplex[0])):
            neighbors[a].add(b)
            neighbors[b].add(a)

    zi = np.array([len(neighbors[i]) for i in range(n)], dtype=int)

    hull = ConvexHull(points)
    hull_mask = np.zeros(n, dtype=bool)
    hull_mask[hull.vertices] = True

    qi = np.where(hull_mask, 4 - zi, 6 - zi)
    return tri, zi, qi, hull_mask


def peel_convex_hull(points, indices):
    """Remove the current convex-hull vertices from the point set."""
    if len(points) < 4:
        return points, indices
    try:
        hull = ConvexHull(points)
    except QhullError:
        return points, indices
    mask = np.ones(len(points), dtype=bool)
    mask[hull.vertices] = False
    if mask.sum() < 3:
        # don't peel away so much that nothing triangulable remains
        return points, indices
    return points[mask], indices[mask]


def filter_by_radius(points, indices, rmax):
    """Keep only points within distance rmax of the center of mass."""
    com = points.mean(axis=0)
    d = np.linalg.norm(points - com, axis=1)
    mask = d <= rmax
    return points[mask], indices[mask]


# ==========================================================================
# Reusable, fully-tunable plotting functions
#
# Every keyword has a sensible default so the interactive GUI can call
# these with no fuss, but every knob a paper figure typically needs is
# exposed: colormap, color limits, marker/hull/COM styling, colorbar,
# title, font size, figure size (via the `ax`/`figsize` argument), and
# whether to draw each optional element at all. Call these directly from
# your own script (see make_paper_figures.py) to build one clean,
# separately-tunable file per figure.
# ==========================================================================

def shade_defect_triangles(ax, pts, simplices, qi, cmap="coolwarm",
                            vmin=-3, vmax=3, alpha=0.45, flat_color=None):
    """
    Fill every triangle that touches at least one 'defective' vertex
    (qi != 0, i.e. z_i != 6 in the bulk or z_i != 4 on the hull).

    By default the fill color reflects the sign/size of the defect
    (cmap/vmin/vmax), so a triangle spanning a positive- and a
    negative-charge vertex (the classic bound dislocation pair) shows
    up as adjacent red/blue wedges.

    If `flat_color` is given (e.g. "0.5" for mid-gray, or "gray",
    "black", "#888888"), every defect triangle is filled with that
    single flat color instead — useful when the dot colors already
    encode z_i/q_i and you just want to flag "a defect touches here"
    without a second color scale to read.
    """
    patches = []
    if flat_color is not None:
        for simplex in simplices:
            if np.any(qi[simplex] != 0):
                patches.append(Polygon(pts[simplex]))
        if patches:
            pc = PatchCollection(patches, facecolor=flat_color, edgecolor="none",
                                  alpha=alpha, zorder=0)
            ax.add_collection(pc)
        return

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)
    patches, colors = [], []
    for simplex in simplices:
        vqi = qi[simplex]
        if np.any(vqi != 0):
            patches.append(Polygon(pts[simplex]))
            colors.append(cmap_obj(norm(vqi.mean())))
    if patches:
        pc = PatchCollection(patches, facecolor=colors, edgecolor="none",
                              alpha=alpha, zorder=0)
        ax.add_collection(pc)


def plot_triangulation_panel(
        pts, tri, values, *,
        ax=None, figsize=(5, 5),
        cmap="viridis", vmin=None, vmax=None,
        marker_size=45, marker_edgecolor="k", marker_edgewidth=0.4,
        marker_mask=None,
        tri_color="0.6", tri_linewidth=0.6,
        shade_defects=False, shade_qi=None, shade_cmap="coolwarm",
        shade_vmin=-3, shade_vmax=3, shade_alpha=0.45, shade_flat_color=None,
        hull_mask=None, show_hull=True, hull_color="lime",
        hull_size=110, hull_linewidth=1.5, hull_label="convex hull",
        show_com=True, com=None, com_color="black", com_label="center of mass",
        colorbar=True, cbar_label=None, cbar_shrink=0.8,
        show_legend=False, legend_kwargs=None,
        title=None, fontsize=11, equal_aspect=True):
    """
    Draw one Delaunay-triangulation panel colored by `values` (pass zi
    or qi). Returns (fig, ax, scatter_handle) so you can further tweak
    or save the figure yourself.

    `marker_mask`: optional boolean array (same length as pts) selecting
    which particles get a dot marker drawn. The triangulation mesh (and
    hull markers, COM) are always drawn for the FULL point set regardless
    of this mask — only the per-particle dots are filtered. Use this to,
    e.g., hide markers for defect-free particles (q_i == 0) so only
    defective vertices are marked:  marker_mask = (qi != 0)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if shade_defects:
        if shade_qi is None:
            raise ValueError("shade_defects=True requires shade_qi (the qi array).")
        shade_defect_triangles(ax, pts, tri.simplices, shade_qi,
                                cmap=shade_cmap, vmin=shade_vmin,
                                vmax=shade_vmax, alpha=shade_alpha,
                                flat_color=shade_flat_color)

    ax.triplot(pts[:, 0], pts[:, 1], tri.simplices,
               color=tri_color, linewidth=tri_linewidth, zorder=1)

    values = np.asarray(values)
    if marker_mask is None:
        marker_mask = np.ones(len(pts), dtype=bool)
    else:
        marker_mask = np.asarray(marker_mask, dtype=bool)

    sc = ax.scatter(pts[marker_mask, 0], pts[marker_mask, 1],
                     c=values[marker_mask], cmap=cmap,
                     vmin=vmin, vmax=vmax, s=marker_size,
                     edgecolor=marker_edgecolor, linewidth=marker_edgewidth,
                     zorder=2)

    if show_hull and hull_mask is not None:
        ax.scatter(pts[hull_mask, 0], pts[hull_mask, 1],
                   facecolor="none", edgecolor=hull_color,
                   s=hull_size, linewidth=hull_linewidth, zorder=3,
                   label=hull_label if show_legend else None)

    if show_com:
        c = com if com is not None else pts.mean(axis=0)
        ax.plot(*c, marker="+", color=com_color, markersize=14, mew=2,
                zorder=3, label=com_label if show_legend else None)

    if colorbar:
        fig.colorbar(sc, ax=ax, shrink=cbar_shrink, label=cbar_label)

    if title:
        ax.set_title(title, fontsize=fontsize)

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    if show_legend:
        ax.legend(**(legend_kwargs or dict(loc="upper right", fontsize=8,
                                            framealpha=0.8)))

    return fig, ax, sc


def plot_histogram(values, *, ax=None, figsize=(5, 4), bins=None,
                    color="steelblue", edgecolor="k", log=False,
                    title=None, xlabel=None, ylabel="count", fontsize=11):
    """
    Draw one integer-valued histogram (use for zi or qi). Returns
    (fig, ax) for further tweaking/saving.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    values = np.asarray(values)
    if bins is None:
        vmin_, vmax_ = int(values.min()), int(values.max())
        bins = np.arange(vmin_ - 0.5, vmax_ + 1.5, 1)

    ax.hist(values, bins=bins, color=color, edgecolor=edgecolor)
    if log:
        ax.set_yscale("log")
    if title:
        ax.set_title(title, fontsize=fontsize)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)

    return fig, ax


def save_particle_data_csv(path, pts, zi, qi, hull_mask):
    """Dump per-particle x, y, z_i, q_i, on_hull to a CSV for reuse elsewhere."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z_i", "q_i", "on_hull"])
        for (x, y), z, q, h in zip(pts, zi, qi, hull_mask):
            writer.writerow([x, y, int(z), int(q), int(h)])


# ==========================================================================
# Interactive application
# ==========================================================================

class TopologyApp:
    def __init__(self, points, export_dir="paper_figures"):
        self.original = points.copy()
        self.active_points = points.copy()          # survives hull peels
        self.active_indices = np.arange(len(points))
        self.export_dir = export_dir
        self._export_counter = 0

        self.fig = plt.figure(figsize=(13, 10))
        gs = self.fig.add_gridspec(3, 2, height_ratios=[3, 1.1, 0.6],
                                    hspace=0.5, wspace=0.3,
                                    left=0.06, right=0.95, top=0.95, bottom=0.20)

        self.ax_z = self.fig.add_subplot(gs[0, 0])
        self.ax_main = self.fig.add_subplot(gs[0, 1])
        self.ax_hist_z = self.fig.add_subplot(gs[1, 0])
        self.ax_hist_q = self.fig.add_subplot(gs[1, 1])
        self.ax_info = self.fig.add_subplot(gs[2, :])
        self.ax_info.axis("off")

        # --- widgets -------------------------------------------------
        ax_slider = self.fig.add_axes([0.15, 0.10, 0.45, 0.03])
        r0 = self._max_radius(self.active_points)
        self.slider = Slider(ax_slider, "Radius cutoff", 0.0, r0, valinit=r0)
        self.slider.on_changed(self._on_slider)

        ax_peel = self.fig.add_axes([0.64, 0.09, 0.14, 0.05])
        self.btn_peel = Button(ax_peel, "Peel convex hull")
        self.btn_peel.on_clicked(self._on_peel)

        ax_reset = self.fig.add_axes([0.64, 0.02, 0.14, 0.05])
        self.btn_reset = Button(ax_reset, "Reset to original")
        self.btn_reset.on_clicked(self._on_reset)

        ax_export = self.fig.add_axes([0.80, 0.09, 0.14, 0.05])
        self.btn_export = Button(ax_export, "Export panels")
        self.btn_export.on_clicked(self._on_export)

        self.cbar_q = None
        self.cbar_z = None
        self.update()

    @staticmethod
    def _max_radius(points):
        com = points.mean(axis=0)
        return float(np.linalg.norm(points - com, axis=1).max()) if len(points) else 1.0

    def _current_view(self):
        """Points currently shown, after applying the radius slider to active_points."""
        rmax = self.slider.val
        pts, idx = filter_by_radius(self.active_points, self.active_indices, rmax)
        return pts, idx

    def _on_slider(self, _val):
        self.update()

    def _on_peel(self, _event):
        # Peel the hull of what is CURRENTLY DISPLAYED (radius-filtered view),
        # permanently removing those particles from the active set.
        pts, idx = self._current_view()
        new_pts, new_idx = peel_convex_hull(pts, idx)
        if len(new_pts) == len(pts):
            print("Nothing peeled (too few points left, or peel would empty the set).")
            return
        # Remove the peeled-away points from active_points/active_indices.
        removed = set(idx.tolist()) - set(new_idx.tolist())
        keep_mask = np.array([i not in removed for i in self.active_indices])
        self.active_points = self.active_points[keep_mask]
        self.active_indices = self.active_indices[keep_mask]

        # Reset slider range to match the new active set, keep it "showing all".
        new_rmax = self._max_radius(self.active_points)
        self.slider.ax.clear()
        self.slider.__init__(self.slider.ax, "Radius cutoff", 0.0, max(new_rmax, 1e-9),
                              valinit=new_rmax)
        self.slider.on_changed(self._on_slider)
        self.update()

    def _on_reset(self, _event):
        self.active_points = self.original.copy()
        self.active_indices = np.arange(len(self.original))
        new_rmax = self._max_radius(self.active_points)
        self.slider.ax.clear()
        self.slider.__init__(self.slider.ax, "Radius cutoff", 0.0, max(new_rmax, 1e-9),
                              valinit=new_rmax)
        self.slider.on_changed(self._on_slider)
        self.update()

    def _on_export(self, _event):
        """
        Save the CURRENT view as four separate publication-ready files
        (PDF, vector) plus a CSV of the underlying numbers, so each
        panel can be reopened and tuned independently for a paper.
        """
        pts, _idx = self._current_view()
        if len(pts) < 3:
            print("Nothing to export (need >= 3 points).")
            return
        try:
            tri, zi, qi, hull_mask = compute_topology(pts)
        except QhullError as e:
            print(f"Export failed: triangulation error ({e}).")
            return

        os.makedirs(self.export_dir, exist_ok=True)
        self._export_counter += 1
        tag = f"{self._export_counter:03d}_N{len(pts)}"

        # zi triangulation, with defect-triangle shading
        fig_z, ax_z, _ = plot_triangulation_panel(
            pts, tri, zi, cmap="viridis",
            vmin=min(int(zi.min()), 4), vmax=max(int(zi.max()), 8),
            shade_defects=True, shade_qi=qi,
            hull_mask=hull_mask, cbar_label="$z_i$",
            title=f"Delaunay triangulation, N={len(pts)}")
        p1 = os.path.join(self.export_dir, f"triangulation_zi_{tag}.pdf")
        fig_z.savefig(p1, bbox_inches="tight")
        plt.close(fig_z)

        # qi triangulation
        fig_q, ax_q, _ = plot_triangulation_panel(
            pts, tri, qi, cmap="coolwarm", vmin=-3, vmax=3,
            hull_mask=hull_mask, cbar_label="$q_i$", show_legend=True,
            title=f"Delaunay triangulation, N={len(pts)}")
        p2 = os.path.join(self.export_dir, f"triangulation_qi_{tag}.pdf")
        fig_q.savefig(p2, bbox_inches="tight")
        plt.close(fig_q)

        # histograms
        fig_hz, _ = plot_histogram(zi, color="steelblue",
                                    title="Coordination number $z_i$",
                                    xlabel="$z_i$")
        p3 = os.path.join(self.export_dir, f"hist_zi_{tag}.pdf")
        fig_hz.savefig(p3, bbox_inches="tight")
        plt.close(fig_hz)

        fig_hq, _ = plot_histogram(qi, color="indianred",
                                    title="Topological charge $q_i$",
                                    xlabel="$q_i$")
        p4 = os.path.join(self.export_dir, f"hist_qi_{tag}.pdf")
        fig_hq.savefig(p4, bbox_inches="tight")
        plt.close(fig_hq)

        # raw data
        p5 = os.path.join(self.export_dir, f"data_{tag}.csv")
        save_particle_data_csv(p5, pts, zi, qi, hull_mask)

        print(f"Exported {len(pts)}-particle view to '{self.export_dir}/':")
        for p in (p1, p2, p3, p4, p5):
            print(f"  {p}")
        print("Edit make_paper_figures.py (or call plot_triangulation_panel / "
              "plot_histogram directly) to re-style any of these individually.")

    def update(self):
        pts, _idx = self._current_view()

        self.ax_main.clear()
        self.ax_z.clear()
        self.ax_hist_z.clear()
        self.ax_hist_q.clear()
        self.ax_info.clear()
        self.ax_info.axis("off")

        n = len(pts)
        if n < 3:
            self.ax_main.set_title("Not enough points to triangulate (need >= 3)")
            self.ax_main.scatter(*pts.T, c="k") if n else None
            self.fig.canvas.draw_idle()
            return

        try:
            tri, zi, qi, hull_mask = compute_topology(pts)
        except QhullError as e:
            self.ax_main.set_title(f"Triangulation failed: {e}")
            self.fig.canvas.draw_idle()
            return

        # --- left plot: Delaunay triangulation colored by zi -----------
        zmin_d, zmax_d = int(zi.min()), int(zi.max())
        _, _, sc_z = plot_triangulation_panel(
            pts, tri, zi, ax=self.ax_z, cmap="viridis",
            vmin=min(zmin_d, 4), vmax=max(zmax_d, 8),
            shade_defects=True, shade_qi=qi,
            hull_mask=hull_mask, colorbar=False,
            title=(f"Delaunay triangulation (N={n}) — color = $z_i$\n"
                   f"shaded triangles touch a defective vertex ($q_i \\neq 0$)"),
            fontsize=10)

        if self.cbar_z is None:
            self.cbar_z = self.fig.colorbar(sc_z, ax=self.ax_z, shrink=0.8, label="$z_i$")
        else:
            self.cbar_z.update_normal(sc_z)

        # --- right plot: Delaunay triangulation colored by qi ----------
        _, _, sc_q = plot_triangulation_panel(
            pts, tri, qi, ax=self.ax_main, cmap="coolwarm", vmin=-3, vmax=3,
            hull_mask=hull_mask, colorbar=False, show_legend=True,
            title=f"Delaunay triangulation (N={n}) — color = $q_i$")

        if self.cbar_q is None:
            self.cbar_q = self.fig.colorbar(sc_q, ax=self.ax_main, shrink=0.8, label="$q_i$")
        else:
            self.cbar_q.update_normal(sc_q)

        # --- histograms -------------------------------------------------
        plot_histogram(zi, ax=self.ax_hist_z, color="steelblue",
                       title="Coordination number $z_i$", xlabel="$z_i$")
        plot_histogram(qi, ax=self.ax_hist_q, color="indianred",
                       title="Topological charge $q_i$", xlabel="$q_i$")

        # --- info panel ---------------------------------------------
        total_q = int(qi.sum())
        n_hull = int(hull_mask.sum())
        info = (f"N particles shown: {n}   |   On convex hull: {n_hull}   |   "
                f"Interior: {n - n_hull}   |   "
                f"mean $z_i$ = {zi.mean():.3f}   |   "
                f"TOTAL TOPOLOGICAL CHARGE  $\\sum_i q_i$ = {total_q}")
        self.ax_info.text(0.5, 0.5, info, ha="center", va="center", fontsize=12,
                           transform=self.ax_info.transAxes,
                           bbox=dict(boxstyle="round", facecolor="lightyellow"))

        print(f"[N={n}] total topological charge sum(q_i) = {total_q}")

        self.fig.canvas.draw_idle()


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "name.dat"
    print(f"Loading points from '{filename}' ...")
    points = load_points(filename)
    print(f"Loaded {len(points)} particles.")

    app = TopologyApp(points)
    plt.show()


if __name__ == "__main__":
    main()
