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
  - Shows histograms of z_i and q_i.
  - Prints (and displays) the total topological charge sum_i q_i.

Usage:
    python3 delaunay_topology.py name.dat

If no filename is given, it defaults to "name.dat" in the current
directory.

Requires: numpy, scipy, matplotlib
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.spatial import Delaunay, ConvexHull, QhullError


# --------------------------------------------------------------------------
# Core numerical routines
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Interactive application
# --------------------------------------------------------------------------

class TopologyApp:
    def __init__(self, points):
        self.original = points.copy()
        self.active_points = points.copy()          # survives hull peels
        self.active_indices = np.arange(len(points))

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
        ax_slider = self.fig.add_axes([0.15, 0.10, 0.55, 0.03])
        r0 = self._max_radius(self.active_points)
        self.slider = Slider(ax_slider, "Radius cutoff", 0.0, r0, valinit=r0)
        self.slider.on_changed(self._on_slider)

        ax_peel = self.fig.add_axes([0.76, 0.09, 0.18, 0.05])
        self.btn_peel = Button(ax_peel, "Peel convex hull")
        self.btn_peel.on_clicked(self._on_peel)

        ax_reset = self.fig.add_axes([0.76, 0.02, 0.18, 0.05])
        self.btn_reset = Button(ax_reset, "Reset to original")
        self.btn_reset.on_clicked(self._on_reset)

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

        com = pts.mean(axis=0)

        # --- left plot: Delaunay triangulation colored by zi -----------
        self.ax_z.triplot(pts[:, 0], pts[:, 1], tri.simplices,
                           color="0.6", linewidth=0.6, zorder=1)
        zmin_d, zmax_d = int(zi.min()), int(zi.max())
        sc_z = self.ax_z.scatter(pts[:, 0], pts[:, 1], c=zi, cmap="viridis",
                                  vmin=min(zmin_d, 4), vmax=max(zmax_d, 8),
                                  s=45, edgecolor="k", linewidth=0.4, zorder=2)
        self.ax_z.scatter(pts[hull_mask, 0], pts[hull_mask, 1],
                           facecolor="none", edgecolor="lime",
                           s=110, linewidth=1.5, zorder=3)
        self.ax_z.plot(*com, marker="+", color="black", markersize=14, mew=2, zorder=3)
        self.ax_z.set_title(f"Delaunay triangulation (N={n}) — color = $z_i$")
        self.ax_z.set_aspect("equal", adjustable="box")

        if self.cbar_z is None:
            self.cbar_z = self.fig.colorbar(sc_z, ax=self.ax_z, shrink=0.8, label="$z_i$")
        else:
            self.cbar_z.update_normal(sc_z)

        # --- right plot: Delaunay triangulation colored by qi ----------
        self.ax_main.triplot(pts[:, 0], pts[:, 1], tri.simplices,
                              color="0.6", linewidth=0.6, zorder=1)
        sc = self.ax_main.scatter(pts[:, 0], pts[:, 1], c=qi, cmap="coolwarm",
                                   vmin=-3, vmax=3, s=45, edgecolor="k",
                                   linewidth=0.4, zorder=2)
        self.ax_main.scatter(pts[hull_mask, 0], pts[hull_mask, 1],
                              facecolor="none", edgecolor="lime",
                              s=110, linewidth=1.5, zorder=3, label="convex hull")
        self.ax_main.plot(*com, marker="+", color="black", markersize=14,
                           mew=2, zorder=3, label="center of mass")
        self.ax_main.set_title(f"Delaunay triangulation (N={n}) — color = $q_i$")
        self.ax_main.set_aspect("equal", adjustable="box")
        self.ax_main.legend(loc="upper right", fontsize=8, framealpha=0.8)

        if self.cbar_q is None:
            self.cbar_q = self.fig.colorbar(sc, ax=self.ax_main, shrink=0.8, label="$q_i$")
        else:
            self.cbar_q.update_normal(sc)

        # --- histograms -------------------------------------------------
        zmin, zmax = int(zi.min()), int(zi.max())
        bins_z = np.arange(zmin - 0.5, zmax + 1.5, 1)
        self.ax_hist_z.hist(zi, bins=bins_z, color="steelblue", edgecolor="k")
        self.ax_hist_z.set_title("Coordination number $z_i$")
        self.ax_hist_z.set_xlabel("$z_i$")
        self.ax_hist_z.set_ylabel("count")

        qmin, qmax = int(qi.min()), int(qi.max())
        bins_q = np.arange(qmin - 0.5, qmax + 1.5, 1)
        self.ax_hist_q.hist(qi, bins=bins_q, color="indianred", edgecolor="k")
        self.ax_hist_q.set_title("Topological charge $q_i$")
        self.ax_hist_q.set_xlabel("$q_i$")

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
