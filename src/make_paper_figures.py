#!/usr/bin/env python3
"""
make_paper_figures.py

A NON-interactive template for producing publication-quality figures
from a particle configuration, with every panel tuned separately.

This script reuses the exact same computation and plotting functions as
delaunay_topology.py (loaded via `import`), so results match what you
saw in the interactive GUI — but here every knob is a plain Python
variable you can edit directly: which points to keep (radius cut,
number of hull peels), colormap and color limits, marker/hull/COM
styling, figure size, font size, output format (pdf/png/svg/eps), dpi.

Workflow:
  1. Edit the "CONFIGURE" section below to match what you want.
  2. Run:  python3 make_paper_figures.py
  3. Each panel is saved as its own file in OUTPUT_DIR. Tweak the
     per-panel dict for that panel and re-run — the others are
     untouched.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from delaunay_topology import (
    load_points, compute_topology, peel_convex_hull, filter_by_radius,
    plot_triangulation_panel, plot_histogram, save_particle_data_csv,
    save_histogram_csv,
)

# ==========================================================================
# COMMAND LINE
#
#   python3 make_paper_figures.py name.dat
#   python3 make_paper_figures.py name.dat --radius 8 --peels 2 --outdir figs
#
# Any of these override the CONFIGURE defaults below. Everything else
# (colors, sizes, fonts, formats) is still set by editing this file.
# ==========================================================================

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("datafile", nargs="?", default=None,
                    help="Path to the two-column (x y) data file "
                         "(default: 'name.dat', or the CONFIGURE value below)")
    p.add_argument("--radius", type=float, default=None,
                    help="Radius cutoff from the center of mass "
                         "(overrides RADIUS_CUTOFF below)")
    p.add_argument("--peels", type=int, default=None,
                    help="Number of convex-hull layers to peel "
                         "(overrides N_HULL_PEELS below)")
    p.add_argument("--outdir", type=str, default=None,
                    help="Output directory (overrides OUTPUT_DIR below)")
    return p.parse_args()


# ==========================================================================
# CONFIGURE
#
# These are the defaults used when the corresponding command-line option
# is not given.
# ==========================================================================

DATA_FILE = "name.dat"
OUTPUT_DIR = "paper_figures"

# --- Selection: which particles to actually analyze ----------------------
RADIUS_CUTOFF = None      # e.g. 8.0 to keep only particles within 8 units
                           # of the center of mass; None = keep all
N_HULL_PEELS = 0           # how many times to strip the outer convex-hull
                           # layer before plotting (0 = don't peel)

# --- Global figure style ---------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",       # comment out if you don't have a serif font
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
})
OUTPUT_FORMAT = "pdf"      # "pdf", "svg", "eps", or "png"

# --- Panel 1: triangulation colored by z_i --------------------------------
PANEL_ZI = dict(
    figsize=(5, 5),
    cmap="viridis",
    vmin=4, vmax=8,             # fix these so multiple figures share a scale
    marker_size=30, #55,        # (edited moira)
    marker_edgecolor="k",
    marker_edgewidth=0.5,
    tri_color="0.65",
    tri_linewidth=0.5,
    shade_defects=True,          # highlight triangles touching a defect
    shade_cmap="coolwarm",       # used only if shade_flat_color is None
    shade_vmin=-3, shade_vmax=3,
    shade_alpha=0.40,
    shade_flat_color="0.5", #None,       # e.g. "0.5" for flat gray shading instead, (moira edited)
                                  # of color-by-charge; overrides shade_cmap
    show_hull=False, #True (after moira)
    hull_color="lime",
    show_com=True,
    colorbar=True,
    cbar_label=r"$z_i$",
    title=None,                  # set to a string, or leave None for no title
)

# --- Panel 2: triangulation colored by q_i --------------------------------
PANEL_QI = dict(
    figsize=(5, 5),
    cmap="coolwarm",
    vmin=-3, vmax=3,
    marker_size=25, #55,
    marker_edgecolor="k",
    marker_edgewidth=0.5,
    tri_color="0.65",
    tri_linewidth=0.5,
    shade_defects=False,
    show_hull=False, #True, (after Moira)
    hull_color="lime",
    show_com=True,
    colorbar=True,
    cbar_label=r"$q_i$",
    show_legend=True,
    title=None,
)

# --- Panel 3: histogram of z_i --------------------------------------------
HIST_ZI = dict(
    figsize=(4.5, 3.5),
    color="steelblue",
    edgecolor="k",
    log=False,
    xlabel=r"$z_i$",
    ylabel="count",
    title=None,
)

# --- Panel 4: histogram of q_i --------------------------------------------
HIST_QI = dict(
    figsize=(4.5, 3.5),
    color="indianred",
    edgecolor="k",
    log=False,
    xlabel=r"$q_i$",
    ylabel="count",
    title=None,
)

# ==========================================================================
# RUN (usually no need to edit below this line)
# ==========================================================================

def main():
    args = _parse_args()
    data_file = args.datafile or DATA_FILE
    radius_cutoff = args.radius if args.radius is not None else RADIUS_CUTOFF
    n_hull_peels = args.peels if args.peels is not None else N_HULL_PEELS
    output_dir = args.outdir or OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    points = load_points(data_file)
    indices = np.arange(len(points))
    print(f"Loaded {len(points)} particles from '{data_file}'.")

    if radius_cutoff is not None:
        points, indices = filter_by_radius(points, indices, radius_cutoff)
        print(f"After radius cutoff {radius_cutoff}: {len(points)} particles.")

    for k in range(n_hull_peels):
        new_points, new_indices = peel_convex_hull(points, indices)
        if len(new_points) == len(points):
            print(f"Stopped peeling after {k} layer(s): nothing left to peel.")
            break
        points, indices = new_points, new_indices
    if n_hull_peels > 0:
        print(f"After {n_hull_peels} hull peel(s): {len(points)} particles.")

    tri, zi, qi, hull_mask = compute_topology(points)
    total_q = int(qi.sum())
    print(f"N = {len(points)}  |  on hull = {int(hull_mask.sum())}  |  "
          f"mean z_i = {zi.mean():.3f}  |  TOTAL TOPOLOGICAL CHARGE = {total_q}")

    # --- Panel 1: z_i triangulation ---------------------------------
    fig1, ax1, _ = plot_triangulation_panel(
        points, tri, zi, hull_mask=hull_mask, shade_qi=qi, 
        marker_mask=(zi != 6),   # only draw dots for defective vertices 
        **PANEL_ZI)

    ax1.axis("off") # Moira edited

    out1 = os.path.join(output_dir, f"triangulation_zi.{OUTPUT_FORMAT}")
    fig1.savefig(out1, bbox_inches="tight")
    print(f"Saved {out1}")

    # --- Panel 2: q_i triangulation ---------------------------------
    fig2, ax2, _ = plot_triangulation_panel(
        points, tri, qi, hull_mask=hull_mask, 
        marker_mask=(qi != 0),   # only draw dots for defective vertices
        **PANEL_QI)

    ax2.axis("off") # Moira edited

    out2 = os.path.join(output_dir, f"triangulation_qi.{OUTPUT_FORMAT}")
    fig2.savefig(out2, bbox_inches="tight")
    print(f"Saved {out2}")



    # --- Panel 3: z_i histogram --------------------------------------
    fig3, ax3 = plot_histogram(zi, **HIST_ZI)
    out3 = os.path.join(output_dir, f"hist_zi.{OUTPUT_FORMAT}")
    fig3.savefig(out3, bbox_inches="tight")
    print(f"Saved {out3}")

    # --- Panel 4: q_i histogram --------------------------------------
    fig4, ax4 = plot_histogram(qi, **HIST_QI)
    out4 = os.path.join(output_dir, f"hist_qi.{OUTPUT_FORMAT}")
    fig4.savefig(out4, bbox_inches="tight")
    print(f"Saved {out4}")

    # --- Raw numbers, for a table or your own re-plotting ------------
    out5 = os.path.join(output_dir, "data.csv")
    save_particle_data_csv(out5, points, zi, qi, hull_mask)
    print(f"Saved {out5}")

    # --- Histogram bin counts (value, count) --------------------------
    out6 = os.path.join(output_dir, "hist_zi.csv")
    save_histogram_csv(out6, zi, value_label="z_i")
    print(f"Saved {out6}")

    out7 = os.path.join(output_dir, "hist_qi.csv")
    save_histogram_csv(out7, qi, value_label="q_i")
    print(f"Saved {out7}")

    #plt.show()  # comment out if running non-interactively / on a cluster


if __name__ == "__main__":
    main()
