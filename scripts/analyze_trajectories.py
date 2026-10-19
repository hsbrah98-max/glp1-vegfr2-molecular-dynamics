#!/usr/bin/env python3
"""
Trajectory Analysis for 100ns GLP-1RA & VEGFR2 GROMACS Molecular Dynamics.
Analyzes:
1. Root Mean Square Deviation (RMSD): Protein backbone and complex stability.
2. Root Mean Square Fluctuation (RMSF): Per-residue flexibility.
3. Radius of Gyration (Rg): Complex compactness over 100ns simulation.
4. Intermolecular Hydrogen Bonding (HbNum): Hydrogen bond network stability.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_xvg(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None, None
    x, y = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("@") or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x.append(float(parts[0]))
                    y.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(x), np.array(y)


def analyze_and_plot():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # 1. RMSD
    t_rmsd, rmsd = load_xvg("lixi100_rmsd_protein.xvg")
    if t_rmsd is not None:
        axes[0, 0].plot(t_rmsd / 1000.0, rmsd * 10.0, color="#1F4E78", linewidth=1.2)
        axes[0, 0].set_title("A. Backbone RMSD (Stability over 100ns)", fontsize=11, fontweight="bold")
        axes[0, 0].set_xlabel("Time (ns)", fontsize=9.5)
        axes[0, 0].set_ylabel("RMSD (Å)", fontsize=9.5)
        axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # 2. RMSF
    res_idx, rmsf = load_xvg("lixi100_rmsf.xvg")
    if res_idx is not None:
        axes[0, 1].plot(res_idx, rmsf * 10.0, color="#B22222", linewidth=1.2)
        axes[0, 1].set_title("B. Per-Residue Fluctuation (RMSF)", fontsize=11, fontweight="bold")
        axes[0, 1].set_xlabel("Residue Index", fontsize=9.5)
        axes[0, 1].set_ylabel("RMSF (Å)", fontsize=9.5)
        axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # 3. Radius of Gyration
    t_gyr, gyr = load_xvg("lixi100_gyrate_peptide.xvg")
    if t_gyr is not None:
        axes[1, 0].plot(t_gyr / 1000.0, gyr * 10.0, color="#2E8B57", linewidth=1.2)
        axes[1, 0].set_title("C. Peptide Compactness (Radius of Gyration)", fontsize=11, fontweight="bold")
        axes[1, 0].set_xlabel("Time (ns)", fontsize=9.5)
        axes[1, 0].set_ylabel("Rg (Å)", fontsize=9.5)
        axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    # 4. Hydrogen Bonds
    t_hb, hb = load_xvg("lixi100_hbnum.xvg")
    if t_hb is not None:
        axes[1, 1].plot(t_hb / 1000.0, hb, color="#8E44AD", linewidth=1.0, alpha=0.8)
        axes[1, 1].set_title("D. Intermolecular Hydrogen Bonds", fontsize=11, fontweight="bold")
        axes[1, 1].set_xlabel("Time (ns)", fontsize=9.5)
        axes[1, 1].set_ylabel("Hydrogen Bond Count", fontsize=9.5)
        axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("100ns GROMACS Molecular Dynamics: Lixisenatide-VEGFR2 Complex", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "100ns_trajectory_metrics.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")


if __name__ == "__main__":
    analyze_and_plot()
