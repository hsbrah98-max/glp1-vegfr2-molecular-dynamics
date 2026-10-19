# GLP-1RA & VEGFR2 100ns Molecular Dynamics & MM-PBSA Trajectory Analysis

**Author:** Harman Brah, MD

Complete 100-nanosecond all-atom GROMACS molecular dynamics simulation trajectory metrics, AMBER gmx_MMPBSA binding free energy decomposition scripts, and PyMOL 3D structural models evaluating GLP-1 receptor agonists (Lixisenatide, Tirzepatide, Taspoglutide, Efpeglenatide) against Vascular Endothelial Growth Factor Receptor 2 (VEGFR2).

## Scientific Background

In wet age-related macular degeneration (wet AMD) and diabetic retinopathy, aberrant pathological angiogenesis is driven by VEGF-A/VEGFR2 hyperactivation. This project investigates the structural biophysics and competitive binding potential of peptide GLP-1 receptor agonists against VEGFR2, evaluating whether GLP-1 analogues can stabilize the receptor binding pocket and modulate angiogenic signaling.

## Repository Structure

- `scripts/analyze_trajectories.py`: Standalone Python script to parse GROMACS `.xvg` trajectory files and compute RMSD, per-residue RMSF, peptide radius of gyration, and hydrogen bonding networks.
- `scripts/plot_mmpbsa.py`: GMX_MMPBSA visualization suite computing binding free energy convergence and per-residue energy decompositions.
- `scripts/lixi_gb_decomp.in`: AMBER MM-PBSA generalized Born (GB) input deck specifying salt concentration and per-residue decomposition parameters.
- `data/`: Extracted GROMACS trajectory metrics (`.xvg`) and MM-PBSA binding free energy outputs (`FINAL_RESULTS_MMPBSA.csv`).
- `figures/100ns_trajectory_metrics.png`: Four-panel plot demonstrating structural stability over the 100ns simulation.
- `HARMAN-LIX-GLP.pse`: Interactive PyMOL 3D structural session displaying the docked Lixisenatide-VEGFR2 binding interface.

## Quick Start

```bash
# Generate trajectory stability metrics and four-panel figure
python3 scripts/analyze_trajectories.py
```
