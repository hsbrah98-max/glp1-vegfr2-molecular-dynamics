#!/usr/bin/env python3
"""
Plot gmx_MMPBSA (GB) results: summary energy terms, per-frame convergence,
and per-residue decomposition.

Usage
-----
    python plot_mmpbsa.py                       # auto-detect files in cwd
    python plot_mmpbsa.py --summary FINAL_RESULTS_MMPBSA.dat \
                          --frames  FINAL_RESULTS_MMPBSA.csv \
                          --decomp  FINAL_DECOMP_MMPBSA.dat \
                          --outdir  figures --top 15

Requires: numpy, pandas, matplotlib
"""

import argparse
import glob
import io
import os
import re
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# style
# --------------------------------------------------------------------------
mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
})

FAV = "#2c6fbb"      # favourable
UNFAV = "#c0392b"    # unfavourable
NEUTRAL = "#7f8c8d"

TERM_COLORS = {
    "van der Waals": "#2c6fbb",
    "Electrostatic": "#e67e22",
    "Polar Solvation": "#16a085",
    "Non-Polar Solv.": "#8e44ad",
    "Internal": "#95a5a6",
    "Electrostatic + Polar solv.": "#e67e22",
}


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def read_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read().replace("\r\n", "\n").replace("\r", "\n")


def parse_summary(path):
    """FINAL_RESULTS_MMPBSA.dat -> {block_name: DataFrame(index=term)}.

    Blocks are 'Complex', 'Receptor', 'Ligand', 'Delta'.
    Columns: Average, SD(Prop.), SD, SEM(Prop.), SEM
    """
    text = read_text(path)
    blocks, current, rows = {}, None, []

    header_re = re.compile(r"^(Complex|Receptor|Ligand|Delta)\b.*:\s*$")
    # term name, then 5 floats
    row_re = re.compile(
        r"^(\S(?:[^\d\n]*\S)?)\s+"
        r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$"
    )

    def flush():
        if current and rows:
            df = pd.DataFrame(
                rows,
                columns=["Term", "Average", "SD(Prop.)", "SD", "SEM(Prop.)", "SEM"],
            ).set_index("Term")
            blocks[current] = df

    for line in text.split("\n"):
        line = line.rstrip()
        if line.startswith("|"):
            continue
        m = header_re.match(line.strip())
        if m:
            flush()
            current, rows = m.group(1), []
            continue
        m = row_re.match(line.strip())
        if m and current:
            term = m.group(1).lstrip("\u0394").strip()  # strip leading delta
            rows.append([term] + [float(x) for x in m.groups()[1:]])
    flush()

    if not blocks:
        raise ValueError(f"No energy blocks parsed from {path}")
    return blocks


def parse_frames(path):
    """FINAL_RESULTS_MMPBSA.csv -> {block_name: DataFrame(index='Frame #')}."""
    text = read_text(path)
    lines = text.split("\n")

    starts = []
    for i, line in enumerate(lines):
        m = re.match(r"^(Complex|Receptor|Ligand|Delta) Energy Terms\s*$", line.strip())
        if m:
            starts.append((i, m.group(1)))

    blocks = {}
    for k, (idx, name) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        chunk = [ln for ln in lines[idx + 1:end] if ln.strip()]
        if not chunk:
            continue
        df = pd.read_csv(io.StringIO("\n".join(chunk)))
        df = df.set_index(df.columns[0])
        blocks[name] = df.apply(pd.to_numeric, errors="coerce")

    if not blocks:
        raise ValueError(f"No per-frame blocks parsed from {path}")
    return blocks


def parse_decomp(path):
    """FINAL_DECOMP_MMPBSA.dat -> {(block, table): DataFrame}.

    block  in {'Complex','Receptor','Ligand','DELTAS'}
    table  in {'Total','Sidechain','Backbone'}
    Columns are a MultiIndex of (term, stat) with stat in {Avg,SD,SEM}.
    """
    text = read_text(path)
    terms = ["Internal", "van der Waals", "Electrostatic",
             "Polar Solvation", "Non-Polar Solv.", "TOTAL"]
    stats = ["Avg", "SD", "SEM"]
    cols = pd.MultiIndex.from_product([terms, stats])

    tables, block, table, rows = {}, None, None, []

    def flush():
        if block and table and rows:
            idx = [r[0] for r in rows]
            data = np.array([r[1:] for r in rows], dtype=float)
            df = pd.DataFrame(data, index=idx, columns=cols)
            df.index.name = "Residue"
            tables[(block, table)] = df

    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("|") or not s:
            continue
        m = re.match(r"^(Complex|Receptor|Ligand|DELTAS)\s*:?\s*$", s)
        if m:
            flush()
            block, table, rows = m.group(1), None, []
            continue
        m = re.match(r"^(Total|Sidechain|Backbone) Energy Decomposition:", s)
        if m:
            flush()
            table, rows = m.group(1), []
            continue
        if re.match(r"^[LR]:", s):
            parts = s.split(",")
            if len(parts) >= 19:
                try:
                    rows.append([parts[0]] + [float(x) for x in parts[1:19]])
                except ValueError:
                    pass
    flush()

    if not tables:
        raise ValueError(f"No decomposition tables parsed from {path}")
    return tables


def residue_label(res_id):
    """'L:A:GLU:3' -> ('GLU3', 'L')."""
    parts = res_id.split(":")
    if len(parts) == 4:
        return f"{parts[2].title()}{parts[3]}", parts[0]
    return res_id, "?"


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def fig_energy_terms(summary, outdir):
    """Bar chart of the binding free-energy decomposition (Delta block)."""
    if "Delta" not in summary:
        return None
    df = summary["Delta"]
    wanted = ["VDWAALS", "EEL", "EGB", "ESURF", "GGAS", "GSOLV", "TOTAL"]
    labels = {
        "VDWAALS": r"$\Delta E_{vdW}$", "EEL": r"$\Delta E_{elec}$",
        "EGB": r"$\Delta G_{polar}$", "ESURF": r"$\Delta G_{nonpolar}$",
        "GGAS": r"$\Delta G_{gas}$", "GSOLV": r"$\Delta G_{solv}$",
        "TOTAL": r"$\Delta G_{bind}$",
    }
    terms = [t for t in wanted if t in df.index]
    vals = df.loc[terms, "Average"].values
    errs = df.loc[terms, "SEM"].values

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [NEUTRAL if t == "TOTAL" else (FAV if v < 0 else UNFAV)
              for t, v in zip(terms, vals)]
    bars = ax.bar([labels.get(t, t) for t in terms], vals, yerr=errs,
                  color=colors, capsize=4, width=0.65,
                  error_kw=dict(ecolor="#34495e", lw=1.2))
    if "TOTAL" in terms:
        bars[terms.index("TOTAL")].set_edgecolor("black")
        bars[terms.index("TOTAL")].set_linewidth(1.4)

    span = max(abs(vals.min()), abs(vals.max()))
    for b, v, e in zip(bars, vals, errs):
        off = 0.04 * span
        ax.text(b.get_x() + b.get_width() / 2,
                v + (off + e if v >= 0 else -off - e),
                f"{v:.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)

    ax.axhline(0, color="black", lw=0.9)
    ax.set_ylabel("Energy (kcal/mol)")
    ax.set_title("MM/GBSA binding free-energy components")
    ax.margins(y=0.18)
    ax.grid(axis="x", visible=False)

    path = os.path.join(outdir, "01_energy_components.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_convergence(frames, outdir, term="TOTAL"):
    """Per-frame Delta energy + cumulative mean + histogram."""
    if "Delta" not in frames or term not in frames["Delta"].columns:
        return None
    y = frames["Delta"][term].dropna()
    x = y.index.values.astype(float)
    v = y.values
    cum = np.cumsum(v) / np.arange(1, len(v) + 1)
    win = max(3, len(v) // 20)
    roll = pd.Series(v).rolling(win, center=True, min_periods=1).mean().values

    fig, (ax, axh) = plt.subplots(
        1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [3, 1]}, sharey=True)

    ax.plot(x, v, color="#95a5a6", lw=0.8, alpha=0.8, label="per frame")
    ax.plot(x, roll, color=FAV, lw=1.8, label=f"rolling mean ({win} frames)")
    ax.plot(x, cum, color=UNFAV, lw=1.8, ls="--", label="cumulative mean")
    ax.axhline(v.mean(), color="black", lw=0.9, ls=":",
               label=f"mean = {v.mean():.2f}")
    ax.fill_between(x, v.mean() - v.std(ddof=1), v.mean() + v.std(ddof=1),
                    color="black", alpha=0.06)
    ax.set_xlabel("Frame")
    ax.set_ylabel(r"$\Delta G_{bind}$ (kcal/mol)")
    ax.set_title(f"Convergence of $\\Delta${term}")
    ax.legend(loc="best", fontsize=9)

    axh.hist(v, bins=25, orientation="horizontal",
             color=FAV, alpha=0.75, edgecolor="white")
    axh.axhline(v.mean(), color="black", lw=1.0, ls=":")
    axh.set_xlabel("Count")
    axh.set_title(f"SD = {v.std(ddof=1):.2f}")
    axh.grid(axis="y", visible=False)

    fig.tight_layout()
    path = os.path.join(outdir, "02_convergence.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_term_traces(frames, outdir):
    """Per-frame traces of the individual Delta components."""
    if "Delta" not in frames:
        return None
    df = frames["Delta"]
    wanted = [c for c in ["VDWAALS", "EEL", "EGB", "ESURF", "TOTAL"]
              if c in df.columns]
    if not wanted:
        return None

    fig, axes = plt.subplots(len(wanted), 1, figsize=(9, 1.9 * len(wanted)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, term in zip(axes, wanted):
        v = df[term].values
        ax.plot(df.index.values, v, lw=0.9, color=FAV if v.mean() < 0 else UNFAV)
        ax.axhline(v.mean(), color="black", lw=0.8, ls=":")
        ax.set_ylabel(term, fontsize=9)
        ax.text(0.995, 0.9, f"{v.mean():.2f} ± {v.std(ddof=1):.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color="#34495e")
    axes[-1].set_xlabel("Frame")
    axes[0].set_title("Per-frame energy components (kcal/mol)")
    fig.tight_layout()

    path = os.path.join(outdir, "03_component_traces.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_per_residue(decomp, outdir, top=15, table="Total"):
    """Horizontal bars of the strongest per-residue contributions (DELTAS)."""
    key = ("DELTAS", table)
    if key not in decomp:
        return None
    df = decomp[key]
    tot = df[("TOTAL", "Avg")]
    err = df[("TOTAL", "SEM")]

    order = tot.abs().sort_values(ascending=False).index[:top]
    order = tot.loc[order].sort_values().index          # most favourable first

    labels = [residue_label(r)[0] for r in order]
    chains = [residue_label(r)[1] for r in order]
    vals = tot.loc[order].values
    errs = err.loc[order].values

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(order) + 2.0))
    ypos = np.arange(len(order))
    colors = [FAV if v < 0 else UNFAV for v in vals]
    ax.barh(ypos, vals, xerr=errs, color=colors, height=0.68,
            capsize=3, error_kw=dict(ecolor="#34495e", lw=1.0))
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{l} ({c})" for l, c in zip(labels, chains)],
                       fontsize=9)
    ax.axvline(0, color="black", lw=0.9)
    ax.set_xlabel("Contribution to $\\Delta G_{bind}$ (kcal/mol)")
    ax.set_title(f"Top {len(order)} residue contributions ({table.lower()})")
    ax.grid(axis="y", visible=False)

    span = max(abs(vals.min()), abs(vals.max()))
    for y, v, e in zip(ypos, vals, errs):
        off = 0.03 * span + e
        ax.text(v + (off if v >= 0 else -off), y, f"{v:.1f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.margins(x=0.16)

    path = os.path.join(outdir, "04_per_residue.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_residue_breakdown(decomp, outdir, top=15, table="Total", raw_terms=False):
    """Stacked term-by-term breakdown for the strongest residues.

    By default Electrostatic and Polar Solvation are combined into a single net
    term. They are large and opposite for charged residues, so plotting them
    separately compresses every other residue into an unreadable sliver.
    """
    key = ("DELTAS", table)
    if key not in decomp:
        return None
    df = decomp[key]
    tot = df[("TOTAL", "Avg")]
    order = tot.abs().sort_values(ascending=False).index[:top]
    order = tot.loc[order].sort_values().index

    if raw_terms:
        series = {t: df.loc[order, (t, "Avg")].values
                  for t in ["van der Waals", "Electrostatic",
                            "Polar Solvation", "Non-Polar Solv."]}
    else:
        series = {
            "van der Waals": df.loc[order, ("van der Waals", "Avg")].values,
            "Electrostatic + Polar solv.":
                df.loc[order, ("Electrostatic", "Avg")].values
                + df.loc[order, ("Polar Solvation", "Avg")].values,
            "Non-Polar Solv.": df.loc[order, ("Non-Polar Solv.", "Avg")].values,
        }
    internal = df.loc[order, ("Internal", "Avg")].values
    if np.abs(internal).max() > 1e-6:
        series["Internal"] = internal

    labels = [residue_label(r)[0] for r in order]
    ypos = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(order) + 2.0))
    left_pos = np.zeros(len(order))
    left_neg = np.zeros(len(order))
    for term, v in series.items():
        base = np.where(v >= 0, left_pos, left_neg)
        ax.barh(ypos, v, left=base, height=0.68,
                color=TERM_COLORS.get(term, NEUTRAL), label=term,
                edgecolor="white", lw=0.4)
        left_pos = left_pos + np.clip(v, 0, None)
        left_neg = left_neg + np.clip(v, None, 0)

    ax.plot(tot.loc[order].values, ypos, "o", color="black", ms=4.5,
            label="net total", zorder=5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", lw=0.9)
    ax.set_xlabel("Energy (kcal/mol)")
    ax.set_title(f"Per-residue energy breakdown ({table.lower()})", pad=26)
    ax.legend(ncol=4, fontsize=8, loc="lower center",
              bbox_to_anchor=(0.5, 1.02))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()

    path = os.path.join(outdir, "05_residue_breakdown.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_residue_map(decomp, outdir, table="Total", cutoff=1.0):
    """Contribution vs. residue index, split by chain, labelling hotspots."""
    key = ("DELTAS", table)
    if key not in decomp:
        return None
    df = decomp[key]
    tot = df[("TOTAL", "Avg")]

    parsed = []
    for res in df.index:
        p = res.split(":")
        if len(p) == 4 and p[3].isdigit():
            parsed.append((p[0], p[2], int(p[3]), tot.loc[res]))
    if not parsed:
        return None
    pdf = pd.DataFrame(parsed, columns=["chain", "resname", "resid", "value"])

    chains = sorted(pdf["chain"].unique())
    fig, axes = plt.subplots(len(chains), 1, figsize=(10, 3.0 * len(chains)))
    axes = np.atleast_1d(axes)
    names = {"L": "Ligand", "R": "Receptor"}

    for ax, ch in zip(axes, chains):
        sub = pdf[pdf["chain"] == ch].sort_values("resid")
        colors = [FAV if v < 0 else UNFAV for v in sub["value"]]
        ax.bar(sub["resid"], sub["value"], color=colors, width=0.85)
        ax.axhline(0, color="black", lw=0.9)
        ax.axhline(-cutoff, color="#34495e", lw=0.7, ls="--", alpha=0.6)
        ax.set_title(f"{names.get(ch, ch)} (chain {ch})", loc="left")
        ax.set_xlabel("Residue number")
        ax.set_ylabel("kcal/mol")
        for _, r in sub[sub["value"].abs() >= cutoff].iterrows():
            ax.annotate(f"{r.resname.title()}{r.resid}",
                        (r.resid, r.value), fontsize=7.5, rotation=90,
                        ha="center",
                        va="bottom" if r.value >= 0 else "top",
                        xytext=(0, 3 if r.value >= 0 else -3),
                        textcoords="offset points")
        ax.margins(y=0.28)
        ax.grid(axis="x", visible=False)

    fig.suptitle(f"Per-residue contributions along the sequence "
                 f"(|ΔG| ≥ {cutoff} kcal/mol labelled)", y=1.0)
    fig.tight_layout()
    path = os.path.join(outdir, "06_residue_map.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_sidechain_backbone(decomp, outdir, top=15):
    """Sidechain vs backbone split for the strongest residues."""
    if ("DELTAS", "Sidechain") not in decomp or ("DELTAS", "Backbone") not in decomp:
        return None
    sc = decomp[("DELTAS", "Sidechain")][("TOTAL", "Avg")]
    bb = decomp[("DELTAS", "Backbone")][("TOTAL", "Avg")]
    common = [r for r in sc.index if r in bb.index]
    net = (sc.loc[common] + bb.loc[common])
    order = net.abs().sort_values(ascending=False).index[:top]
    order = net.loc[order].sort_values().index

    labels = [residue_label(r)[0] for r in order]
    ypos = np.arange(len(order))
    h = 0.38

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(order) + 2.0))
    ax.barh(ypos + h / 2, sc.loc[order].values, height=h,
            color="#2c6fbb", label="sidechain")
    ax.barh(ypos - h / 2, bb.loc[order].values, height=h,
            color="#e67e22", label="backbone")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", lw=0.9)
    ax.set_xlabel("Contribution (kcal/mol)")
    ax.set_title("Sidechain vs. backbone contributions")
    ax.legend(fontsize=9)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()

    path = os.path.join(outdir, "07_sidechain_backbone.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_correlation(frames, outdir):
    """Correlation between Delta components across frames."""
    if "Delta" not in frames:
        return None
    df = frames["Delta"]
    cols = [c for c in ["VDWAALS", "EEL", "EGB", "ESURF", "GGAS", "GSOLV", "TOTAL"]
            if c in df.columns and df[c].std(ddof=1) > 1e-8]
    if len(cols) < 2:
        return None
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Pearson r")
    ax.set_title("Component correlations across frames")
    ax.grid(False)
    fig.tight_layout()

    path = os.path.join(outdir, "08_correlations.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def autodetect(pattern, explicit):
    if explicit:
        return explicit
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser(description="Plot gmx_MMPBSA GB output.")
    ap.add_argument("--summary", help="FINAL_RESULTS_MMPBSA*.dat")
    ap.add_argument("--frames", help="FINAL_RESULTS_MMPBSA*.csv")
    ap.add_argument("--decomp", help="FINAL_DECOMP_MMPBSA*.dat")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--top", type=int, default=15,
                    help="number of residues in the ranked plots")
    ap.add_argument("--cutoff", type=float, default=1.0,
                    help="|kcal/mol| threshold for labelling residues")
    ap.add_argument("--raw-terms", action="store_true",
                    help="plot electrostatic and polar solvation separately")
    ap.add_argument("--show", action="store_true", help="also open the figures")
    args = ap.parse_args()

    summary_f = autodetect("FINAL_RESULTS_MMPBSA*.dat", args.summary)
    frames_f = autodetect("FINAL_RESULTS_MMPBSA*.csv", args.frames)
    decomp_f = autodetect("FINAL_DECOMP_MMPBSA*.dat", args.decomp)

    if not any([summary_f, frames_f, decomp_f]):
        sys.exit("No gmx_MMPBSA files found. Pass paths with --summary/--frames/--decomp.")

    os.makedirs(args.outdir, exist_ok=True)
    made = []

    if summary_f:
        print(f"summary : {summary_f}")
        summary = parse_summary(summary_f)
        made.append(fig_energy_terms(summary, args.outdir))
        if "Delta" in summary and "TOTAL" in summary["Delta"].index:
            d = summary["Delta"].loc["TOTAL"]
            print(f"  dG_bind = {d['Average']:.2f} +/- {d['SEM']:.2f} kcal/mol")

    if frames_f:
        print(f"frames  : {frames_f}")
        frames = parse_frames(frames_f)
        made.append(fig_convergence(frames, args.outdir))
        made.append(fig_term_traces(frames, args.outdir))
        made.append(fig_correlation(frames, args.outdir))

    if decomp_f:
        print(f"decomp  : {decomp_f}")
        decomp = parse_decomp(decomp_f)
        made.append(fig_per_residue(decomp, args.outdir, top=args.top))
        made.append(fig_residue_breakdown(decomp, args.outdir, top=args.top,
                                          raw_terms=args.raw_terms))
        made.append(fig_residue_map(decomp, args.outdir, cutoff=args.cutoff))
        made.append(fig_sidechain_backbone(decomp, args.outdir, top=args.top))

        key = ("DELTAS", "Total")
        if key in decomp:
            tot = decomp[key][("TOTAL", "Avg")]
            csv_path = os.path.join(args.outdir, "per_residue_ranked.csv")
            out = decomp[key].copy()
            out.columns = [f"{a} {b}" for a, b in out.columns]
            out.reindex(tot.abs().sort_values(ascending=False).index).to_csv(csv_path)
            made.append(csv_path)

    made = [p for p in made if p]
    print("\nWrote:")
    for p in made:
        print("  " + p)

    if args.show:
        for p in made:
            if p.endswith(".png"):
                img = plt.imread(p)
                fig, ax = plt.subplots(figsize=(10, 7))
                ax.imshow(img)
                ax.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
