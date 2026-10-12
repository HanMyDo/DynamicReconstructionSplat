"""Horizon curves for the slides: ours vs MoVieS vs NeoVerse, two sequences.

THE POINT OF THIS FIGURE. The result is a CROSSING, not a number. Plotting
ABSOLUTE values (not deltas) makes the crossing literal -- the lines cross and the
ordering inverts across the panel -- which a table makes the reader compute.

    python tools/plot_horizon_curves.py --out horizon.png

UPDATING AFTER A RE-RUN. Only the "ours" rows change when our renders change; the
baselines are fixed. Paste the new numbers into DATA below and re-run. The script
prints the crossing points it finds, so you can check them against the plot.

⚠️ The MoVieS comparison scored at 448x392 and the NeoVerse one at 448x336, each
matched to that baseline's render size, so there are two slightly different "ours"
measurements. They agree to 0.02 dB PSNR and 0.008 LPIPS; the MoVieS-table values
are used here. Say so if asked -- it is not a surprise worth being caught by.

Palette: slots 1-3 of the reference categorical theme, validated with
scripts/validate_palette.py (all checks PASS; the aqua slot WARNs at 2.74:1 against
the surface, which obligates the direct end-labels this script draws).
"""
import argparse

HORIZONS = [2, 4, 6, 8, 10]          # seconds of temporal baseline (stride 4..20)

# ---------------------------------------------------------------------------
# MEASURED VALUES -- matched leave-one-out, same frames/masks/resolution per pair.
# Update ONLY the "ours" rows after a re-run.
# ---------------------------------------------------------------------------
DATA = {
    "synchronous2": {
        "PSNR (dB)": {
            "ours":     [20.50, 18.78, 18.11, 17.15, 15.56],
            "MoVieS":   [20.92, 19.10, 17.94, 16.44, 14.88],
            "NeoVerse": [23.80, 19.18, 16.29, 14.83, 13.48],
        },
        "PSNR, moving content (dB)": {
            "ours":     [19.75, 16.80, 15.80, 14.21, 11.70],
            "MoVieS":   [17.90, 15.73, 14.69, 12.92, 10.67],
            "NeoVerse": [20.98, 16.65, 14.24, 13.41, 12.02],
        },
        "LPIPS (dynamic crop)": {
            "ours":     [0.2527, 0.3119, 0.3484, 0.3782, 0.4053],
            "MoVieS":   [0.2090, 0.2339, 0.2645, 0.2955, 0.3137],
            "NeoVerse": [0.2106, 0.3060, 0.3903, 0.4112, 0.4250],
        },
    },
    "balloon": {
        "PSNR (dB)": {
            "ours":     [19.14, 16.74, 14.88, 13.33, 12.86],
            "MoVieS":   [21.40, 18.05, 16.51, 15.14, 14.25],
            "NeoVerse": [22.46, 18.38, 15.80, 13.54, 11.69],
        },
        "PSNR, moving content (dB)": {
            "ours":     [16.84, 14.87, 13.63, 12.68, 12.30],
            "MoVieS":   [18.52, 15.78, 13.12, 11.69, 11.19],
            "NeoVerse": [19.51, 15.77, 13.69, 11.77, 10.22],
        },
        "LPIPS (dynamic crop)": {
            "ours":     [0.3894, 0.4774, 0.5405, 0.5926, 0.6016],
            "MoVieS":   [0.2426, 0.3122, 0.3560, 0.3888, 0.4185],
            "NeoVerse": [0.3013, 0.4063, 0.4760, 0.5379, 0.5826],
        },
    },
}

# Categorical slots 1-3, fixed order -- never cycled, never reassigned by rank.
COLORS = {"ours": "#2a78d6", "MoVieS": "#eb6834", "NeoVerse": "#1baf7a"}
ORDER = ["ours", "MoVieS", "NeoVerse"]
SURFACE, INK, INK_SOFT = "#fcfcfb", "#0b0b0b", "#52514e"
HIGHER_IS_BETTER = {"PSNR (dB)": True, "PSNR, moving content (dB)": True,
                    "LPIPS (dynamic crop)": False}
# Lead slide: synchronous2, overall + moving content. The full grid (both
# sequences, LPIPS included) is one flag away -- keep it as the backup slide
# rather than deleting it, so an "does it hold elsewhere?" question is
# answered rather than deflected.
LEAD = (["synchronous2"], ["PSNR (dB)", "PSNR, moving content (dB)"])


def crossings(xs, a, b, higher_better):
    """Where `a` overtakes `b`, by linear interpolation between samples."""
    out = []
    for i in range(len(xs) - 1):
        d0 = (a[i] - b[i]) * (1 if higher_better else -1)
        d1 = (a[i + 1] - b[i + 1]) * (1 if higher_better else -1)
        if d0 < 0 <= d1:
            t = -d0 / (d1 - d0) if d1 != d0 else 0.0
            out.append(xs[i] + t * (xs[i + 1] - xs[i]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="horizon_curves.png")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--all", action="store_true",
                    help="backup slide: both sequences, every metric including LPIPS")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib not found -- run this in an env that has it (dynrec does)")

    seqs, metrics = (list(DATA), list(DATA[list(DATA)[0]])) if args.all else LEAD
    fig, axes = plt.subplots(len(metrics), len(seqs),
                             figsize=(5.9 * len(seqs) + 0.6, 3.6 * len(metrics)),
                             facecolor=SURFACE, squeeze=False)

    for r, metric in enumerate(metrics):
        for c, seq in enumerate(seqs):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            series = DATA[seq][metric]

            for name in ORDER:
                ys = series[name]
                lw = 2.6 if name == "ours" else 2.0
                ax.plot(HORIZONS, ys, color=COLORS[name], linewidth=lw,
                        marker="o", markersize=5.5, markeredgecolor=SURFACE,
                        markeredgewidth=1.4, zorder=3 if name == "ours" else 2,
                        label=name, clip_on=False)
                # Direct end-labels: identity never by colour alone, and the
                # contrast WARN on the aqua slot requires visible labels.
                ax.annotate(name, (HORIZONS[-1], ys[-1]), xytext=(7, 0),
                            textcoords="offset points", va="center",
                            fontsize=9, color=INK_SOFT,
                            fontweight="bold" if name == "ours" else "normal")

            hb = HIGHER_IS_BETTER[metric]
            # Collect first, then place: two crossings can land within a second of
            # each other and the labels overprint (measured: 4.4s and 5.3s on
            # synchronous2 PSNR). Stagger the height of near neighbours.
            marks = sorted((x, rival) for rival in ("MoVieS", "NeoVerse")
                           for x in crossings(HORIZONS, series["ours"], series[rival], hb))
            span = (HORIZONS[-1] - HORIZONS[0]) or 1
            prev_x, level = None, 0
            for x, rival in marks:
                level = (level + 1) % 2 if (prev_x is not None
                                            and (x - prev_x) / span < 0.22) else 0
                prev_x = x
                ax.axvline(x, color=INK_SOFT, linewidth=0.8, linestyle=":",
                           alpha=0.55, zorder=1)
                ax.annotate(f"overtakes {rival}\n~{x:.1f}s", (x, 0.04 + 0.13 * level),
                            xycoords=("data", "axes fraction"),
                            ha="center", va="bottom", fontsize=7.5, color=INK_SOFT)

            ax.set_xticks(HORIZONS)
            ax.grid(axis="y", color="#000000", alpha=0.07, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color("#000000")
                ax.spines[side].set_alpha(0.25)
            ax.tick_params(colors=INK_SOFT, labelsize=9)
            ax.margins(x=0.16)
            if c == 0:
                arrow = "higher is better" if hb else "lower is better"
                ax.set_ylabel(f"{metric}\n({arrow})", color=INK, fontsize=10)
            if r == len(metrics) - 1:
                ax.set_xlabel("temporal baseline (s)", color=INK, fontsize=10)
            if r == 0:
                ax.set_title(seq, color=INK, fontsize=12, fontweight="bold", pad=10)

    # Legend present for >=2 series, in addition to the direct labels.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=9.5, labelcolor=INK_SOFT, bbox_to_anchor=(0.5, -0.005))
    # Title length has to suit the figure width, or it clips at both edges in the
    # one-sequence layout.
    title = ("Reconstruction quality vs temporal baseline — matched leave-one-out"
             if len(seqs) > 1 else "Quality vs temporal baseline\n(matched leave-one-out)")
    fig.suptitle(title, color=INK, fontsize=13 if len(seqs) > 1 else 11.5,
                 fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94 if len(seqs) == 1 else 0.955))
    fig.savefig(args.out, dpi=args.dpi, facecolor=SURFACE)
    print(f"wrote {args.out}")

    print("\ncrossings (ours overtakes):")
    for seq in seqs:
        for metric in metrics:
            hb = HIGHER_IS_BETTER[metric]
            for rival in ("MoVieS", "NeoVerse"):
                a, b = DATA[seq][metric]["ours"], DATA[seq][metric][rival]
                xs = crossings(HORIZONS, a, b, hb)
                lead = [(x > y) == hb for x, y in zip(a, b)]
                if xs:
                    got = ", ".join(f"{x:.1f}s" for x in xs)
                elif all(lead):
                    got = "ahead at every horizon (no crossing needed)"
                elif not any(lead):
                    got = "behind at every horizon"
                else:
                    got = "mixed, no clean crossing"
                print(f"  {seq:14s} {metric:26s} vs {rival:9s}: {got}")


if __name__ == "__main__":
    main()
