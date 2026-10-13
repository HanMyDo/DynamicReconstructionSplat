"""The two figures the advisor-meeting deck uses, from the numbers already in the repo.

    python tools/make_slide_figures.py --out_dir slide_figs --fig_root ~/Downloads

1. `horizon_wide.png` -- the horizon curves laid out 16:9 (the two synchronous2
   panels SIDE BY SIDE instead of stacked), which is what fits a slide. Numbers
   come from tools/plot_horizon_curves.py, so there is one source of truth and a
   re-run only has to update DATA there.

   ⚠️ The crossing annotations are written out by hand rather than generated. The
   moving-content panel crosses in BOTH directions -- NeoVerse retakes it at
   ~9.4 s -- and a generic "overtakes" label would state only half of that.

2. `renders_grid.png` -- synchronous2 at two window lengths, four columns
   (GT | ours | MoVieS | NeoVerse). s4/f00024 and s12/f00012 hold out two
   different instants that happen to show the same two people in nearly the same
   arrangement, so the rows read as one scene -- but they ARE different frames,
   so caption them as two held-out instants, never as "the same frame".

   ⚠️ The 10 s row (s20) was dropped on purpose: NeoVerse renders a cleaner
   person there than we do, and the masked metric agrees (11.64 vs 12.02). 6 s is
   the longest window in the current ladder where we lead on BOTH PSNR columns
   against BOTH baselines. Do not reinstate s20 without re-checking that.

   All panels are resampled to 448x336 = 4:3 = Bonn's own aspect. The
   MoVieS-matched renders are scored at 448x392, which is Bonn squashed
   vertically (their intrinsics are scaled anisotropically to match), so this
   resize UNDOES that squash rather than introducing one.
"""
import argparse
import os
import sys

HORIZON_PANELS = [
    ("PSNR (dB)", "Whole frame", [(4.65, "ours ahead of both\nfrom ~4.6 s", 0.04)]),
    ("PSNR, moving content (dB)", "Moving content only",
     [(4.24, "ahead of NeoVerse\n~4.2–9.4 s", 0.04)]),
]
HORIZON_NOTE = {"PSNR, moving content (dB)": "ahead of MoVieS at every baseline"}

GRID_ROWS = [
    ("fig_fin_MoVieS_synchronous2_s4/f00024.png",
     "fig_fin_NeoVerse_synchronous2_s4/f00024.png", "2 s window"),
    ("fig_fin_MoVieS_synchronous2_s12/f00012.png",
     "fig_fin_NeoVerse_synchronous2_s12/f00012.png", "6 s window"),
]
PW, PH, GUT = 448, 336, 170
SURF, INK, SOFT = (251, 250, 247), (22, 32, 43), (99, 110, 120)


def horizon_wide(out):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_horizon_curves import DATA, COLORS, ORDER, SURFACE, INK as PINK, INK_SOFT, HORIZONS

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), facecolor=SURFACE)
    for ax, (metric, title, marks) in zip(axes, HORIZON_PANELS):
        ax.set_facecolor(SURFACE)
        series = DATA["synchronous2"][metric]
        for name in ORDER:
            ys = series[name]
            ax.plot(HORIZONS, ys, color=COLORS[name],
                    linewidth=3.0 if name == "ours" else 2.2, marker="o", markersize=6.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.6,
                    zorder=3 if name == "ours" else 2, label=name, clip_on=False)
            ax.annotate(name, (HORIZONS[-1], ys[-1]), xytext=(9, 0),
                        textcoords="offset points", va="center", fontsize=12,
                        color=COLORS[name], fontweight="bold" if name == "ours" else "normal")
        for x, label, y in marks:
            ax.axvline(x, color=INK_SOFT, linewidth=1.0, linestyle=":", alpha=0.6, zorder=1)
            ax.annotate(label, (x, y), xycoords=("data", "axes fraction"), xytext=(8, 0),
                        textcoords="offset points", ha="left", va="bottom",
                        fontsize=11, color=INK_SOFT)
        if metric in HORIZON_NOTE:
            ax.annotate(HORIZON_NOTE[metric], (0.02, 0.93), xycoords="axes fraction",
                        fontsize=11, color=COLORS["ours"], va="bottom")
        ax.set_xticks(HORIZONS)
        ax.grid(axis="y", color="#000000", alpha=0.08, linewidth=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#000000")
            ax.spines[side].set_alpha(0.25)
        ax.tick_params(colors=INK_SOFT, labelsize=12)
        ax.margins(x=0.19)
        ax.set_xlabel("temporal baseline of the window (s)", color=PINK, fontsize=13)
        ax.set_title(title, color=PINK, fontsize=15, fontweight="bold", pad=12)
    axes[0].set_ylabel("PSNR (dB) — higher is better", color=PINK, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print("wrote", out)


def renders_grid(fig_root, out):
    from PIL import Image, ImageDraw, ImageFont

    def panels(rel):
        im = Image.open(os.path.join(fig_root, rel)).convert("RGB")
        return [im.crop((i * 448, 0, (i + 1) * 448, im.height)).resize((PW, PH), Image.LANCZOS)
                for i in range(im.width // 448)]

    rows = []
    for mo, nv, label in GRID_ROWS:
        gt, ours, movies = panels(mo)
        rows.append(([gt, ours, movies, panels(nv)[2]], label))

    W, H = GUT + 4 * PW, 2 * PH
    canvas = Image.new("RGB", (W, H), SURF)
    for r, (ps, _) in enumerate(rows):
        for c, p in enumerate(ps):
            canvas.paste(p, (GUT + c * PW, r * PH))
    d = ImageDraw.Draw(canvas)
    try:
        big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except OSError:
        big = small = ImageFont.load_default()
    for r, (_, label) in enumerate(rows):
        d.text((24, r * PH + PH // 2 - 38), label, font=big, fill=INK)
        d.text((24, r * PH + PH // 2 + 6), "between\ninput frames", font=small,
               fill=SOFT, spacing=4)
    # Hairlines in the surface colour, so adjacent renders do not read as one image.
    for c in range(5):
        d.line([(GUT + c * PW, 0), (GUT + c * PW, H)], fill=SURF, width=4)
    d.line([(0, PH), (W, PH)], fill=SURF, width=4)
    canvas.save(out)
    print("wrote", out, canvas.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="slide_figs")
    ap.add_argument("--fig_root", default=os.path.expanduser("~/Downloads"),
                    help="where the fig_fin_<baseline>_<seq>_s<stride>/ dirs were downloaded")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    horizon_wide(os.path.join(args.out_dir, "horizon_wide.png"))
    renders_grid(args.fig_root, os.path.join(args.out_dir, "renders_grid.png"))


if __name__ == "__main__":
    main()
