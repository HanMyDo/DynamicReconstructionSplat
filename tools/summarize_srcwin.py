"""One page: --src_window_dyn 2 against 4, per sequence and horizon.

Reads the probe metrics.json files chain_srcwin.sh produced and writes a table
of DELTAS (2 minus 4) plus a count of how many sequences each metric moved in
the same direction, which is the thing that decides adoption -- a mean over
five sequences can be carried by one of them.

SIGNS ARE NOT ALL THE SAME WAY. For psnr and ssim, higher is better, so a
positive delta favours 2. For lpips, LOWER is better, so a NEGATIVE delta
favours 2. Every row below is printed as "favours 2?" rather than a raw sign,
because getting that backwards in a summary is how a rejected arm gets adopted.

The dynamic column is expected to favour 4 -- on balloon it was -0.64 -- so the
question this answers is whether the other four metrics and the renders are
worth that, not whether 2 wins everywhere.

USAGE  python tools/summarize_srcwin.py --tag sw2
"""
import argparse
import glob
import json
import os
import re

# (key, label, higher_is_better)
METRICS = [
    ("psnr", "overall", True),
    ("psnr_dynamic", "dynamic", True),
    ("psnr_static", "static", True),
    ("ssim", "ssim", True),
    ("lpips", "lpips", False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sw2")
    args = ap.parse_args()

    # output_probe_frozen_d2_s8_<short>_<tag>/metrics.json
    pat = re.compile(r"output_probe_frozen_(d2|d4)_s(\d+)_(.+)_" + re.escape(args.tag) + r"$")
    runs = {}
    for d in sorted(glob.glob(f"output_probe_frozen_d*_s*_*_{args.tag}")):
        m = pat.match(d)
        if not m or not os.path.isfile(os.path.join(d, "metrics.json")):
            continue
        arm, stride, seq = m.group(1), int(m.group(2)), m.group(3)
        with open(os.path.join(d, "metrics.json")) as fh:
            runs[(seq, stride, arm)] = json.load(fh)

    pairs = sorted({(s, st) for (s, st, a) in runs}
                   & {(s, st) for (s, st, a) in runs if a == "d2"}
                   & {(s, st) for (s, st, a) in runs if a == "d4"})

    out = [f"# --src_window_dyn 2 vs 4 ({args.tag})", "",
           "Deltas are **2 minus 4**, fill OFF, `F` otherwise identical, both arms in",
           "one allocation. `favours 2` accounts for the metric's direction (lower LPIPS",
           "is better), so a tick always means 2 is the better arm on that metric.", "",
           "⚠️ dynamic is EXPECTED to favour 4 (balloon measured −0.64). The question is",
           "whether the other four metrics plus the renders are worth that trade.", ""]
    if not pairs:
        out += ["No paired runs found — check that both arms completed.", ""]
        open(f"srcwin_{args.tag}.md", "w").write("\n".join(out))
        print("\n".join(out))
        return

    hdr = "| sequence | s | " + " | ".join(lbl for _, lbl, _ in METRICS) + " |"
    out += [hdr, "|---" * (len(METRICS) + 2) + "|"]
    wins = {k: 0 for k, _, _ in METRICS}
    tot = 0
    for seq, st in pairs:
        a, b = runs[(seq, st, "d2")], runs[(seq, st, "d4")]
        cells = []
        tot += 1
        for key, _, hib in METRICS:
            if a.get(key) is None or b.get(key) is None:
                cells.append("—")
                continue
            d = a[key] - b[key]
            good = (d > 0) if hib else (d < 0)
            wins[key] += int(good)
            fmt = f"{d:+.4f}" if key in ("ssim", "lpips") else f"{d:+.3f}"
            cells.append(f"{fmt} {'✅' if good else '❌'}")
        out.append(f"| {seq} | {st} | " + " | ".join(cells) + " |")

    out += ["", f"**Rows: {tot}.** How often 2 is the better arm:", ""]
    for key, lbl, _ in METRICS:
        out.append(f"- {lbl}: **{wins[key]}/{tot}**")
    out += ["",
            "Adopt only if the non-dynamic metrics are consistent AND the renders agree —",
            "four coverage mechanisms were rejected on Oct 23 precisely because a metric",
            "moved while the picture did not."]
    open(f"srcwin_{args.tag}.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
