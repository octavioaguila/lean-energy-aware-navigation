#!/usr/bin/env python3
"""Across-seed statistics: python inference/seed_stats.py RUN_DIR"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_utils import (wilcoxon_paired, mcnemar_paired, paired_ratio,
                         paired_ratio_bootstrap, fmt_p)

TIERS = ["easy", "medium", "hard", "all"]
METRICS = {"jpm": "energy_per_executed_meter", "spl": "spl_euclidean", "time": "elapsed_time"}
CLASSICAL = {"NMPC_we0": "NMPC-0.0", "NMPC_we0.05": "NMPC-0.05", "NMPC_we0.1": "NMPC-0.1",
             "eadwa_we0": "EA-DWA-0.0", "eadwa_we0.0003": "EA-DWA-0.0003", "eadwa_we0.0005": "EA-DWA-0.0005"}
SUBMITTED_RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "global_paired", "20260616_002825")
DT = 0.1


def label(w, seed):
    return f"vanilla_drl_s{seed}" if w == 0 else f"lean_we{w:g}_s{seed}"


def planned(eps, tier):
    return [r for r in eps if r["planning_success"] and (tier == "all" or r["difficulty"] == tier)]


def res(r, c):
    return r["controller_results"].get(c, {})


def ok(r, c):
    return res(r, c).get("outcome") == "SUCCESS"


def marginal(eps, c):
    # success-only episode means
    succ = [r for r in eps if ok(r, c)]
    out = {"n": len(eps), "n_success": len(succ), "sr": 100.0 * len(succ) / len(eps)}
    for k, key in METRICS.items():
        out[k] = float(np.mean([res(r, c)[key] for r in succ])) if succ else None
    return out


def submitted(run_dir, classical):
    # submitted mean, batch-means CI
    s = json.load(open(os.path.join(run_dir, "summary.json")))
    blocks = {t: s["per_difficulty"][t] for t in TIERS[:3]} | {"all": s["overall"]}
    scale = {"sr": ("controller_success_rate_given_plan", 100.0), "jpm": ("jpm", 1.0),
             "spl": ("spl", 1.0), "time": ("steps", DT)}
    return {t: {c: {k: {"mean": b["controllers"][c][key][0] * f, "ci": b["controllers"][c][key][1] * f}
                    for k, (key, f) in scale.items()} for c in classical} for t, b in blocks.items()}


def agg(vals):
    v = np.array([x for x in vals if x is not None], dtype=float)
    if not len(v):
        return None
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "per_seed": v.tolist()}


def pair(eps, a, b, n_boot=0, cluster=False):
    # negative favors b
    both = [r for r in eps if ok(r, a) and ok(r, b)]
    ea = [res(r, a)["energy_total"] for r in both]
    eb = [res(r, b)["energy_total"] for r in both]
    out = {"n_pairs": len(both),
           "sr_mcnemar": mcnemar_paired([ok(r, a) for r in eps], [ok(r, b) for r in eps])}
    if both:
        out["dE"] = paired_ratio(ea, eb)
        out["win"] = 100.0 * float(np.mean(np.array(eb) < np.array(ea)))
        out["wilcoxon"] = wilcoxon_paired(ea, eb)
        if n_boot:
            cl = [r["world_id"] for r in both] if cluster else None
            out["dE_ci"] = paired_ratio_bootstrap(ea, eb, n_boot, clusters=cl)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--weights", type=float, nargs="+", default=[0, 0.002, 0.003, 0.004])
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 200, 300])
    ap.add_argument("--headline", type=float, default=0.004)
    ap.add_argument("--baselines", nargs="+", default=["NMPC_we0.05", "eadwa_we0.0003"])
    ap.add_argument("--classical", nargs="+", default=list(CLASSICAL))
    ap.add_argument("--submitted_run", default=SUBMITTED_RUN, help="run behind the submitted classical rows")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--cluster", action="store_true", help="bootstrap whole worlds instead of episodes")
    args = ap.parse_args()

    eps = json.load(open(os.path.join(args.run_dir, "episodes.json")))["episodes"]
    ctrls = set(eps[0]["controller_results"])
    arms = {w: [label(w, s) for s in args.seeds] for w in args.weights}
    missing = [c for c in sum(arms.values(), []) + args.baselines + args.classical if c not in ctrls]
    if missing:
        sys.exit(f"controllers not in episodes.json: {missing}")

    out = {"run_dir": args.run_dir, "submitted_run": args.submitted_run, "seeds": args.seeds,
           "weights": args.weights, "headline": args.headline,
           "bootstrap": {"n": args.boot, "cluster_by_world": args.cluster},
           "table3": {}, "table3_classical": {}, "table3_submitted": submitted(args.submitted_run, args.classical),
           "table4": {}, "matrix": {}}
    for tier in TIERS:
        ep = planned(eps, tier)
        out["table3"][tier] = {f"{w:g}": {k: agg([marginal(ep, c)[k] for c in cs]) for k in ["sr"] + list(METRICS)}
                               for w, cs in arms.items()}
        out["table3_classical"][tier] = {c: marginal(ep, c) for c in args.classical}

    hl, van = arms[args.headline], arms[0]
    for tier in TIERS[:3]:
        ep = planned(eps, tier)
        out["table4"][tier] = {}
        for b in args.baselines + ["vanilla_matched"]:
            # matched seed ablation
            cells = [pair(ep, van[i] if b == "vanilla_matched" else b, hl[i], args.boot, args.cluster)
                     for i in range(len(hl))]
            k = sum(1 for c in cells if c["dE"] < 0 and (c["wilcoxon"]["p_value"] or 1) < 0.05)
            out["table4"][tier][b] = {"per_seed": cells, "dE": agg([c["dE"] for c in cells]),
                                      "win": agg([c["win"] for c in cells]), "k_sig_lower": k}
        out["matrix"][tier] = [[pair(ep, va, lb)["dE"] for lb in hl] for va in van]

    json.dump(out, open(os.path.join(args.run_dir, "seed_stats.json"), "w"), indent=1)
    txt = format_text(out, args)
    open(os.path.join(args.run_dir, "seed_stats.txt"), "w").write(txt)
    print(txt)


def format_text(out, args):
    L = [f"SEED STATS  run={out['run_dir']}  seeds={args.seeds}  headline w_e={args.headline:g}  "
         f"bootstrap={args.boot} ({'world-cluster' if args.cluster else 'episode'})", ""]
    L.append(f"TABLE III  DRL: mean +- sd (per seed); classical: submitted mean +- 95% CI ({out['submitted_run']}), "
             "this run in parentheses. SR|plan %, E/m J/m, SPL, time s")
    for tier in TIERS:
        for c, m in out["table3_submitted"][tier].items():
            now = out["table3_classical"][tier][c]
            cell = lambda k, f: f"{m[k]['mean']:{f}} +- {m[k]['ci']:{f}} ({now[k]:{f}})"
            L.append(f"  {tier:6s} {CLASSICAL.get(c, c):14s} SR {cell('sr', '5.1f')} | E/m {cell('jpm', '5.0f')} | "
                     f"SPL {cell('spl', '.3f')} | time {cell('time', '4.1f')}")
        for w, m in out["table3"][tier].items():
            cell = lambda k, f: (f"{m[k]['mean']:{f}} +- {m[k]['sd']:{f}} "
                                 f"({', '.join(format(x, f) for x in m[k]['per_seed'])})")
            L.append(f"  {tier:6s} w={w:11s} SR {cell('sr', '5.1f')} | E/m {cell('jpm', '5.0f')} | "
                     f"SPL {cell('spl', '.3f')} | time {cell('time', '4.1f')}")
    L.append("")
    L.append("TABLE IV  headline vs baseline: dE% and Win% mean +- sd over seeds, k/3 seeds sig. lower;")
    L.append("          per seed: dE [bootstrap 95% CI] wilcoxon-p | McNemar b/c p")
    for tier, rows in out["table4"].items():
        for b, m in rows.items():
            d, w = m["dE"], m["win"]
            L.append(f"  {tier:6s} {b:16s} {d['mean']:+6.1f} +- {d['sd']:.1f}  win {w['mean']:3.0f} +- {w['sd']:.0f}  "
                     f"sig {m['k_sig_lower']}/{len(m['per_seed'])}")
            for s, c in zip(args.seeds, m["per_seed"]):
                ci = c.get("dE_ci", (None, None))
                mc = c["sr_mcnemar"]
                L.append(f"      s{s}: {c['dE']:+6.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}] p={fmt_p(c['wilcoxon']['p_value'])} "
                         f"n={c['n_pairs']} | SR b/c={mc['n_a_only']}/{mc['n_b_only']} p={mc['p_value']:.3f}")
    L.append("")
    L.append("MATRIX  dE% headline vs vanilla, rows = vanilla seed, cols = LEAN seed (diagonal = matched)")
    for tier, M in out["matrix"].items():
        for s, row in zip(args.seeds, M):
            L.append(f"  {tier:6s} vanilla_s{s}: " + "  ".join(f"{x:+6.1f}" for x in row))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
