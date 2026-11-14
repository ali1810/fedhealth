"""Run the full comparison and write results/results.md plus figures.

    python benchmark.py            # 5 seeds, about a minute on a laptop CPU
    python benchmark.py --seeds 2  # quicker
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fedhealth.data import load_dataset, make_clients, quality_report  # noqa: E402
from fedhealth.federation import FLConfig, run_centralised, run_federated, run_local_only  # noqa: E402

OUT = Path("results")
N_CLIENTS = 5


def mean_std(xs):
    return float(np.mean(xs)), float(np.std(xs))


def fmt(ms, pct=True):
    m, s = ms
    return f"{100 * m:.1f} ± {100 * s:.1f}" if pct else f"{m:.2f} ± {s:.2f}"


def main(seeds: int, rounds: int) -> None:
    OUT.mkdir(exist_ok=True)
    base = FLConfig(rounds=rounds)
    methods = {
        "FedAvg": replace(base, secure_aggregation=False),
        "FedAvg + SecAgg": replace(base),
        "FedProx + SecAgg": replace(base, strategy="fedprox"),
        "DP-FedAvg + SecAgg (z=0.5)": replace(base, dp_noise_multiplier=0.5),
        "DP-FedAvg + SecAgg (z=1.0)": replace(base, dp_noise_multiplier=1.0),
        "DP-FedAvg + SecAgg (z=2.0)": replace(base, dp_noise_multiplier=2.0),
    }
    alphas = {"strong non-IID (α=0.3)": 0.3, "moderate (α=1)": 1.0, "near IID (α=100)": 100.0}

    rows, curves, quality = [], {}, None
    for a_name, alpha in alphas.items():
        res: dict[str, dict[str, list]] = {}
        for s in range(seeds):
            Xtr, Xte, ytr, yte = load_dataset(seed=s)
            clients = make_clients(Xtr, ytr, n_clients=N_CLIENTS, alpha=alpha, seed=s)
            if quality is None and alpha == 0.3:
                quality = [quality_report(c) for c in clients]
            cfg0 = replace(base, seed=s)
            for name, val in [("Centralised (pooled data)", run_centralised(clients, Xte, yte, cfg0)),
                              ("Local only (mean of hospitals)", run_local_only(clients, Xte, yte, cfg0))]:
                r = res.setdefault(name, {"acc": [], "auc": []})
                r["acc"].append(val[0])
                r["auc"].append(val[1])
            for name, cfg in methods.items():
                h = run_federated(clients, Xte, yte, replace(cfg, seed=s))
                r = res.setdefault(name, {"acc": [], "auc": [], "local": [], "eps": [], "mb": [],
                                          "sec": [], "curve": []})
                r["acc"].append(h.global_acc[-1])
                r["auc"].append(h.global_auc[-1])
                r["local"].append(h.local_acc[-1])
                r["eps"].append(h.epsilon)
                r["mb"].append(sum(h.bytes_sent) / 1e6)
                r["sec"].append(h.seconds)
                r["curve"].append(h.global_auc)
        for name, r in res.items():
            rows.append({"setting": a_name, "method": name,
                         "acc": mean_std(r["acc"]), "auc": mean_std(r["auc"]),
                         "local": mean_std(r["local"]) if "local" in r else None,
                         "eps": r["eps"][0] if "eps" in r else None,
                         "mb": float(np.mean(r["mb"])) if "mb" in r else None,
                         "sec": float(np.mean(r["sec"])) if "sec" in r else None})
            if "curve" in r:
                curves[(a_name, name)] = np.array(r["curve"])

    # ---------- markdown report
    lines = [f"# Results\n\n{N_CLIENTS} simulated hospitals, {rounds} rounds, {seeds} seeds "
             "(mean ± std). Global test = held-out 20% of patients no hospital trained on.\n"]
    for a_name in alphas:
        lines += [f"\n## {a_name}\n",
                  "| Method | Global accuracy % | Global ROC AUC % | Accuracy on own hospital test % "
                  "| ε (δ=1e-5) | Traffic (MB) | Time (s) |",
                  "|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["setting"] != a_name:
                continue
            eps = "-" if r["eps"] is None else ("∞ (no DP)" if np.isinf(r["eps"]) else f"{r['eps']:.1f}")
            local = "-" if r["local"] is None else fmt(r["local"])
            mb = "-" if r["mb"] is None else f"{r['mb']:.2f}"
            sec = "-" if r["sec"] is None else f"{r['sec']:.1f}"
            lines.append(f"| {r['method']} | {fmt(r['acc'])} | {fmt(r['auc'])} | {local} | {eps} | {mb} | {sec} |")
    lines += ["\n## Data quality reports (strong non-IID split, seed 0)\n",
              "| Hospital | Training samples | Positive rate | Missing | Outlier share | Warnings |",
              "|---|---|---|---|---|---|"]
    for q in quality:
        lines.append(f"| {q.name} | {q.n_samples} | {q.positive_rate:.2f} | {q.missing_fraction:.2f} | "
                     f"{q.outlier_fraction:.3f} | {', '.join(q.warnings) or '-'} |")
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(
        {"config": asdict(base), "seeds": seeds, "rows": rows}, indent=2, default=float), encoding="utf-8")

    # ---------- figures
    fig, axes = plt.subplots(1, len(alphas), figsize=(15, 4.2), sharey=True)
    for ax, a_name in zip(axes, alphas):
        for name in methods:
            c = curves[(a_name, name)]
            m = c.mean(0)
            ax.plot(range(1, len(m) + 1), m, label=name)
            ax.fill_between(range(1, len(m) + 1), m - c.std(0), m + c.std(0), alpha=0.15)
        ax.set_title(a_name)
        ax.set_xlabel("Communication round")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ROC AUC on held-out patients")
    axes[0].set_ylim(0.5, 1.0)
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    dp = [r for r in rows if r["setting"] == "moderate (α=1)" and r["eps"] is not None and not np.isinf(r["eps"])]
    ref = next(r for r in rows if r["setting"] == "moderate (α=1)" and r["method"] == "FedAvg + SecAgg")
    ax.errorbar([r["eps"] for r in dp], [r["auc"][0] for r in dp], yerr=[r["auc"][1] for r in dp],
                marker="o", capsize=4, label="DP-FedAvg + SecAgg")
    ax.axhline(ref["auc"][0], ls="--", color="grey", label="no DP")
    ax.set_xscale("log")
    ax.set_xlabel("ε (client-level, δ=1e-5) - lower is more private")
    ax.set_ylabel("ROC AUC")
    ax.set_title("Privacy vs. utility (moderate non-IID)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "privacy_utility.png", dpi=150)
    print(f"Wrote {OUT / 'results.md'}, {OUT / 'convergence.png'} and {OUT / 'privacy_utility.png'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--rounds", type=int, default=30)
    args = p.parse_args()
    main(args.seeds, args.rounds)
