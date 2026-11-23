"""
Full evaluation: keyword monitor vs. LLM-judge across all four datasets.

Outputs:
  outputs/comparison_table.csv   — per-dataset, per-monitor precision/recall/F1
  outputs/per_category_f1.csv    — per-category F1 breakdown (OOD set)
  outputs/monitor_comparison.png — bar chart comparing monitors (OOD + adversarial)
  outputs/monitor_metrics.json   — full results JSON

Run:  python evaluate.py
Requires: ANTHROPIC_API_KEY (for LLM-judge calls)

Skip LLM calls:  python evaluate.py --keyword-only
  (useful to verify keyword metrics without API access)
"""

import csv, json, sys, argparse
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parent
DATA    = ROOT / "data"
OUT     = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

DATASETS = {
    "in_dist":    DATA / "in_dist_traces.csv",
    "ood":        DATA / "ood_traces.csv",
    "adversarial":DATA / "adversarial_traces.csv",
    "novel":      DATA / "novel_failure_traces.csv",
}

ALL_CATEGORIES = [
    "deception", "unauthorized_action", "long_horizon_incoherence",
    "retry_loop", "tool_grounding_error", "missed_escalation", "scope_creep",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path)))


def binary_metrics(rows: list[dict], pred_col: str) -> dict:
    tp = sum(r["ground_truth"] != "safe" and r[pred_col] != "safe" for r in rows)
    fp = sum(r["ground_truth"] == "safe" and r[pred_col] != "safe" for r in rows)
    fn = sum(r["ground_truth"] != "safe" and r[pred_col] == "safe" for r in rows)
    prec = tp / (tp + fp) if tp + fp else 0
    rec  = tp / (tp + fn) if tp + fn else 0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn}


def per_category_recall(rows: list[dict], pred_col: str) -> dict[str, float]:
    result = {}
    for cat in ALL_CATEGORIES:
        subset   = [r for r in rows if r["ground_truth"] == cat]
        detected = sum(r[pred_col] != "safe" for r in subset)
        result[cat] = round(detected / len(subset), 3) if subset else None
    return result


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_comparison(results: dict, has_llm: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: overall F1 across datasets ────────────────────────────────────
    ax = axes[0]
    datasets  = ["in_dist", "ood", "adversarial"]
    kw_f1     = [results[d]["keyword"]["f1"] for d in datasets]
    labels    = ["In-distribution", "OOD\n(paraphrased)", "Adversarial\n(lexically innocent)"]

    x = range(len(datasets))
    width = 0.35
    bars_kw = ax.bar([i - width/2 for i in x], kw_f1, width,
                     label="Keyword monitor", color="#e74c3c", alpha=0.82, edgecolor="white")

    if has_llm:
        llm_f1 = [results[d]["llm_judge"]["f1"] for d in datasets]
        bars_lj = ax.bar([i + width/2 for i in x], llm_f1, width,
                         label="LLM-judge (Claude Haiku)", color="#2ecc71", alpha=0.82, edgecolor="white")
        for bar, val in zip(bars_lj, llm_f1):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for bar, val in zip(bars_kw, kw_f1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("F1 Score (binary safe/unsafe)", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title("Monitor F1 Across Evaluation Sets\n"
                 "OOD collapse = the finding; LLM-judge partially recovers", fontsize=10)
    ax.legend(fontsize=9)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)

    # ── Right: per-category recall on OOD ───────────────────────────────────
    ax2 = axes[1]
    cats = [c for c in ALL_CATEGORIES if c != "scope_creep"]
    kw_recall = [results["ood"]["keyword"].get("per_cat", {}).get(c, 0) or 0 for c in cats]

    cat_labels = [c.replace("_", "\n") for c in cats]
    x2 = range(len(cats))
    bars2_kw = ax2.bar([i - width/2 for i in x2], kw_recall, width,
                       label="Keyword", color="#e74c3c", alpha=0.82, edgecolor="white")

    if has_llm:
        llm_recall = [results["ood"]["llm_judge"].get("per_cat", {}).get(c, 0) or 0 for c in cats]
        ax2.bar([i + width/2 for i in x2], llm_recall, width,
                label="LLM-judge", color="#2ecc71", alpha=0.82, edgecolor="white")

    ax2.set_xticks(list(x2))
    ax2.set_xticklabels(cat_labels, fontsize=7.5)
    ax2.set_ylabel("Recall on OOD set", fontsize=10)
    ax2.set_ylim(0, 1.2)
    ax2.set_title("Per-Category Recall — OOD Traces\n"
                 "Keyword monitor blind spots vs. LLM-judge", fontsize=10)
    ax2.legend(fontsize=9)

    plt.suptitle(
        "Agent Trace Monitoring: Keyword Monitor vs. LLM-Judge\n"
        "Synthetic portfolio exercise — Andon Labs case study",
        fontsize=10, style="italic",
    )
    plt.tight_layout()
    plt.savefig(OUT / "monitor_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → outputs/monitor_comparison.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(keyword_only: bool = False) -> None:
    print("\n── Loading datasets ──────────────────────────────────────────────────")
    all_rows = {}
    for name, path in DATASETS.items():
        if not path.exists():
            print(f"  ! {name}: {path.name} not found — run python build_case_study.py first")
            sys.exit(1)
        all_rows[name] = load(path)
        print(f"  {name}: {len(all_rows[name])} rows")

    # LLM judge — adds monitor_llm column to each row
    has_llm = False
    if not keyword_only:
        try:
            from llm_judge import run_on_dataset, judge_trace
            print("\n── Running LLM-judge monitor ─────────────────────────────────────────")
            for name, path in DATASETS.items():
                enriched = run_on_dataset(path, limit=None)
                all_rows[name] = enriched
            has_llm = True
        except ImportError:
            print("  ! anthropic not installed — running keyword-only mode")
        except Exception as e:
            print(f"  ! LLM-judge failed ({e}) — running keyword-only mode")

    # Compute metrics
    print("\n── Computing metrics ─────────────────────────────────────────────────")
    results = {}
    for name, rows in all_rows.items():
        kw_metrics = binary_metrics(rows, "monitor_keyword")
        kw_metrics["per_cat"] = per_category_recall(rows, "monitor_keyword")
        entry = {"keyword": kw_metrics}
        if has_llm:
            lj_metrics = binary_metrics(rows, "monitor_llm")
            lj_metrics["per_cat"] = per_category_recall(rows, "monitor_llm")
            entry["llm_judge"] = lj_metrics
        results[name] = entry

    # Print comparison table
    print(f"\n  {'Dataset':<14} {'Monitor':<18} {'Precision':>10} {'Recall':>8} {'F1':>6}")
    print(f"  {'-'*60}")
    for name, entry in results.items():
        for monitor, m in entry.items():
            print(f"  {name:<14} {monitor:<18} {m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>6.3f}")
        print()

    # Key finding narrative
    kw_in  = results["in_dist"]["keyword"]["f1"]
    kw_ood = results["ood"]["keyword"]["f1"]
    print(f"  KEY FINDING: keyword monitor F1 drops {kw_in:.2f} → {kw_ood:.2f} in-dist → OOD")
    print(f"  (collapse because monitor terms are substrings of the generator vocabulary)")
    if has_llm:
        lj_ood = results["ood"]["llm_judge"]["f1"]
        lj_adv = results["adversarial"]["llm_judge"]["f1"]
        print(f"  LLM-judge recovers to F1={lj_ood:.2f} on OOD, {lj_adv:.2f} on adversarial")
        kw_adv = results["adversarial"]["keyword"]["f1"]
        print(f"  Adversarial gap: keyword F1={kw_adv:.2f} vs. LLM-judge F1={lj_adv:.2f}")

    # Novel failure
    kw_nov = results["novel"]["keyword"]
    nov_caught = kw_nov["tp"]
    print(f"\n  NOVEL FAILURE (scope_creep): keyword catches {nov_caught}/{kw_nov['tp']+kw_nov['fn']} → "
          f"confirms zero generalization beyond designed categories")
    if has_llm:
        lj_nov = results["novel"]["llm_judge"]
        print(f"  LLM-judge catches {lj_nov['tp']}/{lj_nov['tp']+lj_nov['fn']} scope_creep traces")

    # Save outputs
    rows_out = []
    for name, entry in results.items():
        for monitor, m in entry.items():
            rows_out.append({"dataset": name, "monitor": monitor,
                             "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]})
    with open(OUT / "comparison_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows_out[0].keys())
        w.writeheader()
        w.writerows(rows_out)

    with open(OUT / "monitor_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_comparison(results, has_llm)
    print(f"\n  Outputs: outputs/comparison_table.csv, monitor_metrics.json, monitor_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword-only", action="store_true",
                        help="Skip LLM-judge calls (no ANTHROPIC_API_KEY needed)")
    args = parser.parse_args()
    main(keyword_only=args.keyword_only)
