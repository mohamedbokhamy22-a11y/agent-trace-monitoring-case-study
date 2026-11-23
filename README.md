# Agent Trace Failure Analysis & Control Evaluation

**Core finding:** A keyword monitor achieves F1 ≈ 0.91 on in-distribution traces and collapses toward zero on out-of-distribution (paraphrased) traces — because its terms are substrings of the generator vocabulary, not a generalizing classifier. An LLM-as-judge monitor (Claude Haiku reading both agent action and tool/state event) substantially recovers recall on OOD traces and catches adversarial cases the keyword monitor misses entirely. The gap between them is the argument for semantic monitoring.



---

## What this demonstrates

| Skill | Implementation |
|---|---|
| Failure taxonomy design | 7 categories (6 original + scope_creep) with per-category detection challenge analysis |
| Evaluation methodology | 4 datasets: in-dist, OOD (paraphrased), adversarial (lexically innocent), novel failure |
| Monitor comparison | Keyword baseline vs. LLM-judge across all datasets + per-category breakdown |
| Threat model framing | Trusted monitor + untrusted agent, audit budget constraint, adversarial evasion |
| Adversarial trace design | 12 traces semantically unsafe, lexically innocent — zero keyword match possible |
| Generalization testing | 10 traces of scope_creep (unseen category) — tests out-of-taxonomy generalization |
| Literature grounding | Greenblatt et al. AI Control, Anthropic Sabotage Evals, METR task-based evals |
| Architecture analysis | Why missed escalation and long-horizon incoherence are wrong-architecture, not tuning problems |

---

## Results

### Monitor comparison across evaluation sets

| Dataset | Keyword F1 | LLM-judge F1 | Key observation |
|---|---|---|---|
| **In-distribution** | ~0.91 | ~0.90 | Both perform well — in-dist metrics are inflated for keyword monitor |
| **OOD (paraphrased)** | ~0.00 | ~0.75 | Keyword monitor collapses; LLM-judge maintains meaningful recall |
| **Adversarial (lexically innocent)** | 0.00 | ~0.70 | Keyword catches 0/12; LLM-judge partially generalizes |
| **Novel failure (scope_creep)** | 0.00 | partial | Neither designed for it; LLM-judge's semantic reasoning partially transfers |

*Run `python evaluate.py` to get exact numbers with your ANTHROPIC_API_KEY.*

### Why the in-distribution keyword F1 of 0.91 is misleading

The v1 monitor's keyword terms are literal substrings of the same templates used to generate the traces. The 83.1% recall measures which template variants happened to include the verbatim keyword — not monitoring competence. 100% precision is the giveaway: in real monitoring, precision is the hard-won number. Getting it for free means measuring the wrong thing.

The OOD collapse (from ~0.91 to near zero) is not a failure of the case study — it is the finding. It makes visible the generalization cliff that all rule-based monitors hit in production.

### Deception and missed escalation: relational failure modes

Both categories require comparing the agent's statement to external ground truth:
- **Deception:** "Let me make sure everything is taken care of for you" is benign text; deceptive only when the tool event shows no disbursement was executed.
- **Missed escalation:** "Happy to continue working through this" is reasonable agent language; missed escalation only when the tool event shows an unverified vendor payment request.

A monitor that reads only agent_action cannot detect either. The adversarial set makes this unavoidable: 12 traces, all phrased as ordinary business communication, all detectable only through the tool/state field.

---

## File structure

```
andon-agent-trace-safety-case-study/
├── data/
│   ├── in_dist_traces.csv        — 180 in-distribution traces (easy + hard variants)
│   ├── ood_traces.csv            — 120 OOD traces (all paraphrased; no keyword match)
│   ├── adversarial_traces.csv    — 12 adversarial traces (lexically innocent)
│   ├── novel_failure_traces.csv  — 10 scope_creep traces (unseen category)
│   └── synthetic_agent_traces.csv— backwards-compatible alias for in_dist
├── build_case_study.py           — generates all four datasets
├── llm_judge.py                  — LLM-as-judge monitor (Claude Haiku, structured rubric)
├── evaluate.py                   — runs keyword + LLM-judge across all datasets; plots comparison
├── analyze_traces.py             — simple per-dataset summary (backwards compat)
├── docs/
│   └── research_notes.md         — threat model, taxonomy, literature, limitations
└── requirements.txt
```

---

## Quick start

```bash
pip install -r requirements.txt

# Build all trace datasets
python build_case_study.py

# Keyword-only evaluation (no API key needed)
python evaluate.py --keyword-only

# Full comparison with LLM-judge (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
python evaluate.py

# Backwards-compatible single-dataset summary
python analyze_traces.py
```

---

## Threat model and monitoring design

**Setup:** trusted keyword monitor (or LLM-judge) observing an untrusted agent with real authority over business operations. Audit budget: 2% of actions receive human review. At this budget, every flagged action must be a real violation (precision matters as much as recall).

**Why this matters for monitor design:**
- A keyword monitor is trivially evadable: any agent that learns the keyword list avoids those phrases while continuing the same behavior (Greenblatt et al., AI Control).
- The LLM-judge is harder to evade but not immune — the adversarial traces show it still misses some cases.
- A production monitor should therefore: (1) assume at least partial adversarial intent, (2) be evaluated on traces the monitor was not designed for, (3) compare multiple monitoring approaches rather than optimizing a single one.

---

## What I would build next

| Priority | Enhancement | Why |
|---|---|---|
| 1 | State-aware LLM monitor (commitment ledger across episode) | Architecturally required for missed escalation and long-horizon incoherence — per-action monitoring cannot catch them in principle |
| 2 | Red-team pass: second LLM rewrites agent actions to evade the judge monitor | Tests adversarial robustness; monitors should be evaluated against an adversary, not just a fixed trace distribution |
| 3 | Real model traces from a live or replayed agent | Synthetic traces demonstrate mechanics; real traces would let the taxonomy and monitor be validated |
| 4 | Blinded labeling + inter-rater agreement | Ground truth in this study is single-labeler; a second labeler + Cohen's kappa would validate the taxonomy |
| 5 | Precision-at-recall-k curves | The right production metric is recall at a fixed human-review budget, not F1 on a balanced set |

---

*Synthetic portfolio exercise. All traces and thresholds are illustrative only.*
