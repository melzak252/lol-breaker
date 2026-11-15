# LoL-Breaker

LoL-Breaker is a League of Legends forecasting and research project: causal pre-match series probabilities, tournament simulation, draft and prop-distribution studies, and a FastAPI/React application for odds review and manual bet tracking. **It does not place bets automatically.**

> **Evidence status:** the locally configured C20R sports model is a historical-development candidate, not an adopted or prospectively certified model. The project has not established general bookmaker superiority, a live betting edge, or reliable ±2 percentage-point EV calibration. Local configuration is not deployment.

## What the project contains

- **Pre-match forecasting:** sports-only BO1/BO3/BO5 series probabilities from player history and regional context.
- **Market review:** compare forecasts with timestamped or archived bookmaker prices; fixed 50/50 logit shrinkage is evaluated as a separate candidate.
- **Manual betting records:** review offers and track manually entered bets, wallet activity, and settlements.
- **Tournament simulation:** exact, hybrid, and Monte Carlo methods support different tournament formats; match accuracy does not certify tournament distributions.
- **Draft research:** predict champion picks and post-draft outcomes. Post-draft forecasts require locked-in draft information; they are not pre-match forecasts.
- **Standalone props research:** historical distributions for Map 1 kills, Map 1 duration, and series map count. These are not live feature adapters or joint/parlay probability models.

Research results are kept separate from application-serving identities. C20R is a locally configured research identity, not a claim about a deployed model.

## Match-model evidence

C20R is the locally configured `fresh20 + regional` identity, version `c0-native-fresh20-2026-10-21-v1-regional-v1`. The table scores a historical 2026 assessment on **3,083 regional-supported series**; 20 of 3,103 assessment rows lack regional support and are not imputed.

| Model | Series | Log loss ↓ | Brier ↓ | ECE-10 |
|---|---:|---:|---:|---:|
| Annual A0 | 3,083 | 0.5382 | 0.1821 | 2.3% |
| Historical C0 | 3,083 | 0.5369 | 0.1816 | 2.3% |
| Fresh20 without regional | 3,083 | 0.5354 | 0.1809 | 2.1% |
| **C20R** (fresh20 + regional) | **3,083** | **0.5330** | **0.1800** | **1.9%** |

This is an exposed historical development comparison. C20R failed required BO5/LPL replacement guards and is **not adopted**; lower aggregate loss does not establish a general improvement.

## Model versus bookmaker prices

Matched fixtures, labels, and orientations. Negative ΔLL favors the sports model. “First observed” and “last retained” are recorded endpoints—not authenticated market open/close prices.

| Source | Endpoint | Fixtures | Sports LL | Market LL | Sports − market ΔLL CI |
|---|---|---:|---:|---:|---:|
| OddsPortal | First observed | 9,132 | 0.5834 | 0.5870 | [-0.0085, +0.0014] |
| OddsPortal | Last retained | 9,805 | 0.5826 | 0.5779 | [+0.0009, +0.0085] |
| Pinnacle | First observed | 1,040 | 0.5588 | 0.5789 | [-0.0424, +0.0013] |
| Pinnacle | Last retained | 993 | 0.5606 | 0.5487 | [-0.0171, +0.0312] |

OddsPortal intervals are nominal 95%; Pinnacle intervals are 98.75% family-adjusted for four primary contrasts. These are historical endpoint comparisons, not authenticated open/close evaluations.

### Bookmaker scores by lead time

Polish leave-one-out (LOO) consensus uses at least two distinct other books. Each row is a separate historical lead-time cohort; differences across rows are descriptive, not a causal timing effect.

| Time before start | Fixtures | LOO consensus LL | C20R LL | Hybrid-50 LL |
|---|---:|---:|---:|---:|
| 0–2h | 291 | 0.5916 | 0.5902 | 0.5859 |
| 2–6h | 325 | 0.5821 | 0.5826 | 0.5768 |
| 6–12h | 328 | 0.5769 | 0.5781 | 0.5718 |
| 12–24h | 304 | 0.5684 | 0.5728 | 0.5645 |
| 24–48h | 249 | 0.5758 | 0.5945 | 0.5789 |
| 48–96h | 230 | 0.5755 | 0.5864 | 0.5745 |
| 96h+ | 169 | 0.5631 | 0.5652 | 0.5571 |

Across the full 400-fixture panel, fixed 50/50 logit shrinkage scores **0.5762 LL** versus **0.5803** for LOO consensus. The fixture-weighted 95% ΔLL interval is **[-0.0160, +0.0029]** and crosses zero; the equal-month interval is **[-0.0173, -0.0021]**. This is mixed historical evidence, not an adopted blend or a profitability result.

### Recorded-price CLV diagnostics

In 12,000 sampled timestamp checks, the fixed policy produced **451 recorded recommendations across 74 fixtures**. Recorded-price references on those recommendations:

| Reference | Covered | Mean | 95% interval |
|---|---:|---:|---:|
| Same-book recorded price move | 421 / 451 | +6.4% | [+5.5%, +7.6%] |
| Other-book fair-price reference | 451 / 451 | -0.3% | [-1.5%, +1.1%] |

These are conditional historical diagnostics, not verified execution. References use the latest recorded pre-start quote; the same-entry quote is allowed. They are **not authenticated official closes or actual fills**, and repeated recommendations share fixtures.

## Repository map

| Path | Purpose |
|---|---|
| `src/` | Research models, causal feature banks, tournament engines, and analysis |
| `betting_app/` | FastAPI app, model worker, persistence, quote gates, wallets, scrapers, scheduler, and tests |
| `client/` | React interface |
| `scripts/` | Research CLIs, saved-evidence replay, and verification tools |
| `conf/` | Serving and benchmark contracts |
| `data/` | Local study inputs, model artifacts, outputs, and receipts; usually gitignored |
| `docs/` and `reports/` | Research status, quantitative summaries, reproducibility requirements, and dated evidence |

The DVC pipeline is a retained historical workflow; it is not the entry point for every current study.

## Recreate the numbers above

The numeric scorecards above come from `scripts/reproduce_project_results.py`. It reads bound local artifacts, checks their hashes and saved reference results, and writes a new `project_summary.json` plus a replay receipt. It does **not** fit models, run neural inference, scrape, access a database, or use the network.

```bash
.venv/bin/python scripts/reproduce_project_results.py \
  --output-dir /tmp/lol-breaker-readme-replay
```

The README tables use `model_scores.regional_supported_3083`, `bookmaker_comparisons.{oddsportal,pinnacle}.endpoints`, `bookmaker_comparisons.polish.by_lead_bin`, and `production_opportunities.snapshot_weighted.recorded_close_clv` in the replay output. Reproduction requires bound local study artifacts under `data/`; they are not all in Git or downloaded by the script. `docs/results/project_summary.json` is the checked-in reference snapshot, not a fallback input. Reproduction checks saved-output consistency; it does not authenticate source clocks, forecast issuance, market closes, fills, or prospective performance.

MIT licensed. Research and education only; not financial advice. LoL-Breaker is not endorsed by Riot Games.
