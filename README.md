# LoL-Breaker

Pre-match League of Legends series-win probabilities (**C20R**, sports-only, zero odds), market-blended **Hybrid-50**, and a FastAPI/React app for odds review and manual bet tracking. **Does not place bets automatically.**

**Bottom line:** Historical predictive signal, **no established general bookmaker superiority or confirmed live edge**. All 2021–2026 results are development-exposed counterfactuals.

## Standalone model (`C20R`)

Three-seed, 20-token player-history ensemble (19 history slots + 1 learned token) with prior regional context, trained chronologically (`release < match_date`, no odds). On 3,083 supported 2026 series:

| Model | N | Log Loss | Brier | ECE-10 |
|---|---:|---:|---:|---:|
| Annual A0 (ref) | 3,083 | 0.538211 | 0.182107 | 2.26% |
| Historical C0 (17T) | 3,083 | 0.536947 | 0.181576 | 2.33% |
| Fresh20 (no regional) | 3,083 | 0.535391 | 0.180906 | 2.12% |
| **C20R** (`fresh20+regional`) | **3,083** | **0.533023** | **0.180030** | **1.93%** |

C20R is locally configured ([serving config](conf/base/c0_serving.json), [model catalog](docs/MODEL_CATALOG.md#main-model-and-local-policy)), **not promoted** (failed BO5/LPL replacement guards).

## Model vs. bookmaker probabilities

Matched fixtures, labels, and orientations; de-vigged two-sided markets. Lower LL/Brier is better; negative $\Delta\text{LL}$ favors the sports model.

| Dataset / endpoint | Fixtures | Sports LL / Brier | Market LL / Brier | $\Delta\text{LL}$ vs. market [CI] |
|---|---:|---:|---:|---|
| OddsPortal, first observed | 9,132 | 0.583438 / 0.199523 | 0.586991 / 0.200994 | -0.003554 [-0.008530, +0.001372] |
| OddsPortal, last retained | 9,805 | 0.582592 / 0.199201 | 0.577945 / 0.197117 | +0.004647 [+0.000931, +0.008507] |
| Pinnacle, first observed | 1,040 | 0.558802 / 0.189400 | 0.578937 / 0.197194 | -0.020135 [-0.042401, +0.001336] |
| Pinnacle, last retained | 993 | 0.560559 / 0.190288 | 0.548674 / 0.184349 | +0.011885 [-0.017119, +0.031245] |
| Polish entry LOO (≥2 peers, ≤5m) | 400 | 0.585025 / 0.202815 | 0.580298 / 0.199347 | +0.004727 [-0.019232, +0.019035] |
| **Hybrid-50** (`0.5·logit(C20R) + 0.5·logit(LOO)`) | 400 | **0.576162 / 0.198601** | 0.580298 / 0.199347 | -0.004136 [-0.016001, +0.002928] |

95% monthly-block bootstrap CIs (98.75% four-contrast family for Pinnacle). Sources overlap; "first/last" are earliest/latest recorded quotes, not certified open/close. Full coverage and lead-time profiles: [detailed report](docs/PROJECT_REPORT.md#three-market-datasets).

## Opportunities & recorded-close CLV

Zero-tax Polish-panel counterfactuals (no executed fills; DRO is a stress gate, not a calibrated bound):

| View | Picks (Fixtures) | Expected EV | Unit Yield [95% CI] | Fair LOO CLV [95% CI] |
|---|---:|---:|---|---|
| First eligible `[0,48)h` | 82 (82) | +10.54% | +11.76% [-6.63%, +42.41%] | +2.17% [-0.33%, +6.17%] |
| Snapshot checks (`12k` across 471) | 451 (74) | +11.01% | -4.99% [-21.96%, +17.69%] | -0.30% [-1.46%, +1.08%] |

Same-book raw shortening (`+8.32%` / `+6.43%`) does not establish positive fair value; closing references are latest recorded pre-start quotes, not authenticated closes. Full tax, wallet, Glicko-2, and calibration audits: [detailed report](docs/PROJECT_REPORT.md).

## Reproduce

With [bound local artifacts](docs/REPRODUCING.md#required-artifacts-and-evidence-checking) installed:

```bash
.venv/bin/python scripts/reproduce_project_results.py --output-dir /tmp/c20r-readme-evidence-replay
```

Saved-only scoring (no fitting, DB, or network). [Full-precision snapshot](docs/results/project_summary.json).

## Docs & license

[Detailed report](docs/PROJECT_REPORT.md) · [Reproduction](docs/REPRODUCING.md) · [Model catalog](docs/MODEL_CATALOG.md) · [Architecture](docs/PROJECT_ARCHITECTURE.md) · [App guide](betting_app/README.md) · [Research index](docs/README.md)

[MIT License](LICENSE). Research and education only, not financial advice. LoL-Breaker is not endorsed by Riot Games.
