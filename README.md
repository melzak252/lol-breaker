# LoL-Breaker

Pre-match League of Legends series-win probabilities (**C20R**, sports-only, zero odds), market-blended **Hybrid-50**, and a FastAPI/React app for odds review and manual bet tracking. **Does not place bets automatically.**

**Bottom line:** Historical predictive signal, **no established general bookmaker superiority or confirmed live edge**. All 2021–2026 results are development-exposed counterfactuals.

## Standalone model (`C20R`)

Three-seed, 20-token player-history ensemble (19 history slots + 1 learned token) with prior regional context, trained chronologically (`release < match_date`, no odds). On 3,083 supported 2026 series:

| Model | N | Log Loss | ECE-10 |
|---|---:|---:|---:|
| Annual A0 (ref) | 3,083 | 0.5382 | 2.3% |
| Historical C0 (17T) | 3,083 | 0.5369 | 2.3% |
| Fresh20 (no regional) | 3,083 | 0.5354 | 2.1% |
| **C20R** (`fresh20+regional`) | **3,083** | **0.5330** | **1.9%** |

C20R is locally configured ([serving config](conf/base/c0_serving.json), [model catalog](docs/MODEL_CATALOG.md#main-model-and-local-policy)), **not promoted** (failed BO5/LPL replacement guards).

## Model vs. bookmaker probabilities

Matched fixtures, labels, and orientations; de-vigged two-sided markets. Lower log loss (LL) is better; negative $\Delta\text{LL}$ favors the sports model.

| Dataset / endpoint | N | Sports LL | Market LL | $\Delta\text{LL}$ CI |
|---|---:|---:|---:|---|
| OddsPortal, first observed | 9,132 | 0.5834 | 0.5870 | [-0.0085, +0.0014] |
| OddsPortal, last retained | 9,805 | 0.5826 | 0.5779 | [+0.0009, +0.0085] |
| Pinnacle, first observed | 1,040 | 0.5588 | 0.5789 | [-0.0424, +0.0013] |
| Pinnacle, last retained | 993 | 0.5606 | 0.5487 | [-0.0171, +0.0312] |
| Polish entry LOO (≥2 peers, ≤5m) | 400 | 0.5850 | 0.5803 | [-0.0192, +0.0190] |
| **Hybrid-50** (`0.5·C20R + 0.5·LOO`) | 400 | **0.5762** | 0.5803 | [-0.0160, +0.0029] |

95% monthly-block bootstrap CIs (98.75% four-contrast family for Pinnacle). Sources overlap; "first/last" are earliest/latest recorded quotes, not certified open/close. Full coverage, Brier, and lead-time profiles: [detailed report](docs/PROJECT_REPORT.md#three-market-datasets).

## Opportunities & recorded-close CLV

Zero-tax Polish-panel counterfactuals (no executed fills; DRO is a stress gate, not a calibrated bound):

| View | Picks (Fixtures) | Unit Yield 95% CI | Fair LOO CLV 95% CI |
|---|---:|---|---|
| First eligible `[0,48)h` | 82 (82) | [-6.6%, +42.4%] | [-0.3%, +6.2%] |
| Snapshot checks (`12k` across 471) | 451 (74) | [-22.0%, +17.7%] | [-1.5%, +1.1%] |

Closing references are latest recorded pre-start quotes, not authenticated closes. Full precision, EV, Brier, tax, wallet, Glicko-2, and calibration audits: [detailed report](docs/PROJECT_REPORT.md).

## Reproduce

With [bound local artifacts](docs/REPRODUCING.md#required-artifacts-and-evidence-checking) installed:

```bash
.venv/bin/python scripts/reproduce_project_results.py --output-dir /tmp/c20r-readme-evidence-replay
```

Saved-only scoring (no fitting, DB, or network). [Full-precision snapshot](docs/results/project_summary.json).

## Docs & license

[Detailed report](docs/PROJECT_REPORT.md) · [Reproduction](docs/REPRODUCING.md) · [Model catalog](docs/MODEL_CATALOG.md) · [Architecture](docs/PROJECT_ARCHITECTURE.md) · [App guide](betting_app/README.md) · [Research index](docs/README.md)

[MIT License](LICENSE). Research and education only, not financial advice. LoL-Breaker is not endorsed by Riot Games.
