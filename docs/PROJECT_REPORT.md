# Project report

LoL-Breaker is a League of Legends match-forecasting and manual-analysis project: historical data pipelines support sports probabilities, market comparisons and evaluation; a local application helps review matches and offers but does **not** place bets automatically. This report highlights the saved results that can currently be replayed and separates them from what the evidence does not establish.

The project also contains specialized tournament-event and player/champion/team-statistics capabilities, including distributional prop-model work. See the [model catalog](MODEL_CATALOG.md) and [architecture overview](PROJECT_ARCHITECTURE.md) for capability boundaries; this report makes no new prop or tournament performance claims.

## Cohorts and current model

**C20R** is a documentation-only name for the configured serving model `c0-native-fresh20-2026-10-21-v1-regional-v1`: a three-seed player-history ensemble with 19 history input slots plus a learned token and a regional adjustment. Its historical feature recipe uses causal player/map history across 90-, 180- and 365-day windows and preserves optional organization semantics. The alias does not change runtime configuration, the `Causal-C0` model name, or wire/artifact IDs; historical C0 is a distinct model.

The saved assessment spans 3,103 unique 2026 matches and is development-exposed, not a prospective or independent future holdout. The frozen study partition has 36,205 pre-July-2025 training rows, 1,328 2025H2 calibration rows and this 3,103-row 2026 assessment cohort. Regional scores exist for 3,083/3,103 rows (99.36%); the other 20 are missing and excluded, not imputed. The table shows log loss, Brier score and ECE (lower is better); C20R is shown only on its supported cohort.

| Model | Cohort | N | Log loss | Brier | ECE-10 |
|---|---|---:|---:|---:|---:|
| Historical C0 | full assessment | 3,103 | 0.53545 | 0.18103 | 2.29% |
| Fresh20, no regional adjustment | full assessment | 3,103 | 0.53393 | 0.18038 | 2.08% |
| Annual A0 | full assessment | 3,103 | 0.53676 | 0.18157 | 2.20% |
| Historical C0 | regional-supported common cohort | 3,083 | 0.53695 | 0.18158 | 2.33% |
| Fresh20, no regional adjustment | regional-supported common cohort | 3,083 | 0.53539 | 0.18091 | 2.12% |
| Annual A0 | regional-supported common cohort | 3,083 | 0.53821 | 0.18211 | 2.26% |
| **C20R** | **regional-supported common cohort** | **3,083** | **0.53302** | **0.18003** | **1.93%** |

These are historical assessment scores, not proof of deployment profitability or future performance. The fresh20+regional factorial assessment failed the frozen BO5 and LPL upper-bound replacement guards against both C0 and A0 (each is judged against a +0.005 margin); no challenger cleared that gate. This does not establish a global best or adoption. See the dated [research record](RESEARCH.md) for the full guard context and historical negative results.

## Flat-stake endpoint diagnostics

The saved OddsPortal evaluation spans 2021–2026 and uses first-observed/eligible recorded quotes, not authenticated market opening prices. Policies use a 1-unit flat stake, a 5% minimum net expected value, DRO ε=0.025, maximum odds 4.0 and at most one selection per canonical fixture. Same-book 50/50 logit blends model probability with that bookmaker's multiplicative no-vig probability; it is **not** the application's hybrid, which uses at least two other fresh books. These are endpoint-policy diagnostics, not a replay of the complete application gate stack.

| Diagnostic | Selected | Known / unresolved | Known-label profit | Known-label stake-turnover ROI | 95% month-bootstrap interval |
|---|---:|---:|---:|---:|---:|
| Sports only, 0% tax | 4,484 | 4,482 / 2 | +269.08 units | +6.0036% | [+2.5435%, +9.3413%] |
| Same-book 50/50 logit, 0% tax | 1,968 | 1,967 / 1 | +269.27 units | +13.6894% | [+8.4207%, +18.9447%] |
| Same-book fixed original tickets, 12% tax | 1,968 | 1,967 / 1 | +0.9176 units | +0.04665% known-only | [−4.5898%, +4.6714%] known-only |
| Same-book policy reselected at 12% tax | 315 | 315 / 0 | +71.8392 units | +22.8061% | [+9.3066%, +36.4237%] |

**Intervals.** Estimates use 5,000 resamples of the complete 70-month preselection UTC calendar, seed `20261111`. Where a selected cohort has unresolved labels, the interval is known-label-only and whole-portfolio P&L and ROI are undefined. For fixed 12%-tax tickets, the unresolved ticket is retained among all 1,968 selections and excluded only from the known-label calculation. The 315 requalified 12%-tax selections all have known outcomes, so that interval covers its full selected cohort. ROI is return per known-label stake turnover, **not bankroll growth**. Portfolios overlap and are not additive.

A later-quote comparison pairs 1,709 of 1,968 same-book selections; 1,708 have known outcomes. Recomputing fair probability from paired raw later two-way odds gives mean entry-model EV +12.9732%, later fair-price proxy +4.8198% and realized return +14.3677% on matched known rows. Recorded later quotes are not true closing-line value, executable prices or evidence of fills. Per-bet residuals are descriptive, not pure sports noise.

A separate app-style factorial assessment is a counterweight to standalone endpoint results: fresh20+regional ASSESS timing wallets were 2 / +173.97 PLN first-ready, 8 / −304.10 PLN quote-triggered, 2 / −269.77 PLN UTC-even and 6 / −64.62 PLN UTC-odd. These are sparse separate timing scenarios, not additive returns or a selected timing policy. The full frozen policy study selected zero primary configurations because TRAIN support failed the minimum. These historical results do not validate an executable or profitable deployment; see [RESEARCH.md](RESEARCH.md).

## Reproduction

The [reproduction guide](REPRODUCING.md) describes the exact saved-ledger command, prerequisites and environment limits. The checked-in [aggregate snapshot](results/project_summary.json) contains cohort metrics, financial summaries, bootstrap metadata, dependency versions and file hashes—not tickets, raw data, weights or model outputs. Required saved artifacts are ignored/local, so a clean clone alone cannot replay these aggregates. The replay fails if required inputs are missing; it does not substitute the snapshot.

Raw-data acquisition, neural training and creation of forecasts are separate from this summary replay and require original inputs and model artifacts. No clean-clone reproduction of neural fits or private/local data is claimed.