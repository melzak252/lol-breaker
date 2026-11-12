# LoL-Breaker

## What it does

League of Legends pre-match probabilities, tournament research and betting analytics.
Player-history models and saved-market studies support a local application for reviewing odds,
recording manual bets and tracking wallets. **It does not place bets automatically.**

- Sports-only BO1/BO3/BO5 series-win probabilities.
- Model, bookmaker-consensus and selected-ticket EV comparisons.
- Tournament scenarios, map/series props and historical betting simulations.
- FastAPI and React tools for odds, signals, manual decisions and wallets.

## Main model: C20R

**C20R** is a three-seed ensemble using 20-token player histories plus prior regional context.
It predicts whole-series BO1/BO3/BO5 probabilities **without odds**; organization history is optional.
The 20 inputs comprise 19 history slots and 1 learned token.

C20R is a display name, not a new artifact. The [model catalog](docs/MODEL_CATALOG.md#main-model-and-local-policy)
and [serving configuration](conf/base/c0_serving.json) identify the local runtime.
Local configuration is not a deployment or an adoption decision.

## Prediction quality and shrinkage

**All results below are historical development evidence, not prospective confirmation.**
The [factorial audit](reports/c0_regional_factorial_audit_20261022.md) compares the same
**3,083 supported / 3,103 requested 2026 series**.
Lower LL and Brier are better.

| Model | Log loss | Brier | Accuracy | AUC |
|---|---:|---:|---:|---:|
| Historical C0 | 0.536947 | 0.181576 | 71.94% | 0.798037 |
| C20R | 0.533023 | 0.180030 | 72.40% | 0.801652 |

The 20 unsupported regional rows remain missing, not imputed.
The combined paired comparison favors C20R, but required **BO5 and LPL guards failed**;
the study was **not adopted**. Local configuration does not overturn that result.

### Matched bookmaker comparison

Separate cohort: **400 fixtures, 1,496 fixture-book cells, 51,208 quote cases**,
May–October 2026. LOO excludes the focal book and requires at least two other fresh books;
consensus is the equal-weight mean of their fair **logits**, not probabilities.
Quote losses are averaged within fixture/book, then equally across books and fixtures.

| Recipe | Log loss | Brier | ΔLL vs consensus, nominal 95% CI |
|---|---:|---:|---|
| C20R | 0.585025 | 0.202815 | +0.004727 [-0.019232, +0.019035] |
| Other-book consensus | 0.580298 | 0.199347 | reference |
| Hybrid-50 | 0.576162 | 0.198601 | -0.004136 [-0.016001, +0.002928] |

Fixed logit shrinkage: `p_H = expit(0.5*logit(p_sports) + 0.5*logit(p_market))`.
No alpha was selected in this replay. Hybrid-50 has lower point loss, but its primary
paired LL intervals versus consensus **and C20R include zero**; versus C20R:
-0.008863 [-0.016018, +0.003657].

Intervals use 6 scheduled-start monthly blocks, 5,000 draws;
nominal, not a correction for wider adaptive research. Of 100,949 focal cases across
482 fixtures, 49,741 lack eligible consensus. [Coverage and contrasts](docs/PROJECT_REPORT.md#market-consensus-and-fixed-shrinkage).

<details>
<summary>Forecast scores by time to scheduled start</summary>

| Lead hours | Fixture N | Consensus LL | C20R LL | Hybrid-50 LL |
|---|---:|---:|---:|---:|
| [0,2) | 291 | 0.591644 | 0.590224 | 0.585922 |
| [2,6) | 325 | 0.582057 | 0.582640 | 0.576787 |
| [6,12) | 328 | 0.576932 | 0.578070 | 0.571809 |
| [12,24) | 304 | 0.568443 | 0.572756 | 0.564460 |
| [24,48) | 249 | 0.575792 | 0.594537 | 0.578886 |
| [48,96) | 230 | 0.575465 | 0.586390 | 0.574458 |
| [96,inf) | 169 | 0.563134 | 0.565161 | 0.557140 |

Fixture populations overlap across bands; do not sum them.
These use scheduled starts, not certified actual starts or an optimal timing policy.

</details>

## Player vs team Glicko-2

Separate **10,578-series** ratings-only cohort, 2024-01-01–2026-06-14.
Both controls use saved BO-projected probabilities and separate preceding-year calibration.

| Calibrated control | N | Log loss | Brier |
|---|---:|---:|---:|
| Player Glicko-2 | 10,578 | 0.574096 | 0.195826 |
| Team / organization Glicko-2 | 10,578 | 0.607321 | 0.210434 |

Player−organization ΔLL **-0.033225 [-0.045580, -0.024346]**,
nominal 95% CI across 30 monthly blocks. This supports a historical comparison of these
rating controls, **not a cross-cohort C20R/market ranking** or promotion.
[Raw controls, updates and calibration](docs/PROJECT_REPORT.md#player-versus-organization-glicko-2).

## Production-collected betting results

Audited Polish-bookmaker export: **482 fixtures / 100,949 quotes**, with fixture starts
**2026-05-29–2026-10-22**. This is not data through today.
Entry-time consensus covers 400 fixtures / 51,208 quotes with at least **two other fresh books**.
These are saved **C20R counterfactual simulations, not recorded bets or live-issued forecasts**.

### Betting by time to recorded start

Fixed 50/50 sports/other-book **logit** blend; first eligible observed quote per fixture in each
half-open interval, never the future-best price. Each wallet starts with PLN10,000, uses quarter
Kelly, PLN2–150 stakes and reserves open stakes. The 0% and 12% cases requalify independently;
0% retains the same Polish prices, **not actual UK or Pinnacle prices**.

| Lead hours | LOO-ready fixtures | 0% bets | 0% P&L, PLN | 0% ROI | 12% bets | 12% P&L, PLN | 12% ROI |
|---|---:|---:|---:|---:|---:|---:|---:|
| [0,2) | 291 | 31 | +83.50 | +2.08% | 3 | +50.20 | +13.92% |
| [2,6) | 325 | 39 | +1,289.76 | +24.67% | 5 | +74.10 | +12.24% |
| [6,12) | 328 | 49 | +962.59 | +14.37% | 5 | +73.11 | +11.04% |
| [12,24) | 304 | 44 | -515.87 | -8.42% | 8 | -1,055.75 | -100.00% |
| [24,48) | 249 | 42 | +0.87 | +0.02% | 6 | -523.74 | -61.44% |
| Full [0,48) | 370 | 82 | +1,728.80 | +15.40% | 16 | -397.69 | -18.46% |

ROI is profit/stake turnover, not bankroll growth. All displayed portfolios are fully settled;
the same fixture can appear in several wallets, so **do not sum rows or choose a winning band**.
The zero-tax full-window nominal 95% ROI interval is **[-4.22%, +48.59%]**.
The 12% interval is withheld: 8/5,000 monthly resamples have no bets.
Six start-month blocks include zero-action months; this is not independent edge confirmation.
Rating-disagreement and contemporaneous drift safeguards remain **not assessed**.
[Wins, turnover, all intervals and exact frozen gates](docs/PROJECT_REPORT.md#production-collected-interval-wallets).

### Production closing-price proxies

**True closing CLV is unavailable.** Roughly two-hour collection does not guarantee a quote at
the actual close or an executable fill. Below, every other-book contributor is within **five
minutes before conservative recorded start**, excludes the selected book, and at least one is
strictly later than entry. Missing references remain missing, not zero.

| Entry lead hours | 0% covered / bets | 0% net fair-price proxy | 12% covered / bets | 12% net fair-price proxy |
|---|---:|---:|---:|---:|
| [0,2) | 7/31 | -1.16% | 1/3 | -8.28% |
| [2,6) | 9/39 | +0.71% | 1/5 | -8.28% |
| [6,12) | 12/49 | -1.81% | 1/5 | -8.28% |
| [12,24) | 11/44 | +3.02% | 4/8 | +0.12% |
| [24,48) | 12/42 | +1.54% | 1/6 | -13.24% |
| Full [0,48) | 19/82 | +1.82% | 4/16 | -1.12% |

Equal-ticket proxy on **covered selections only**:
`(1-tax)*entry_odds*later_other_book_fair_probability - 1`.
It is not whole-portfolio CLV, realized ROI or guaranteed edge; the reference's age is measured
before start, not after entry. Later prices are assessment references, not entry-time shrinkage.
[Same-book raw price moves, gross/net separation, 60-minute and uncapped-age sensitivities](docs/PROJECT_REPORT.md#production-recorded-price-references).

## Historical betting results

Annual year-eligible fresh20+regional forecasts, OddsPortal 2021–2026 first-observed study;
mapped target dates **2021-01-09–2026-09-07**.
Here shrinkage uses the **focal book's own no-vig price**, not independent LOO consensus.

| Portfolio | Tax | Known / selected | Mean EV | Realized yield, 95% CI | EV−yield, 95% CI (pp) |
|---|---:|---:|---:|---|---|
| Sports only | 0% | 4,482 / 4,484 | 19.72% | 6.00% [+2.54%, +9.34%] | +13.72 [+10.36, +17.26] |
| Focal-book 50/50 | 0% | 1,967 / 1,968 | 12.90% | 13.69% [+8.42%, +18.94%] | -0.79 [-6.06, +4.40] |
| Focal-book 50/50, requalified | 12% | 315 / 315 | 13.64% | 22.81% [+9.31%, +36.42%] | -9.16 [-22.93, +4.30] |

Each row is independently qualified: EV≥5%, a 2.5pp probability stress haircut,
odds≤4, one-unit stakes and at most one ticket per fixture. Yield is profit/stake turnover,
not bankroll growth; this is **not the complete application gate stack**.
Intervals retain all 70 preselection UTC months, January 2021–October 2026, including zero-action months.

The 12% portfolio is **reselected**, not a tax-only improvement or a causal shrinkage estimate.
Tax reduces the full winning payout: `O*(1-tax)-1`.
Sports only has 2 unresolved outcomes, zero-tax shrinkage 1, taxed requalification 0;
unknowns leave whole-portfolio returns undefined. Portfolios overlap and are not additive.
The [fixed-original-ticket tax audit](docs/PROJECT_REPORT.md#fixed-original-ticket-tax-audit) remains available separately.

EV−yield is selected-ticket forecast error, not an identified optimizer's-curse component.
Neither shrinkage tax regime establishes the project's **±2pp EV-reliability target**,
including the strict (5%,10%] band. [Same-ticket and near-band evidence](docs/PROJECT_REPORT.md#near-band-reliability).

## CLV and execution evidence

Separate **2021–2026 OddsPortal** evidence, not the production export above.
**Authenticated closing prices and executable fills are unavailable.** These are same-book,
strictly later recorded pre-start prices, paired separately to each portfolio's actual entries.
The table uses **matched-known tickets**, not each full portfolio.

| Scenario | Matched known / known selected | Raw later-price move | Net no-vig proxy, 95% CI | Median later quote lead |
|---|---:|---:|---|---:|
| 0% focal-book 50/50 | 1,708 / 1,967 | +12.68% | +4.82% [+3.53%, +6.03%] | 1.71h |
| 12% requalified | 268 / 315 | +24.68% | +2.09% [-1.73%, +5.99%] | 1.48h |

Raw move: `O_entry/O_later_selected - 1`.
Net proxy: `(1-tax)*O_entry*q_later_selected - 1`, recomputing no-vig q from both later prices.
Finance's same 70-month bootstrap applies; raw movement does not change with tax.

Price-only coverage is 1,709/1,968 and 268/315, respectively.
Only 111 and 20 matched-known later quotes are within five minutes of recorded start.
These proxies are **not true CLV or guaranteed edge**; yield minus proxy is not identified pure sporting luck.
[Coverage, absences and matched-ticket returns](docs/PROJECT_REPORT.md#clv-and-execution-evidence).

## Reproduce and run

With the [exact local artifact prerequisites](docs/REPRODUCING.md#required-artifacts-and-evidence-checking) installed,
run from the repository root, choosing an **unused output directory**:

```bash
.venv/bin/python scripts/reproduce_project_results.py --output-dir /tmp/c20r-trader-readme-replay
```

This scores saved artifacts; no fitting, inference, collection, database or network access.
A clean checkout does **not** include the ignored data/model inputs.
Full-precision results, domain-specific seeds and hashes: [aggregate snapshot](docs/results/project_summary.json).
Python 3.11+ is required; the [guide](docs/REPRODUCING.md) records the exercised environment.

For the local application, use the [application guide](betting_app/README.md).
Review database, secrets and scheduler precautions first; default Compose is not a disposable offline setup.

## Project and documentation

| Area | Start here |
|---|---|
| Detailed evidence and limitations | [Project report](docs/PROJECT_REPORT.md) |
| Replay and artifact prerequisites | [Reproduction guide](docs/REPRODUCING.md) |
| Models and local serving identity | [Model catalog](docs/MODEL_CATALOG.md) |
| Architecture and entry points | [Architecture](docs/PROJECT_ARCHITECTURE.md) |
| Data, research history and scripts | [Documentation index](docs/README.md) |
| Application and manual workflows | [Application guide](betting_app/README.md) |

## License and disclaimer

Released under the [MIT License](LICENSE). Research and education only, not financial advice.
Historical simulations do not guarantee future returns.

LoL-Breaker is not endorsed by Riot Games. League of Legends and Riot Games are trademarks
or registered trademarks of Riot Games, Inc.
