# LoL-Breaker

Player-based League of Legends probability modeling, bookmaker benchmarking and manual betting analysis.
The project asks whether sports information adds value beyond market prices—not whether a selected
historical backtest can look profitable. **It does not place bets automatically.**

- Sports-only pre-match BO1/BO3/BO5 series-win probabilities.
- Separately identified market-assisted forecasts and odds-review tools.
- Tournament/prop research; FastAPI and React tools for manual bets and wallets.

## Current evidence

**Historical predictive signal; no established general bookmaker superiority or confirmed live betting edge.**
The raw model has better point scores at some first-observed prices, but not consistently at later
prices. Fixed shrinkage improves the Polish-panel point scores; its primary improvement intervals
include zero. All results below are development-exposed historical evidence, not prospective confirmation.

## Model

**C20R** is a three-seed, player-history ensemble with prior regional context: 19 history input slots
plus one learned token. It predicts whole-series probabilities **without odds**; organization history
is optional. The multi-year replay uses year-eligible annual counterparts, not a 2026 model backcast.

The [model catalog](docs/MODEL_CATALOG.md#main-model-and-local-policy) and
[configuration](conf/base/c0_serving.json) identify the local artifact; configuration is not deployment.
The standalone C20R study improved aggregate scores but **failed BO5/LPL replacement guards**;
that study remains not adopted. [Standalone benchmark](docs/PROJECT_REPORT.md#standalone-sports-model).

## Three market datasets

| Dataset | Fixture-date coverage | Source / usable comparison |
|---|---|---|
| OddsPortal archive | 2021-01-09–2026-09-07 mapped | 13,473 source events; 125,322 endpoint quote records; 9,132 first / 9,805 last scored fixtures |
| Pinnacle history | 2026-01-14–2026-09-28 source | 2,182 source fixtures; 1,040 first / 993 last model-paired fixtures |
| Production-collected Polish prices | 2026-05-29–2026-10-22 | 482 model-scorable fixtures / 100,949 quote records; 400 fixtures / 51,208 eligible entry-LOO records |

These sources can overlap; they are **not three independent replications**. The Polish panel is a
bound analysis subset, not every raw fixture ever collected or data through today. Quoted records
are not independent matches. Pinnacle model-paired fixtures end on 2026-09-07.
[Coverage, exclusions and matching](docs/PROJECT_REPORT.md#three-market-datasets).

## Model versus bookmaker probabilities

Every row compares identical fixtures, labels and orientations at its declared endpoint.
Score cells are **log loss / Brier**; lower is better. Negative ΔLL favors the sports model.
Markets use two-sided no-vig probabilities; benchmark construction differs by dataset.

| Dataset / endpoint | Fixtures | Sports LL / Brier | Market LL / Brier | ΔLL versus market [interval] |
|---|---:|---:|---:|---|
| OddsPortal, first observed | 9,132 | 0.583438 / 0.199523 | 0.586991 / 0.200994 | -0.003554 [-0.008530, +0.001372] |
| OddsPortal, last retained | 9,805 | 0.582592 / 0.199201 | 0.577945 / 0.197117 | +0.004647 [+0.000931, +0.008507] |
| Pinnacle, first observed | 1,040 | 0.558802 / 0.189400 | 0.578937 / 0.197194 | -0.020135 [-0.042401, +0.001336] |
| Pinnacle, last retained | 993 | 0.560559 / 0.190288 | 0.548674 / 0.184349 | +0.011885 [-0.017119, +0.031245] |
| Polish entry LOO | 400 | 0.585025 / 0.202815 | 0.580298 / 0.199347 | +0.004727 [-0.019232, +0.019035] |

Intervals are nominal **95%**, except Pinnacle's **98.75% four-contrast-family** LL intervals;
neither adjusts for all prior adaptive research. Resampling uses 70 / 9 / 6 monthly blocks for
OddsPortal / Pinnacle / Polish comparisons. First observed is not certified market opening;
last retained is not necessarily the final market price. Endpoint cohorts differ, so their
aggregate scores do not identify a causal timing advantage.

### What fixed shrinkage changes

The same **400 Polish-panel fixtures** use entry-time LOO consensus: at least two other books,
each no more than five minutes old. Losses are averaged within fixture/book, then over books and fixtures.

| Forecast | Log loss | Brier |
|---|---:|---:|
| Fixed Hybrid-50 | 0.576162 | 0.198601 |

`p_H = expit(0.5*logit(p_sports) + 0.5*logit(p_market))`—fixed, not fitted here.
Hybrid−consensus ΔLL: **-0.004136 [-0.016001, +0.002928]**.
Its primary intervals versus consensus and C20R include zero: lower point loss is not proven superiority.
This is market-assisted prediction, not evidence that the sports-only model beat the market.

## Betting opportunities—not a live bet ledger

Saved C20R counterfactuals on collected Polish prices; zero tax and unit returns, **no new bankroll simulation**.
The original fixed policy and the exploratory repeated-check study answer different questions:

| View | Suggested observations | Distinct fixtures | Mean expected EV | Realized unit yield [95% CI] |
|---|---:|---:|---:|---|
| First eligible [0,48)h | 82 | 82 | +10.54% | +11.76% [-6.63%, +42.41%] |
| Time-stratified snapshot checks | 451 | 74 | +11.01% | -4.99% [-21.96%, +17.69%] |

The snapshot study samples 2,000 recorded fixture/timestamps in each of
0–2, 2–6, 6–12, 12–24, 24–48 and **48h+**: 12,000 checks across 471 fixtures,
451 suggestions (3.76%). It evaluates offers recorded exactly then,
not a complete carried-forward bookmaker board. Repeated suggestions share outcomes; do not count
them as independent bets or replace them with a better-looking post-hoc one-per-fixture sample.
[Selection, time profiles and uncertainty](docs/PROJECT_REPORT.md#snapshot-opportunities-and-unit-returns).

### CLV versus last recorded pre-start prices

Closing reference = the **latest recorded strictly pre-start state**, with **no age cap** and no hidden
requirement that it be later than entry. Invalid latest states stay missing; no favorable fallback.
Same entry/closing state is allowed. Price CLV does **not** require an executed fill or a known outcome.

| View | Same-book raw mean (covered) | Other-book fair mean [95% CI] (covered) | Fair median |
|---|---:|---|---:|
| First eligible [0,48)h | +8.32% (78/82) | +2.17% [-0.33%, +6.17%] (82/82) | +1.08% |
| Time-stratified snapshot checks | +6.43% (421/451) | -0.30% [-1.46%, +1.08%] (451/451) | -2.12% |

Raw price CLV: `entry_odds / same_book_recorded_close_odds - 1`.
Fair LOO CLV: `entry_odds * closing_other_book_fair_probability - 1`; at least two other books,
equal-weight fair logits. Means are conditional on their displayed coverage; missing prices are not zero.
In the snapshot study, 205/451 fair references contain a contributor at least two hours old;
the oldest-contributor age has median 1.08h and p95 57.08h.
Favorable same-book shortening is not proof of positive fair value or profit. These are recorded closes,
not authenticated final market quotes; fill evidence is a separate limitation.
[Definitions, ages and distributions](docs/PROJECT_REPORT.md#recorded-close-clv).

## Evidence boundaries

- Historical years are development-exposed; prior rosters/release clocks are scenarios, not authenticated issuance.
- Missing model/market coverage stays explicit. Neither repeated quotes nor bootstrap draws create new matches.
- DRO is a probability stress gate, not a calibrated confidence bound. Rating/drift safeguards were not assessed.
- Positive historical yield does not establish reliable displayed EV. Full tax, wallet, Glicko and calibration
  evidence remains in the [detailed report](docs/PROJECT_REPORT.md), separate from these headline comparisons.

## Reproduce and run

With the [bound local artifacts](docs/REPRODUCING.md#required-artifacts-and-evidence-checking) installed,
choose an **unused output directory**:

```bash
.venv/bin/python scripts/reproduce_project_results.py --output-dir /tmp/c20r-readme-evidence-replay
```

Saved-only scoring: no fitting, inference, collection, database or network access. A clean clone does
not contain ignored datasets/model bundles. [Full-precision snapshot](docs/results/project_summary.json)
includes source hashes, periods, counts, weighting and domain-specific bootstrap conventions.

## Documentation

[Detailed evidence](docs/PROJECT_REPORT.md) · [Reproduction](docs/REPRODUCING.md) ·
[Model catalog](docs/MODEL_CATALOG.md) · [Architecture](docs/PROJECT_ARCHITECTURE.md) ·
[Application setup](betting_app/README.md) · [Research index](docs/README.md)

## License and disclaimer

[MIT License](LICENSE). Research and education, not financial advice. Historical results do not guarantee future returns.
LoL-Breaker is not endorsed by Riot Games. League of Legends and Riot Games are Riot Games trademarks.
