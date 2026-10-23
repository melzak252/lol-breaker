# C0-D-20: bookmaker correlation and shrinkage — 2026-10-23

**COMPLETED / HISTORICAL_DEVELOPMENT_DIAGNOSTIC.** The selected fresh20+regional model is strongly correlated with bookmakers but is not identical to them. Fixed logit shrinkage improves historical point scores. A 50/50 blend is the lowest-log-loss tested weight on both examined cohorts, but superiority over both endpoints is not established. No live optimal alpha or betting profitability has been demonstrated.

Run: `production_probability_20261023_02`. Frozen protocol: `data/05_model_input/production_probability/production_probability_20261023_02/protocol.json`. Evidence: `data/08_reporting/production_probability/production_probability_20261023_02/`.

## Scope and model identity

- Only `c0-native-fresh20-2026-10-21-v1-regional-v1`, source model ID `fresh20-regional-full`, enters this analysis. This is the genuine selected fresh20+regional model, not the older positional-20 implementation. Its 20 positions comprise 19 observed positions and one learned token.
- EXP081 and tabular predictions are excluded from the active analysis. Their historical records and the preceding sealed audit are preserved, not deleted or relabeled.
- Reuses the actual read-only production database export from `192.168.1.17` collected by the preceding audit. This is not a new export or a live production-model change. The exported older forecasts are not substitutes for the fresh20 counterfactual replay.
- Source accounting: `data/08_reporting/production_probability/production_probability_20261023_01/fresh20_regional_comparison_v3/accounting/part-000.parquet`. The full matched cohort has **100,949 quotes / 482 fixtures / eight books / six months, May–October 2026**. It is not the older multi-year OddsPortal population.
- Existing source, outcome, duplicate, unknown-live-status, roster and temporal exclusions are retained. No EV, DRO, tax, confidence or staking filter is applied to these probability comparisons.

## Estimands and weights

Both model and market probabilities refer to canonical team A winning the series. Individual bookmaker probabilities use proportional de-vigging. All headline scores are unclipped. Compute log loss and Brier per quote, average within fixture/book, then equally across books within each fixture, then equally across fixtures. A fixture with thousands of snapshots is not thousands of independent matches. Lead-bin profiles repeat this aggregation within the bin; they do not score a mean probability.

The predeclared grid is alpha = 0, 0.1, 0.25, 0.5, 0.75, 0.9, 1, with **alpha the sports-model share**:

`p_blend = sigmoid(alpha * logit(p_model) + (1-alpha) * logit(p_market))`.

Alpha 0 is market-only; alpha 1 is model-only. This is logit-space shrinkage, not an arithmetic probability average or an additional BO projection. No weights were fitted. The retrospective grid minimum is exploratory, not selection-unseen validation.

## Correlation and disagreement

Main correlation uses the same fixture/book/quote weights as the full cohort. Confidence intervals use 5,000 paired monthly-block draws with seed 20261023 and six blocks.

| Measure | Estimate | Nominal monthly 95% interval |
|---|---:|---:|
| Probability Pearson correlation | 0.89408 | [0.85741, 0.93383] |
| Logit Pearson correlation | 0.88936 | [0.86031, 0.92151] |
| Mean absolute probability gap | 8.5574 pp | [7.5845, 9.1654] pp |
| Favorite agreement, both forecasts nonneutral | 87.1772% | [82.4191%, 90.9663%] |

Both forecasts are nonneutral on 98.6714% of the weighted population; favorite agreement is conditional on that denominator. Weighted median absolute gap is 7.1688 pp and the 90th percentile is 17.4709 pp. Gaps of at least 10 pp occupy 32.2364% of the weighted population. These are weighted quote comparisons, not percentages of distinct fixtures.

Descriptive correlations of fixture-average probabilities are Pearson 0.90108 and Spearman 0.89880; these are not substitutes for the primary per-quote weighted estimand. Symmetrically adding complements gives probability correlation 0.89433: ordinary centered correlation is orientation-distribution-sensitive, not exactly invariant to that augmentation. Complement pairs are not additional independent evidence.

The directional derivative of log loss at the market endpoint is **-0.03398896**, monthly interval **[-0.06885059, -0.01906917]**. It is independently finite-difference checked. The historical discrepancy direction contains useful average information in this sample; high correlation does not mean identical forecasts. Neither correlation nor this derivative proves calibration, a causal source of information, or profitable price-selected bets.

## Fixed shrinkage grid

The full diagnostic mixes with the very bookmaker being compared. The stricter cohort instead mixes with a contemporaneous **leave-one-book-out (LOO)** consensus. The cohorts differ; compare weights within each column, not absolute scores across cohorts.

| Alpha: sports share | Full focal LL, 482 fixtures | Full focal Brier | LOO consensus LL, 400 fixtures | LOO consensus Brier |
|---:|---:|---:|---:|---:|
| 0 — market only | 0.571468 | 0.195336 | 0.580298 | 0.199347 |
| 0.10 | 0.568420 | 0.194183 | 0.578357 | 0.198718 |
| 0.25 | 0.565147 | 0.193041 | 0.576507 | **0.198244** |
| 0.50 | **0.563043** | **0.192617** | **0.576162** | 0.198601 |
| 0.75 | 0.564911 | 0.193832 | 0.579070 | 0.200197 |
| 0.90 | 0.567821 | 0.195226 | 0.582291 | 0.201658 |
| 1 — model only | 0.570466 | 0.196393 | 0.585025 | 0.202815 |

All five interior weights improve both full-cohort point metrics against both endpoints. On LOO, 0.1, 0.25 and 0.5 improve both point metrics against both endpoints. LOO log loss favors 0.5 while Brier favors 0.25; there is no metric-independent optimum.

### Leave-one-book-out construction and coverage

For each focal observation, take the latest retained state at or before its clock for every other book on the same canonical fixture. Require at least two distinct other books, each no more than 300 seconds old. An invalid latest state blocks an older valid state; conflicting same-clock states are withheld. Known nonlive observed-price status, resolved sides, valid prices and known prestart/export bounds remain required. The focal book is never a contributor.

The consensus is the sigmoid of the equal-weight mean of the other books' fair logits. This is a conditional observed-price scenario, not authentication of source publication or executable availability.

The frozen protocol's phrase “geometric mean of ... fair log odds” is imprecise. Execution uses an arithmetic mean of log odds, equivalently a geometric mean of fair odds—not a geometric mean of logits. `verification/interpretation_addendum.json` records this wording clarification without modifying the freeze, probabilities or grid.

Of 100,949 focal cases:

- **51,208 matched**, covering **400 fixtures**.
- 23,733 have no fresh eligible other-book state.
- 23,721 have fewer than two fresh eligible other books.
- 2,287 have no prior other-book state.

All cases are accounted for. Endpoints are rescored on exactly the same LOO observations with cohort-specific fixture/book weights. The different model-only LL on LOO reflects changed support and weights, not changed model predictions.

## Uncertainty and robustness

Negative differences favor the alpha-0.5 blend. These are nominal 95% intervals, not corrections for the grid search or the wider adaptive research history.

| Cohort and comparison | Delta LL | Monthly 95% interval | Source-name/year proxy 95% interval |
|---|---:|---:|---:|
| Full: blend minus focal market | -0.008425 | [-0.025176, -0.001324] | [-0.018335, +0.001145] |
| Full: blend minus model | -0.007422 | [-0.014102, +0.010159] | [-0.017832, +0.002296] |
| LOO: blend minus consensus | -0.004136 | [-0.016001, +0.002928] | [-0.012913, +0.005652] |
| LOO: blend minus model | -0.008863 | [-0.016018, +0.003657] | [-0.020414, +0.000741] |

The proxy sensitivity has 67 blocks on the full cohort and 50 on LOO. They are source tournament-name/year labels, not authenticated independent tournament editions. The favorable full-cohort monthly LL result against the focal market does **not** survive this alternative blocking scheme. All four match-weighted monthly Brier intervals cross zero; the LOO proxy Brier interval favors the blend against model-only, but its LL interval crosses zero and equal-proxy Brier does too.

Equal-month LL estimates are -0.012638 against the focal market and -0.000231 against the model on full support; on LOO they are -0.010217 against consensus and -0.000068 against the model. Equal-month intervals favor the market comparisons but cross zero against the model. Equal-proxy LL intervals cross zero for all four comparisons. Full interval objects and lead-bin profiles are saved; no best betting time is selected.

## Verification and evidence

Executed the two archived analysis programs, `correlation/analyze_correlation.py` and `shrinkage/analyze_shrinkage.py`, plus independent calculations:

- `verification/independent_focal.py`: independent full-cohort correlation, weights, seven-alpha scores and paired monthly intervals.
- `verification/verify_shrinkage.py`: independently reproduces both seven-alpha grids and alpha-0.5 endpoint intervals; reconstructs consensus directly from raw decimal prices for **151,295 contributor values** across all **51,208 LOO rows**. Maximum consensus error **2.22e-16**. Checks contributor identities, focal exclusion and 0–300-second ages; verifies latest-state selection on **120 spread focal samples**. This is not independent reconstruction of every absence.
- `verification/verify_proxy_sensitivity.py`: independent full/LOO source-label sensitivity, with explicit canonical-ID string normalization.
- Correlation agreement: probability-correlation error at most 2.22e-16, logit-correlation error at most 1.11e-16 and absolute-gap error at most 2.78e-17. All six correlation-bootstrap metrics report zero undefined draws.
- Source analysis checks pass for endpoint identity, side complement, freshness, no future clocks and focal-book exclusion. Source/accounting/protocol hashes are recorded in the analysis receipts.

Verifier-only failures are retained in the execution record: an initial import-path error, a naive floating-point summation discrepancy of 1.42e-14 repaired by compensated aggregation without relaxing tolerance, and an integer/string canonical-ID join mismatch repaired by explicit normalization. Agent metadata/interpretation corrections do not change source/model probabilities, the frozen alpha grid or inference. No new model fit, neural replay, application change or application regression suite was run for this focused investigation; the preceding audit's 49 tests are not new verification here.

## Decision and remaining prerequisite

**Use 50% sports / 50% market in logit space as a candidate for separately frozen confirmation, not as a validated live setting.** Keep 0.25 visible as the LOO Brier minimum rather than pretending all metrics agree. Do not infer impossibility from uncertain intervals, or superiority from favorable point estimates.

Historical data and the model recipe are development-exposed. Roster availability, historical issuance, production/local model identity and source-binding limitations from the preceding audit remain. This probability analysis establishes neither tax-adjusted EV reliability nor policy ROI. No gate/DRO threshold, stake, deployment, live source, old sealed audit, canonical benchmark manifest or forward date was changed. Actual confirmation needs a frozen future-valid recipe and genuinely available inputs; it cannot be manufactured by selecting the best weight on these outcomes.
