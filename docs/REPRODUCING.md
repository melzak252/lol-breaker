# Reproducing the published project aggregates

## Saved-ledger replay

[`scripts/reproduce_project_results.py`](../scripts/reproduce_project_results.py) scores the three
market datasets, fixed shrinkage, saved snapshot opportunities and latest-recorded closing CLV.
It also retains standalone model, ratings, finance and existing wallet evidence.
**No fitting, model inference, data collection, database connection or network access.**
The checked-in [aggregate](results/project_summary.json) is output, never a fallback input.

From the repository root, with all required local artifacts installed:

```bash
.venv/bin/python scripts/reproduce_project_results.py --output-dir /tmp/c20r-readme-evidence-replay
```

The output directory must be **new**. If it exists, choose an unused numbered suffix; do not delete
or overwrite a completed run. Output is `project_summary.json` and `replay_receipt.json`, with
schemas `project_results_replay_v4` and `project_results_replay_receipt_v4`.
Missing inputs, required hash mismatches, duplicate curated identities and inconsistent admitted
pairs fail before output writing. There is no replacement model, price, smaller convenient cohort
or wider tolerance to force a successful replay.

## Aggregate contract

- `bookmaker_comparisons.oddsportal`: annual model/focal-book first and last quality comparisons.
- `bookmaker_comparisons.pinnacle`: primary C20R/Pinnacle pairs, nominal/adjusted intervals and
  separately scoped Fortuna/Superbet and three-way comparisons.
- `bookmaker_comparisons.polish`: the former top-level `consensus_prediction_quality`, migrated
  without retaining an obsolete alias. Includes C20R / consensus / Hybrid-50 and all seven profiles.
- `production_opportunities`: fixed saved 12,000-check replay, original first-eligible [0,48) policy,
  zero-tax unit returns and the current latest-only closing convention.
- `model_scores`, `glicko2_comparison`, `financial_scenarios`, `later_price_proxies` and
  `production_betting`: retained numerical evidence. Legacy price sensitivities keep their original
  strictly-later/freshness rules; they are not silently reinterpreted as current latest-only CLV.
- `reproduction`: actual consumed input/source hashes, versions and `bootstrap_by_domain`.

The schema change does not rewrite original study outputs, model/config/manifest pins or decisions.
Closing-price metrics are measurable without fills; `authenticated_final_market_quote=false` and
execution metadata are separate statements, not a declaration that price CLV cannot be calculated.

## Environment prerequisite

The existing scientific environment was exercised, not a fresh dependency installation:

| Component | Exercised version |
|---|---|
| numpy | 2.4.4 |
| pandas | 3.0.3 |
| pyarrow | 24.0.0 |
| python | 3.12.3 |
| scikit-learn | 1.8.0 |
| scipy | 1.17.1 |

Python 3.11+ is required by project metadata. Direct numerical dependencies are NumPy, pandas,
PyArrow, SciPy and scikit-learn. Current tracked engine/EV/guards/bootstrap code is also imported.
No CUDA, neural inference or environment change is needed for this saved-evidence command.
The historical PyTorch version in a fitting receipt is not a fresh-install compatibility proof.

## Required artifacts and evidence checking

Paths below are repository-relative. Datasets, saved predictions and receipts are generally
**local and ignored**, not GitHub downloads. A clean clone is insufficient; install matching bound
inputs from an authorized source. A DVC pointer or recorded remote does not establish availability
of these later artifacts. All remaining tracked numerical-source paths are inventoried in the
aggregate's `reproduction.input_paths`.

### Standalone sports inputs

| Key | Repository-relative required path |
|---|---|
| `model_predictions` | `data/07_model_output/c0_model_improvement/c0_model_improvement_20261026_01/predictions.parquet` |
| `model_protocol` | `data/05_model_input/c0_model_improvement/c0_model_improvement_20261026_01/protocol.json` |
| `model_binding` | `data/05_model_input/c0_model_improvement/c0_model_improvement_20261026_01/execution_binding.json` |
| `model_verification` | `data/08_reporting/c0_model_improvement/c0_model_improvement_20261026_01/sealed_verification.json` |
| `serving_config` | `conf/base/c0_serving.json` |

Run identity, protocol-to-execution binding, successful model verification/predictions SHA-256 and
local serving identity must agree. There are 3,103 assessment rows, 3,083 supported regional rows
and 20 explicit absences. This command does not reconstruct features or infer from saved weights.

### Annual OddsPortal inputs

| Key | Repository-relative required path |
|---|---|
| `odds_protocol` | `data/05_model_input/fresh20_oddsportal/fresh20_oddsportal_20261111_02/protocol.json` |
| `odds_evaluation` | `data/08_reporting/fresh20_oddsportal/fresh20_oddsportal_20261111_02/evaluation/oddsportal_annual_endpoint_ledger.parquet` |
| `odds_binding` | `data/08_reporting/fresh20_oddsportal/fresh20_oddsportal_20261111_02/evaluation/execution_binding.json` |
| `odds_quality_reference` | `data/08_reporting/fresh20_oddsportal/fresh20_oddsportal_20261111_02/evaluation/probability_quality.parquet` |
| `odds_evaluation_receipt` | `data/08_reporting/fresh20_oddsportal/fresh20_oddsportal_20261111_02/evaluation/evaluation_receipt.json` |
| `events` | `data/04_feature/long_term_market_edge/long_term_market_edge_20261022_01/oddsportal_identity_complete_v3/events.parquet` |

The original protocol and source-event hashes are checked against the saved execution binding.
Full inventory retains source years 2021–2026: 13,473 source events, including those without quotes.
There are 13,338 quoted events / 125,322 endpoint quote rows; 135 source events have no retained
endpoint quotes. The quote ledger and quality reference are hashed in the current replay, **not
independently sealed output files**. Their clock/source authenticity is not certified here.

Quality follows the original `_focal_quality` contract, implemented in tracked reporting code:
filter timed-price/forecast/market eligibility; sort clock/book/source event; take the earliest
eligible focal quote per canonical fixture and endpoint; then score known outcomes. Do not average
all bookmaker losses or select the highest EV book. Compare both retained endpoints independently.
Pooled scores and LL intervals reproduce the two saved 2021–2026 quality rows within 1e−12.

Year-eligible annual fresh20+regional forecasts are consumed, not a 2026 model substituted for all
years. Finance still reuses the existing selector/settlement definitions and all four tax scenarios,
including the fixed-original-ticket audit. Existing strictly-later price pairs preserve the actual
selected book/side/entry and their old 300-second two-side validity condition.

### Pinnacle comparison inputs

| Key | Repository-relative required path |
|---|---|
| `pinnacle_ledger` | `data/07_model_output/pinnacle_model_comparison/pinnacle_model_comparison_20261112_043418/comparison_ledger.parquet` |
| `pinnacle_scores_reference` | `data/08_reporting/pinnacle_model_comparison/pinnacle_model_comparison_20261112_043418/scores.csv` |
| `pinnacle_summary_reference` | `data/08_reporting/pinnacle_model_comparison/pinnacle_model_comparison_20261112_043418/summary.json` |
| `pinnacle_verification` | `data/08_reporting/pinnacle_model_comparison/pinnacle_model_comparison_20261112_043418/verification.json` |

Verification must be PASS for the expected run. The ledger and score CSV SHA-256 values must match
its recorded `sha256` mapping. All 4,364 source endpoint rows / 2,182 fixtures remain accounted for;
only flag-admitted, known, prestart pairs are scored. Missing mappings are normal coverage absences,
not invalid admitted pairs or invented probabilities. Current summary-reference bytes are hashed.

Both endpoints use canonical-oriented probabilities and labels. Primary C20R/Pinnacle cohorts are
1,040 / 993 fixtures; Polish/Pinnacle are 123 / 112; three-way intersections are 100 / 89.
All 14 saved score rows plus primary paired LL/Brier/family intervals are checked at absolute
1e−12. No reprojection, new clipping, inferred labels or model fitting is performed.

Bootstrap months follow **Pinnacle scheduled UTC starts**, not canonical match dates or conservative
earliest starts. The latter still govern quote admission. Two fixtures span month boundaries between
those clocks; enforcing equality to canonical-date months was an incorrect new-verifier assumption,
not a reason to alter the saved cohort or bootstrap grouping.

### Polish entry-consensus inputs

| Key | Repository-relative required path |
|---|---|
| `loo_panel` | `data/08_reporting/production_probability/production_probability_20261023_02/shrinkage/loo_matched_panel.parquet` |
| `loo_case_accounting` | `data/08_reporting/production_probability/production_probability_20261023_02/shrinkage/loo_case_accounting.parquet` |
| `loo_scores_reference` | `data/08_reporting/production_probability/production_probability_20261023_02/shrinkage/loo_consensus_scores.csv` |
| `loo_contrasts_reference` | `data/08_reporting/production_probability/production_probability_20261023_02/shrinkage/loo_consensus_paired_contrasts.csv` |
| `loo_verification` | `data/08_reporting/production_probability/production_probability_20261023_02/verification/sealed_verification.json` |

The seal must identify C20R and `COMPLETED_HISTORICAL_DEVELOPMENT_DIAGNOSTIC`; all four consumed
data/reference hashes must match its `files[path].sha256` map. Panel hash:
`6cfb8c7069494ee091628cae4f26f9bf334213b68e28a6ce8f6d770f7a8e8486`.
Accounting hash: `78ca41a580ab56bc3c2f3655f5158cc169434bd6b4b8e4a5d29d742f49f0e5af`.
Historical source/doc hashes are not repinned to match today's code.

JSON contributor lists are decoded and checked for lengths, focal exclusion, two-book minimum,
nonfuture clocks, <=300-second entry freshness, weights and mean-logit agreement. Matched accounting
joins exactly to the 51,208-record panel; 49,741 exclusions remain in the full 100,949-case accounting.
Losses average within fixture/book, then over books and fixtures. Exactly C20R, consensus and fixed
Hybrid-50 are scored; no alpha selection. All seven saved lead profiles remain available.

### Production opportunities and closing states

| Key | Repository-relative required path |
|---|---|
| `production_events` | `data/05_model_input/c0_gate_optimizer/c0_gate_optimizer_20261023_01/events.parquet` |
| `production_configurations` | `data/05_model_input/c0_gate_optimizer/c0_gate_optimizer_20261023_01/configurations.json` |
| `production_closing_states` | `data/08_reporting/production_zero_tax_clv/production_zero_tax_clv_20261023_01/clv/closing_states.parquet` |
| `sampled_checks` | `data/08_reporting/audit/model_betting_reconciliation_20261113_01/sampled_checks.parquet` |
| `sampled_suggestions_reference` | `data/08_reporting/audit/model_betting_reconciliation_20261113_01/checked_suggestions.parquet` |
| `sampling_reference` | `data/08_reporting/audit/model_betting_reconciliation_20261113_01/results.json` |

Frozen event/configuration bindings and the reconciliation's raw closing-state hash are checked. The
saved 12,000 sampled checks and 451 reference selections are hashed by this replay but were **not
independently sealed at sampling time**. The recorded seed describes the original sample; replay
consumes that exact check ledger instead of drawing a new, potentially favorable sample.

The replay checks unique fixture/timestamps, 2,000 per lead band, source clocks and fixed logit
probabilities. Unchanged zero-tax gates must reproduce the exact selected event IDs and source
odds/p/q/EV/outcomes within 1e−12. No future closing coverage determines entry selection. Observations
occur exactly at stored offer clocks, not a reconstructed full fresh board. Missing rating/drift
inputs remain NOT_ASSESSED, not successful complete-system safeguards.

Closing state = one latest recorded strictly prestart state per fixture/book, no maximum age and
no after-entry condition. Invalid latest states stay absent; no older fallback or favorable
choice. Same entry/closing state is allowed. Recompute own-book raw CLV and peer fair CLV from raw
two-sided prices. Peers exclude the focal book, need two eligible other books and use mean fair
logits. Outcomes/fills/stakes are not prerequisites for price CLV; zero-tax unit yield uses its own
known-label coverage. LOO age statistics use the oldest contributor per observation.

### Retained wallet and reference inputs

| Key | Repository-relative required path |
|---|---|
| `production_protocol` | `data/05_model_input/c0_gate_optimizer/c0_gate_optimizer_20261023_01/protocol.json` |
| `production_fixtures` | `data/05_model_input/c0_gate_optimizer/c0_gate_optimizer_20261023_01/fixtures.parquet` |
| `production_binding` | `data/05_model_input/c0_gate_optimizer/c0_gate_optimizer_20261023_01/binding.json` |
| `production_verification` | `data/08_reporting/c0_gate_optimizer/c0_gate_optimizer_20261023_01/final_verification.json` |
| `production_wallet_summary` | `data/08_reporting/c0_gate_optimizer/c0_gate_optimizer_20261023_01/sweep/summary.parquet` |
| `production_selected` | `data/08_reporting/c0_gate_optimizer/c0_gate_optimizer_20261023_01/sweep/selected.parquet` |
| `production_funded` | `data/08_reporting/c0_gate_optimizer/c0_gate_optimizer_20261023_01/sweep/funded_ledger.parquet` |
| `production_entry_clv` | `data/08_reporting/production_zero_tax_clv/production_zero_tax_clv_20261023_01/clv/entry_clv_panel.parquet` |
| `production_clv_verification` | `data/08_reporting/production_zero_tax_clv/production_zero_tax_clv_20261023_01/verification/sealed_verification.json` |

Twelve original fixed cases are read, not a new optimizer sweep. Selected/funded rows join by case
and opportunity with source/side/clocks/odds checked; saved stakes/Decimal payouts are verified.
The current wallet simulator is not called. Legacy 300-second, 3,600-second and uncapped reference
sensitivities still require a later update; they remain distinct from the current latest-only CLV.

### Corrected ratings-only inputs

| Key | Repository-relative required path |
|---|---|
| `glicko_predictions` | `data/artifacts/corrected-ratings-logloss-20260908-full/holdout_predictions.csv` |
| `glicko_summary` | `data/artifacts/corrected-ratings-logloss-20260908-full/summary.json` |
| `glicko_verification` | `data/artifacts/corrected-ratings-logloss-20260908-full/verification.json` |

The corrected receipt hashes match predictions and summary. All 10,578 unique series must contain
both raw and both preceding-year calibrated saved probabilities, binary outcomes, valid BO1/3/5,
dates and distinct teams. They are already BO-projected: no second projection or calibration fit.
This separate control study is not a same-cohort C20R ranking. Twelve N/LL/Brier references reproduce.

## Bootstrap conventions

All domains use 5,000 draws. Reported intervals are nominal historical 95% unless noted; Pinnacle's
primary LL comparison additionally retains its 98.75% four-contrast-family interval. Neither is a
correction for all adaptive development or prospective confirmation.

| Domain | Month blocks | Seed | Primary estimand |
|---|---:|---:|---|
| `consensus` | 6 | 20261023 | equal fixture after mean quote loss within book and equal books within fixture |
| `finance` | 70 | 20261111 | equal-unit-stake ratio of totals; equal-active-month sensitivity |
| `oddsportal_prediction` | 70 | 20261111 | equal fixture; earliest eligible focal quote within each retained endpoint |
| `pinnacle_prediction` | 9 | 20261112 | paired equal fixture; equal-month sensitivity |
| `production_betting` | 6 | 20261023 | saved-stake ROI ratio; equal-ticket closing proxies, stake-weighted sensitivity |
| `production_opportunities` | 6 | 20261113 | selected-observation mean; fixed one-per-fixture policy separate |
| `ratings` | 30 | 20261111 | equal series; equal-month sensitivity |

OddsPortal/finance retain all 70 preselection months, including zero-action months. Production
retains all six fixture-start months. Pinnacle resamples the nine scheduled-start month blocks.
Glicko uses 30 match-date months. Repeated observations from a fixture stay in the same month;
bootstrap draws do not create independent sports outcomes. Ratio intervals retain undefined draws
and withhold a CI if any denominator is zero. Active-month-only sensitivities and equal-month
estimands are separately labeled. Quantiles of observations are not confidence intervals.

## Verification performed

With deliberately unreachable database URLs:

```bash
TEST_DATABASE_URL=postgresql://task:task@127.0.0.1:9/task_test \
DATABASE_URL=postgresql://task:task@127.0.0.1:9/task_test \
.venv/bin/python -m pytest -q \
  betting_app/tests/test_reproduce_project_results.py \
  betting_app/tests/test_production_saved_evidence.py \
  betting_app/tests/test_saved_market_comparisons.py \
  betting_app/tests/test_latest_recorded_closes.py \
  betting_app/tests/test_long_term_market_evaluation.py \
  betting_app/tests/test_production_market_benchmark.py
```

**115 passed**, one existing Starlette/httpx deprecation warning. No database/application/collector
was started. The actual v4 CLI completed at `/tmp/c20r-readme-evidence-replay-3`; the checked-in
summary is byte-identical, SHA-256 `cdd7d8f420d04f07445dcf5d11f31841628924271083a5bc47ae4fc9407090c3`.
All 3,188 retained numeric fields remain unchanged. Market reference checks cover 28 annual,
90 Pinnacle and 528 Polish-panel values; maximum difference 1.11e−16. Original failed new-verifier
month validation is retained in session history; the correction changed no source or tolerance.

SHA-256 proves byte identity, not original source authenticity, neural internals, publication,
actual starts or real forecast issuance. This is a saved-evidence replay, not a raw-to-neural clean
clone reproduction or live trading certification. Do not execute archived analysis writers into
occupied study namespaces. See the [project report](PROJECT_REPORT.md) for full scope and results.
