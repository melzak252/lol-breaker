# Reproducing the published project aggregates

## Saved-ledger replay

[`scripts/reproduce_project_results.py`](../scripts/reproduce_project_results.py) recomputes five
evidence domains: standalone C20R scores, hierarchical other-book consensus scores,
matched player/organization Glicko-2 controls, multi-year ticket/later-price diagnostics,
and saved production interval wallets with conditional closing-price proxies.
It reuses tracked scoring, selectors, EV, guards, saved-wallet and bootstrap functions.
**No training, inference, data acquisition, database connection or network access.**
The checked-in [summary snapshot](results/project_summary.json) is output, never a fallback input.

From the repository root, with the bound artifacts installed:

```bash
.venv/bin/python scripts/reproduce_project_results.py --output-dir /tmp/c20r-trader-readme-replay
```

The directory must be **new**. If occupied, choose an unused numbered suffix; never overwrite or
delete an existing run. Output is aggregate-only `project_summary.json` and `replay_receipt.json`,
using schemas `project_results_replay_v3` and `project_results_replay_receipt_v3`.
All required identities, receipts and numerical prerequisites are checked before output writing.
Missing inputs, required hash mismatches, duplicated curated identities and inconsistent labels
fail explicitly; no replacement probabilities, prices or convenient smaller cohorts are invented.

## Environment prerequisite

The existing repository scientific environment was exercised, not a fresh dependency install:

| Component | Replay version |
|---|---|
| numpy | 2.4.4 |
| pandas | 3.0.3 |
| pyarrow | 24.0.0 |
| python | 3.12.3 |
| scikit-learn | 1.8.0 |
| scipy | 1.17.1 |

Project metadata requires Python 3.11+. Direct numerical dependencies are NumPy, pandas, PyArrow,
SciPy and scikit-learn. The imported source closure includes repository engine, guard, EV,
probability-metric, paired-cluster and ratio-bootstrap code. The replay requires no CUDA or
model inference. Check imports in the existing environment before running:

```bash
.venv/bin/python -c 'import numpy, pandas, pyarrow, scipy, sklearn; from src.analysis.long_term_market_evaluation import select_fixed_policy, summarize_unit_returns; from src.analysis.production_market_benchmark import paired_cluster_interval; print("saved-ledger imports available")'
```

A fresh install was not tested. The development extra and `uv.lock` are not asserted to supply a
complete application/neural environment. The historical evaluation binding's PyTorch 2.14.0+cpu
is not proof of the original checkpoint's training environment; `requirements-betting.txt`
separately pins PyTorch 2.5.1+cu121. No dependency or environment change is required by this task.

## Required artifacts and evidence checking

Paths below are relative to repository root. Saved data/model/evidence artifacts are **local and
generally ignored**, not GitHub download links. A clean clone alone cannot run the replay.
Install matching bound inputs from an authorized source; the recorded DVC remote does not prove
availability of these later runs. There is no fallback to the committed aggregates.
The public serving config and current numerical source modules are in the checkout.

### Standalone sports inputs

| Key | Required path |
|---|---|
| `model_predictions` | `data/07_model_output/c0_model_improvement/c0_model_improvement_20261026_01/predictions.parquet` |
| `model_protocol` | `data/05_model_input/c0_model_improvement/c0_model_improvement_20261026_01/protocol.json` |
| `model_binding` | `data/05_model_input/c0_model_improvement/c0_model_improvement_20261026_01/execution_binding.json` |
| `model_verification` | `data/08_reporting/c0_model_improvement/c0_model_improvement_20261026_01/sealed_verification.json` |
| `serving_config` | `conf/base/c0_serving.json` |

The replay preserves the original run ID, protocol-to-execution binding, successful model
verification/predictions hash and local serving identity. It scores all 3,103 canonical rows and
retains 3,083 supported / 20 absent regional predictions without imputation.

### Financial and endpoint inputs

| Key | Required path |
|---|---|
| `odds_protocol` | `data/05_model_input/fresh20_oddsportal/fresh20_oddsportal_20261111_02/protocol.json` |
| `odds_evaluation` | `data/08_reporting/fresh20_oddsportal/fresh20_oddsportal_20261111_02/evaluation/oddsportal_annual_endpoint_ledger.parquet` |
| `odds_binding` | `data/08_reporting/fresh20_oddsportal/fresh20_oddsportal_20261111_02/evaluation/execution_binding.json` |
| `events` | `data/04_feature/long_term_market_edge/long_term_market_edge_20261022_01/oddsportal_identity_complete_v3/events.parquet` |

Original OddsPortal protocol and event hashes are validated against the saved financial binding.
The endpoint ledger is hashed into the current output inventory, **not independently sealed or
authenticated by this replay**. Selection reuses `select_first_policy` / `select_fixed_policy`
and settlement reuses `summarize_unit_returns`, unchanged. Tax is a fraction.
All four scenarios remain in JSON, including the fixed-original-ticket 12% tax audit.

Each requalified policy's actual source/event/canonical fixture/book/side/entry is paired to
`LAST_CHANGE_OR_LAST_PRESTART` in this endpoint ledger. Both side prices must be valid and fresh
within 300 seconds of the paired clock; naive clocks and ambiguous duplicates reject.
The historical zero-tax `first_observed_quote_proxies.parquet` is **not consumed or modified**;
it cannot supply the independently requalified 12% portfolio's pairs.

### Production wallet and closing-reference inputs

| Key | Required path |
|---|---|
| `production_protocol` | `data/05_model_input/c0_gate_optimizer/c0_gate_optimizer_20261023_01/protocol.json` |
| `production_configurations` | `data/05_model_input/c0_gate_optimizer/c0_gate_optimizer_20261023_01/configurations.json` |
| `production_fixtures` | `data/05_model_input/c0_gate_optimizer/c0_gate_optimizer_20261023_01/fixtures.parquet` |
| `production_binding` | `data/05_model_input/c0_gate_optimizer/c0_gate_optimizer_20261023_01/binding.json` |
| `production_verification` | `data/08_reporting/c0_gate_optimizer/c0_gate_optimizer_20261023_01/final_verification.json` |
| `production_wallet_summary` | `data/08_reporting/c0_gate_optimizer/c0_gate_optimizer_20261023_01/sweep/summary.parquet` |
| `production_selected` | `data/08_reporting/c0_gate_optimizer/c0_gate_optimizer_20261023_01/sweep/selected.parquet` |
| `production_funded` | `data/08_reporting/c0_gate_optimizer/c0_gate_optimizer_20261023_01/sweep/funded_ledger.parquet` |
| `production_entry_clv` | `data/08_reporting/production_zero_tax_clv/production_zero_tax_clv_20261023_01/clv/entry_clv_panel.parquet` |
| `production_clv_verification` | `data/08_reporting/production_zero_tax_clv/production_zero_tax_clv_20261023_01/verification/sealed_verification.json` |

The wallet seal must be `COMPLETED`; its input-binding SHA-256 and the binding's protocol,
configuration and fixture hashes are checked, along with the three saved wallet data hashes.
The closing seal must be `COMPLETED_HISTORICAL_CONDITIONAL_DIAGNOSTIC`; its entry-panel hash
must match. Original study source/doc hashes are not rewritten to match the new replay.
Current numerical closure additionally includes `src/analysis/production_saved_evidence.py`.

Read exactly `tax{0,0.12}__logit__baseline__{h0_2,h2_6,h6_12,h12_24,h24_48,primary48}`:
twelve fixed existing cases, not a new optimizer sweep. Selected/funded tickets join one-to-one
by case and opportunity, with fixture, side, book, clocks, odds and outcome checked.
Saved stakes and Decimal payouts are verified; the current wallet engine is not called to
replace the archived per-ticket free-cash cap with a different aggregate exposure rule.

Each selected-side entry is paired independently to 300-second, 3,600-second and uncapped-age
recorded references. JSON contributor books, IDs, clocks and ages are decoded and checked;
the focal book is excluded and at least one other contributor must be later than entry.
Reference ages run backward from conservative recorded start, not forward from entry.
Same-book movement, gross LOO and tax-adjusted net LOO proxies remain distinct.
Missing entries/references retain reasons; no reference is manufactured or counted as zero.
Saved fair q is reused from the sealed panel, not independently reconstructed from all raw
closing states. This establishes neither authentic closes nor fills.

### Sealed consensus inputs

| Key | Required path |
|---|---|
| `loo_panel` | `data/08_reporting/production_probability/production_probability_20261023_02/shrinkage/loo_matched_panel.parquet` |
| `loo_case_accounting` | `data/08_reporting/production_probability/production_probability_20261023_02/shrinkage/loo_case_accounting.parquet` |
| `loo_scores_reference` | `data/08_reporting/production_probability/production_probability_20261023_02/shrinkage/loo_consensus_scores.csv` |
| `loo_contrasts_reference` | `data/08_reporting/production_probability/production_probability_20261023_02/shrinkage/loo_consensus_paired_contrasts.csv` |
| `loo_verification` | `data/08_reporting/production_probability/production_probability_20261023_02/verification/sealed_verification.json` |

The seal must identify `c0-native-fresh20-2026-10-21-v1-regional-v1` and status
`COMPLETED_HISTORICAL_DEVELOPMENT_DIAGNOSTIC`. Its
`files[repository_relative_path].sha256` map must match all four consumed data/reference files.
Historical code/doc hashes in that receipt are **not repinned**: current consumed numerical
sources are hashed separately.

Panel SHA-256: `6cfb8c7069494ee091628cae4f26f9bf334213b68e28a6ce8f6d770f7a8e8486`.
Accounting SHA-256: `78ca41a580ab56bc3c2f3655f5158cc169434bd6b4b8e4a5d29d742f49f0e5af`.
Contributor proof fields are JSON strings, decoded and checked for lengths, focal exclusion,
weights, clocks, freshness and mean-logit consistency. Matched accounting identities must join
one-to-one to the panel; excluded reasons and the full 100,949-case population remain explicit.

Exactly C20R / consensus / fixed Hybrid-50 are scored. Quote losses are averaged within fixture/book,
then equally over books and fixtures; probabilities are never averaged before scoring.
All seven saved lead bins and their paired LL/Brier contrasts are checked against the two CSVs
at absolute tolerance **1e-12**, with no new clipping or tolerance widening.

### Corrected ratings-only inputs

| Key | Required path |
|---|---|
| `glicko_predictions` | `data/artifacts/corrected-ratings-logloss-20260908-full/holdout_predictions.csv` |
| `glicko_summary` | `data/artifacts/corrected-ratings-logloss-20260908-full/summary.json` |
| `glicko_verification` | `data/artifacts/corrected-ratings-logloss-20260908-full/verification.json` |

The receipt's `sha256` map must match predictions and summary. Predictions SHA-256:
`ae18e8ff8b4ff30cbf8f5c87af56a027ec4159249372b365677977a434be702f`.
The summary version is `corrected-header-ratings-only-daybatch-20260908`.
All 10,578 unique rows must contain `player_gl_raw`, `team_gl_raw`,
`player_gl_calibrated` and `team_gl_calibrated`, binary `y_true`, BO1/3/5, date and distinct team IDs.
The saved probabilities already include the independent-map BO projection and prior-year
positive slope calibration; this replay neither projects nor fits again. N/LL/Brier are checked
against the bound summary at absolute tolerance 1e-12.

Do **not** execute the archived `analyze_shrinkage.py`, `ratings_logloss.py` or `run_sensitivity.py`
to produce this report: those are historical writers/fitters with occupied output namespaces.
The tracked aggregate entry point implements the necessary saved-only aggregation/pairing.

## Bootstrap conventions and cohorts

All intervals are nominal historical 95%, with 5,000 resamples;
none is a correction for the project's complete adaptive research history.

| Domain | UTC month universe | Seed | Primary estimand |
|---|---|---:|---|
| Finance and later prices | 70 preselection months, January 2021–October 2026 | 20261111 | Equal-unit-stake ratio of sums; zero-action months retained |
| Consensus | 6 fixture scheduled-start months, May–October 2026 | 20261023 | Equal fixture after hierarchical quote/book loss aggregation |
| Ratings | 30 match-date months, January 2024–June 2026 | 20261111 | Equal series on the identical four-column ledger |
| Production wallets and closing proxies | All 6 fixture-start months, May–October 2026 | 20261023 | Saved-stake ROI; equal-ticket covered price metrics and stake-weighted sensitivity |

Standalone sports scores cover 2026-01-08–2026-09-08; market scores use
51,208 quote cases / 400 fixtures; ratings cover 2024-01-01–2026-06-14.
Financial mapped targets span 2021-01-09–2026-09-07, not the entire resampling calendar.
The production export's fixture starts span 2026-05-29–2026-10-22, not later captures.

Consensus and ratings intervals return both fixture/series-weighted and equal-month differences.
Multi-year finance JSON retains active-block counts, equal-active-month estimates, empty/valid replicates,
statuses, known-label and near-band EV−return intervals. Empty selections have null conditional
metrics; unresolved outcomes remain selected and leave whole-portfolio return null.
Later-price statistics separately expose all-price-pair and matched-known scopes and
active-month-only bootstrap sensitivities. Genuine closing CLV stays null with status
`not_available_no_certified_close_or_fill`.

The summary and receipt use `bootstrap_by_domain`; there is no global one-seed convention.
The v3 snapshot adds `production_betting` without replacing the separate multi-year
`financial_scenarios` and `later_price_proxies`.
Production resampling pads all six preselection start months with zero-action totals and uses
NumPy's default integer draw dtype. The archived optimizer-family `int16` draw matrix is a
different random sequence and is not substituted. Any zero-denominator draw withholds its CI;
later-price coverage never stands in for whole-portfolio CLV.

## Verification performed

With deliberately unreachable database URLs, the pure arithmetic suites were run from the root:

```bash
TEST_DATABASE_URL=postgresql://task:task@127.0.0.1:9/task_test \
DATABASE_URL=postgresql://task:task@127.0.0.1:9/task_test \
.venv/bin/python -m pytest -q \
  betting_app/tests/test_reproduce_project_results.py \
  betting_app/tests/test_production_saved_evidence.py
```

**66 passed**, one existing Starlette/httpx deprecation warning. No PostgreSQL, application,
collector or scheduler was started. The complete v3 CLI succeeded; exclusive output:
`/tmp/c20r-production-readme-replay`. The checked-in snapshot is byte-identical to that summary
(SHA-256 `5d455a9530fdacabe87ce9d5f4cf7d67ddc9e1e522d2abfc6b548b930c42fe49`).
The five previous numeric evidence domains are unchanged, including all 528 consensus and
12 ratings reference comparisons.

An independently computed saved-data oracle for all twelve portfolios / 330 repeated ticket
rows agrees on wallet counts/money, closing-price coverage/formulas and monthly intervals:
2,603 comparisons, 1,935 numeric, maximum absolute error 1.14e-13.
The local comparison receipt is `/tmp/c20r-production-readme-replay/oracle_comparison.json`.
It does not independently reconstruct every contributor's mean-logit q or neural inference.
Pre-publication join/Decimal/empty-reference schema failures were corrected without changing
source inputs, model, policy or numerical tolerances; the original failed oracle receipt is retained.

SHA-256 establishes byte identity, not source authenticity or correctness. The snapshot inventories
current consumed artifacts and the listed numerical source closure, including
`src/analysis/production_market_benchmark.py`; it does not certify every transitive dependency,
original neural weights, raw sports inputs or historical issuance. Original sealed study
namespaces, hashes, model weights, serving and financial gates are not rewritten.

## Separate raw-to-forecast work

This command starts from saved predictions, case accounting, references and endpoint ledgers.
Raw-data acquisition, feature construction, original fitting and forecast generation require
additional original sources and checkpoint bundles. Neither a clean-clone neural reproduction,
public access to local data, authenticated announcements, genuine closing CLV nor live trading
performance is claimed. See the [project report](PROJECT_REPORT.md) for evidence boundaries.
