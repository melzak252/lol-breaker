# Reproducing the published project aggregates

## Clean-clone synthetic scenario

The existing evaluator also has a deterministic, artifact-free scenario path:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[demo]'
python scripts/run_unified_betting_evaluation.py --demo
```

The `demo` extra installs CPU-only NumPy, pandas, and scikit-learn constraints; it does not install PyTorch/CUDA or PyArrow. The runner passes inline synthetic quotes through the existing `UnifiedBettingEngine` qualification gates and `simulate_operational_portfolio`, then prints abstentions, tax configuration, overlapping capital reservations, and the ledger/capital trace. The exact `SYNTHETIC_SCENARIO_NOT_MODEL_EVIDENCE` status is deliberate: this is mechanics only, not a fitted model, sports result, or performance estimate. It reads no research data and contacts neither a database nor the network.

## Saved-ledger replay

[`scripts/reproduce_project_results.py`](../scripts/reproduce_project_results.py) scores the three market datasets, fixed shrinkage, saved snapshot opportunities, and latest-recorded closing CLV, while retaining standalone model, Glicko-2, finance, and wallet evidence. **No fitting, neural inference, scraping, database, or network access occurs.** [project_summary.json](results/project_summary.json) is the checked-in output, never a fallback input.

```bash
.venv/bin/python scripts/reproduce_project_results.py --output-dir /tmp/c20r-readme-evidence-replay
```

- `--output-dir` must be **new** (never overwrite an existing run). Outputs: `project_summary.json` (`project_results_replay_v4`) and `replay_receipt.json` (`project_results_replay_receipt_v4`).
- Missing inputs, hash mismatches, duplicate IDs, or inconsistent admitted pairs fail before output writing.

## DVC-versioned replay

The current saved-ledger replay has a dedicated DVC target. Run only this stage; plain `dvc repro` also traverses the historical simulation pipeline:

```bash
python -m pip install -e '.[replay]'
DVC_NO_ANALYTICS=1 DVC_CHECK_UPDATE=false .venv/bin/dvc pull --with-deps project_results_replay_v4
DVC_NO_ANALYTICS=1 DVC_CHECK_UPDATE=false .venv/bin/dvc repro --force project_results_replay_v4
```

It writes to `data/08_reporting/project_results_replay_v4_dvc_20261116`, an exclusive DVC-owned output directory. Standard DVC stage reruns remove only that declared output before invoking the existing runner, which still refuses any occupied `--output-dir`.

The 37 input `.dvc` pointers, stage, lock, and code must be published together. Drive access additionally requires Google authentication and permission to the configured folder; pointer presence does not establish remote object availability. Never commit authentication files or copy another machine's DVC cache as proof of online restoration.

This stage covers saved-ledger aggregate replay only; it does not version the complete C20R inference/training closure or the prop-v2/draft model artifacts.

### DVC verification performed

An isolated workspace restored all 37 inputs (56,803,982 bytes) from local DVC cache, then ran the stage and a forced DVC-owned rerun; both summaries matched `docs/results/project_summary.json` byte-for-byte (SHA-256 `cdd7d8f420d04f07445dcf5d11f31841628924271083a5bc47ae4fc9407090c3`). This proves only the local cache/replay path—not remote object coverage, GitHub-clone restoration, or model-weight/inference availability. The earlier 115-test suite was not rerun.

## Aggregate contract & scope


- `bookmaker_comparisons.oddsportal`: annual model vs focal book (first/last endpoints).
- `bookmaker_comparisons.pinnacle`: primary C20R/Pinnacle pairs, nominal/adjusted CIs, plus Fortuna/Superbet and 3-way overlaps.
- `bookmaker_comparisons.polish`: C20R, LOO consensus, and Hybrid-50 scores across all 7 lead-time bands.
- `production_opportunities`: 12,000-check stratified replay, fixed `[0,48)h` policy, zero-tax unit yields, and latest-only closing CLV.
- `model_scores`, `glicko2_comparison`, `financial_scenarios`, `later_price_proxies`, `production_betting`: retained saved evidence (legacy price sensitivities keep their original strictly-later/age rules).
- `reproduction`: consumed input/source hashes, runtime versions, and `bootstrap_by_domain`.
- **Scope:** Byte-identity verification proves saved-artifact reproducibility, not neural internals, source-clock authenticity, live issuance, or trading profitability. Do not execute archived writers into occupied study directories.

## Environment prerequisite

Install `.[replay]` for the recorded numerical stack: `numpy==2.4.4`, `pandas==3.0.3`, `pyarrow==24.0.0`, `scipy==1.17.1`, and `scikit-learn==1.8.0`, exercised on Python 3.12.3. It also pins `dvc[gdrive]==3.67.1` and `pyOpenSSL==24.2.1`. An unconstrained Drive installation resolved pyOpenSSL 22.0.0 with cryptography 50.0.2 and failed at import (`GEN_EMAIL`); the replay extra resolves a compatible pair. The separate `.[demo]` extra does not include replay dependencies or research data.

## Required artifacts and evidence checking

The 37 consumed `data/` files remain Git-ignored payloads with individual DVC pointers. The replay receipt records SHA-256 values for the 52 paths in `PATHS`. Use the named-stage pull above, not an unrestricted pull of historical datasets. A complete online reproduction requires a successful Drive pull into an empty cache followed by an independently executed replay matching the checked summary; failure or missing authorization must remain explicit.

| Domain | Key | Repository-relative required path |
|---|---|---|
| **Standalone (`C20R`)** | `model_predictions` | `data/07_model_output/c0_model_improvement/c0_model_improvement_20261026_01/predictions.parquet` |
| | `model_protocol` | `data/05_model_input/c0_model_improvement/c0_model_improvement_20261026_01/protocol.json` |
| | `model_binding` | `data/05_model_input/c0_model_improvement/c0_model_improvement_20261026_01/execution_binding.json` |
| | `model_verification` | `data/08_reporting/c0_model_improvement/c0_model_improvement_20261026_01/sealed_verification.json` |
| | `serving_config` | `conf/base/c0_serving.json` |
| **OddsPortal** | `odds_protocol` | `data/05_model_input/fresh20_oddsportal/fresh20_oddsportal_20261111_02/protocol.json` |
| | `odds_evaluation` | `data/08_reporting/fresh20_oddsportal/fresh20_oddsportal_20261111_02/evaluation/oddsportal_annual_endpoint_ledger.parquet` |
| | `odds_binding` | `data/08_reporting/fresh20_oddsportal/fresh20_oddsportal_20261111_02/evaluation/execution_binding.json` |
| | `odds_quality_reference` | `data/08_reporting/fresh20_oddsportal/fresh20_oddsportal_20261111_02/evaluation/probability_quality.parquet` |
| | `odds_evaluation_receipt` | `data/08_reporting/fresh20_oddsportal/fresh20_oddsportal_20261111_02/evaluation/evaluation_receipt.json` |
| | `events` | `data/04_feature/long_term_market_edge/long_term_market_edge_20261022_01/oddsportal_identity_complete_v3/events.parquet` |
| **Pinnacle** | `pinnacle_ledger` | `data/07_model_output/pinnacle_model_comparison/pinnacle_model_comparison_20261112_043418/comparison_ledger.parquet` |
| | `pinnacle_scores_reference` | `data/08_reporting/pinnacle_model_comparison/pinnacle_model_comparison_20261112_043418/scores.csv` |
| | `pinnacle_summary_reference` | `data/08_reporting/pinnacle_model_comparison/pinnacle_model_comparison_20261112_043418/summary.json` |
| | `pinnacle_verification` | `data/08_reporting/pinnacle_model_comparison/pinnacle_model_comparison_20261112_043418/verification.json` |
| **Polish LOO** | `loo_panel` | `data/08_reporting/production_probability/production_probability_20261023_02/shrinkage/loo_matched_panel.parquet` |
| | `loo_case_accounting` | `data/08_reporting/production_probability/production_probability_20261023_02/shrinkage/loo_case_accounting.parquet` |
| | `loo_scores_reference` | `data/08_reporting/production_probability/production_probability_20261023_02/shrinkage/loo_consensus_scores.csv` |
| | `loo_contrasts_reference` | `data/08_reporting/production_probability/production_probability_20261023_02/shrinkage/loo_consensus_paired_contrasts.csv` |
| | `loo_verification` | `data/08_reporting/production_probability/production_probability_20261023_02/verification/sealed_verification.json` |
| **Opportunities & CLV** | `production_events` | `data/05_model_input/c0_gate_optimizer/c0_gate_optimizer_20261023_01/events.parquet` |
| | `production_configurations` | `data/05_model_input/c0_gate_optimizer/c0_gate_optimizer_20261023_01/configurations.json` |
| | `production_closing_states` | `data/08_reporting/production_zero_tax_clv/production_zero_tax_clv_20261023_01/clv/closing_states.parquet` |
| | `sampled_checks` | `data/08_reporting/audit/model_betting_reconciliation_20261113_01/sampled_checks.parquet` |
| | `sampled_suggestions_reference` | `data/08_reporting/audit/model_betting_reconciliation_20261113_01/checked_suggestions.parquet` |
| | `sampling_reference` | `data/08_reporting/audit/model_betting_reconciliation_20261113_01/results.json` |
| **Wallets & Legacy CLV** | `production_protocol` | `data/05_model_input/c0_gate_optimizer/c0_gate_optimizer_20261023_01/protocol.json` |
| | `production_fixtures` | `data/05_model_input/c0_gate_optimizer/c0_gate_optimizer_20261023_01/fixtures.parquet` |
| | `production_binding` | `data/05_model_input/c0_gate_optimizer/c0_gate_optimizer_20261023_01/binding.json` |
| | `production_verification` | `data/08_reporting/c0_gate_optimizer/c0_gate_optimizer_20261023_01/final_verification.json` |
| | `production_wallet_summary` | `data/08_reporting/c0_gate_optimizer/c0_gate_optimizer_20261023_01/sweep/summary.parquet` |
| | `production_selected` | `data/08_reporting/c0_gate_optimizer/c0_gate_optimizer_20261023_01/sweep/selected.parquet` |
| | `production_funded` | `data/08_reporting/c0_gate_optimizer/c0_gate_optimizer_20261023_01/sweep/funded_ledger.parquet` |
| | `production_entry_clv` | `data/08_reporting/production_zero_tax_clv/production_zero_tax_clv_20261023_01/clv/entry_clv_panel.parquet` |
| | `production_clv_verification` | `data/08_reporting/production_zero_tax_clv/production_zero_tax_clv_20261023_01/verification/sealed_verification.json` |
| **Glicko-2** | `glicko_predictions` | `data/artifacts/corrected-ratings-logloss-20260908-full/holdout_predictions.csv` |
| | `glicko_summary` | `data/artifacts/corrected-ratings-logloss-20260908-full/summary.json` |
| | `glicko_verification` | `data/artifacts/corrected-ratings-logloss-20260908-full/verification.json` |

**Domain checks enforced at replay:**
- **Standalone:** Verifies serving ID, binding hashes, and 3,103 rows (3,083 regional-supported + 20 absences).
- **OddsPortal:** Applies `_focal_quality` (earliest eligible focal quote per canonical fixture/endpoint) using year-eligible annual forecasts; reproduces 2 pooled quality rows within `1e-12`.
- **Pinnacle:** Checks verification PASS, 4,364 endpoint rows / 2,182 fixtures accounting, 14 score rows + primary intervals within `1e-12`. Bootstrap groups by **Pinnacle scheduled-start UTC month** (2 fixtures cross month boundaries vs canonical dates).
- **Polish LOO:** Verifies panel SHA-256 (`6cfb8c70...8486`) and accounting SHA-256 (`78ca41a5...e5af`), decodes JSON contributors (`≥2` distinct other books, `≤300s`, nonfuture), checks 51,208 matched + 49,741 excluded = 100,949 cases.
- **Opportunities & CLV:** Replays exact 12,000 checks (2,000/band) and unchanged zero-tax gates (`1e-12`); latest-only closing states require `< start` with no max age or after-entry constraint.
- **Wallets & Glicko-2:** Joins 12 saved wallet cases (330 settled rows); validates 10,578 already-BO-projected Glicko-2 series and 12 reference values.

## Bootstrap conventions

All domains use **5,000** monthly-block draws retaining zero-action months; ratio CIs withhold on any zero-denominator draw.

| Domain | Blocks | Seed | Primary estimand |
|---|---:|---:|---|
| `consensus` | 6 | 20261023 | Equal fixture (mean within book, then equal books) |
| `finance` | 70 | 20261111 | Equal-unit-stake ratio of totals (+ equal-active-month sensitivity) |
| `oddsportal_prediction` | 70 | 20261111 | Equal fixture (earliest eligible focal quote per endpoint) |
| `pinnacle_prediction` | 9 | 20261112 | Paired equal fixture (+ equal-month sensitivity) |
| `production_betting` | 6 | 20261023 | Saved-stake ROI ratio; equal-ticket closing proxies |
| `production_opportunities` | 6 | 20261113 | Selected-observation mean (fixed `[0,48)h` separate) |
| `ratings` | 30 | 20261111 | Equal series (+ equal-month sensitivity) |

## Verification performed

Ran offline with unreachable `DATABASE_URL` / `TEST_DATABASE_URL`:

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

Result: **115 passed** (1 existing Starlette/httpx warning). CLI replay at `/tmp/c20r-readme-evidence-replay-3` is byte-identical to [project_summary.json](results/project_summary.json) (SHA-256 `cdd7d8f420d04f07445dcf5d11f31841628924271083a5bc47ae4fc9407090c3`), matching 28 OddsPortal, 90 Pinnacle, 528 Polish-panel, and 12 Glicko references (max diff `1.11e-16`) and all 3,188 retained v3 numeric fields. See [PROJECT_REPORT.md](PROJECT_REPORT.md) for full results.
