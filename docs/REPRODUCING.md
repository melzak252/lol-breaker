# Reproducing the published project aggregates

## Saved-ledger replay

`scripts/reproduce_project_results.py` recomputes model scores from a saved prediction parquet and selects/summarizes endpoint-policy diagnostics from a saved OddsPortal ledger. It reuses the tracked selectors, scoring, expected-value, guard and month-bootstrap code. It does not train or infer a model, collect source data, connect to a database, or access the network. The committed [summary snapshot](results/project_summary.json) is output for readers, never a replay fallback.

From the repository root, in an environment that can import this checkout, run:

```bash
.venv/bin/python scripts/reproduce_project_results.py --output-dir /tmp/c20r-project-replay
```

The target directory must not already exist. The command writes aggregate-only `project_summary.json` and `replay_receipt.json`. Choose a different new path for another run. Bootstrap settings are fixed: 5,000 resamples, seed `20261111`, complete preselection UTC months January 2021–October 2026 (70 months).

### Environment prerequisite

The exercised repository `.venv` had Python 3.12.3, NumPy 2.4.4, pandas 3.0.3, PyArrow 24.0.0, SciPy 1.17.1 and scikit-learn 1.8.0. These versions are recorded in the snapshot. The direct Python dependencies used by this replay are NumPy, pandas, PyArrow, SciPy and scikit-learn; the source closure also imports the repository's model-engine, guard, EV, probability-metric and bootstrap modules. The archived October 26 model-improvement evaluation binding records a historical evaluation environment, including PyTorch 2.14.0+cpu; this is not evidence of the original C20R checkpoint's training environment. Replay does not run model inference or require CUDA. `requirements-betting.txt` instead pins PyTorch 2.5.1+cu121, and `uv.lock` is not a complete application/neural environment lock.

Check that an existing environment has the expected imports before running:

```bash
.venv/bin/python -c 'import numpy, pandas, pyarrow, scipy, sklearn; from src.analysis.long_term_market_evaluation import select_fixed_policy, summarize_unit_returns; from betting_app.core.ev import expected_value; print("saved-ledger imports available")'
```

A fresh environment installation was not exercised. Do not infer that installing the project's development extra or `uv.lock` alone provides a complete application or neural-training environment; no install command is claimed as verified here.

### Required artifacts and evidence checking

The replay requires these ignored/local files, relative to repository root:

- `data/07_model_output/c0_model_improvement/c0_model_improvement_20261026_01/predictions.parquet`, its protocol and execution binding under `data/05_model_input/c0_model_improvement/`, and sealed verification receipt under `data/08_reporting/c0_model_improvement/`.
- The `fresh20_oddsportal_20261111_02` protocol, evaluation ledger and execution binding under `data/05_model_input/fresh20_oddsportal/` and `data/08_reporting/fresh20_oddsportal/`.
- The paired later-quote proxy parquet under `data/08_reporting/fresh20_oddsportal/fresh20_oddsportal_20261111_02/samebook_shrinkage_sensitivity_01/`.
- The referenced events parquet under `data/04_feature/long_term_market_edge/long_term_market_edge_20261022_01/oddsportal_identity_complete_v3/`.

Model protocol-to-binding, prediction-to-sealed-verification, serving identity, and OddsPortal protocol/events-to-binding checks are performed by the script. The endpoint evaluation ledger and paired later-quote proxy are hashed into the receipt/snapshot but are **not** independently sealed or validated against an authenticated source by this replay. A matching SHA-256 shows byte identity with the published run, not data validity or result certification. The snapshot includes hashes for the artifacts and the replay's listed source closure (selector, score, bootstrap, probability-metric, engine, guard, EV, devig and audit-serialization modules); those hashes are inventory, not proof that the whole repository dependency graph or upstream evidence was independently certified.

These artifacts are not shipped in this GitHub repository; a clean clone alone is insufficient. The repository's recorded DVC remote does not establish that these later run artifacts are available there. Install the matching bound artifacts from an authorized source before replay. Missing required files fail explicitly—there is no fallback to checked-in aggregates. Public code, serving config and the aggregate snapshot are available from the checkout; raw input data, predictions, evaluation ledgers, event inventory and quote proxies remain artifact-required.

## Separate raw-to-forecast work

This replay starts from saved prediction and endpoint-evaluation ledgers. It does not reproduce raw-data acquisition, historical feature construction, neural training or original forecast generation. Those steps require their original inputs and checkpoint bundle, which are not supplied by this summary command. The repository does not claim clean-clone reproduction of neural fits or access to private/local data.