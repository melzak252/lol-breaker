# Model Catalog & Research Registry

**Updated:** 2026-10-22  
**Source of Truth:** [`conf/base/c0_serving.json`](../conf/base/c0_serving.json), [`betting_app/core/models/registry.py`](../betting_app/core/models/registry.py), and [`reports/`](../reports/).

This catalog indexes locally configured models and retained research. Local configuration and historical measurements do not establish live-server identity, production certification or a financial edge.

---

## 1. Active Operational Models & Strategies

| Role / Scope | Model Identifier | Specification & Mechanics | Primary Metrics | Status & Interface |
|---|---|---|---|---|
| **Local Sports Match-Winner** | **`PlayerHistory-Attention-Ensemble`** (`C0`) | **Only legacy 17-token C0** (`c0-native-2026-w32-e12-v1`, history16) is served locally. Long-anchor students remain separate research artifacts. | Same 3,103 exposed 2026 rows: LL **0.535449**, Brier **0.181034**, ECE10 **2.2855%**. | **Local native C0 only; unchanged weights.** |
| **Market-Assisted Local Hybrid** | **`Hybrid-50`** | 0.50/0.50 logit shrinkage; alpha applied once. | No prospective edge or EV-reliability certification. | Minimum-EV/dynamic hurdle, EV ceiling, DRO and other configured gates; optional EXP-039 comparison is not_evaluated when absent. |
| **Decision-time policy** | **Operational qualification** | True leave-one-out consensus from ≥2 **other** distinct books, divergence on pure C0, actual wallet bankroll in the app. | Frozen historical replay: 1 selected bet at300s; 5 at the separately predeclared24h sensitivity. | **HISTORICAL_DEVELOPMENT_ONLY; 0 authenticated admissions.** See current [research results](RESEARCH.md). |
| **Standalone Props: Map 1 Total Kills** | `prop_distribution_20261007_01` (Kills) | Regularized Ridge-1 Poisson-mean / Negative Binomial dispersion over 306 invariant pre-match pace features. | NLL **3.590488** (predictive-mean MAE 7.05 kills, 80% coverage 79.57%). | **`KEEP_BASELINE`**<br/>Bundle: [`data/06_models/prop_distribution/prop_distribution_20261007_01/`](../data/06_models/prop_distribution/prop_distribution_20261007_01/) |
| **Standalone Props: Map 1 Duration** | `prop_distribution_20261007_01` (Duration) | 15-leaf LightGBM / LogNormal distribution. | NLL **3.076194** (MAE 4.24 min, 80% coverage 75.99%). | **`KEEP_RESEARCH_CANDIDATE`**<br/>Code: [`src/models/prop_distribution.py`](../src/models/prop_distribution.py) |
| **Standalone Props: Series Length (BO3/5)** | `prop_distribution_20261007_01` (Length) | Width-128 / Epoch-12 3-seed neural distribution mixture. | NLL **0.714580**, RPS **0.261305** on 2,358 BO3/5 matches. | **`KEEP_RESEARCH_CANDIDATE`**<br/>Code: [`src/models/prop_distribution.py`](../src/models/prop_distribution.py) |
| **Tournament & Pick'Em Research Tools** | `tournament_numerical_20261007_01` + later research tools | Exact/hybrid integration and additional Swiss/draft/player-stat simulation implementations have different scopes. Verify edition rules, inputs and each tool's independent oracles. | Numerical integration is not evidence of calibrated fair odds, official Riot-lottery equivalence, Pick'Em superiority or tournament forecasting quality. | **Numerical/research implementation only, not forecast certification.** Code: [`src/models/advanced_tournament_simulator.py`](../src/models/advanced_tournament_simulator.py), [`src/models/crystal_ball_simulator.py`](../src/models/crystal_ball_simulator.py) |

### C0 edge-recovery audit — 2026-10-22 (bounded independent numerical audit complete; no financial certification)

Frozen protocol v2 and retained artifacts are indexed in [`docs/RESEARCH.md`](RESEARCH.md). Alpha, mechanics, recovery, the full genuine model CLI, the full policy runtime, and the bounded parent quote→core→F821 numerical verification have completed. The receipt covers only the documented numerical scope; it does not certify financial outcomes, prospective performance, production issuance or change the research manifest/serving.

The policy CLI completed in **842.91 s**. The primary matrix contains **350 primary configuration IDs** (including 140 core policy configurations) × four timing schedules × two splits = **2,800 primary case rows**. A separate continuous proper-score fit comparator adds **8 case summaries**, and the baseline noncausal oracle adds **2**, yielding **2,810 case summaries**. Three separate diagnostic configurations add **24 rows** in `source_safety_only_controls.parquet`, outside that 2,810; the **144 locked-model rows** and **8 market-extension rows** are separate as well. The TRAIN rule selected **zero primary configurations**: schedule support was **6/15/0/13** bets against a 20-bet/two-month minimum, so no ASSESS outcome was used to pick a policy. The earlier-only proper-score alpha fit **0.2853137251 sports / 0.7146862749 market** is research-only, not a selected policy. The source-safety-only/zero-DRO diagnostic outcomes and limits are summarized in the research journal; current native C0 configuration remains unchanged.

The unchanged α=0.5/full-gate ASSESS simulation did not lose across all schedules: first-ready **3 / +343.50 PLN**, quote-triggered **15 / +255.56 PLN (+13.17%)**, UTC-even **2 / −254.73 PLN**, and UTC-odd **12 / +268.73 PLN (+17.25%)**. Simulated net payout applies 12% tax to the full winning payout (`stake × odds × 0.88`), not only positive winnings. These are tiny two-month development outcomes, not a selected strategy or independently verified financial evidence; full policy selection chose no configuration. The raw quote tape's 81.47% even-hour concentration and minute55/56 updates can alias fixed UTC phases with the 300s TTL, but do not establish profit causality or C0 degradation. See the research journal for timing and bounded source-safety/DRO sensitivities.

Actual CLI scored all 1,126 cached daily states and all 910/910 focal rows (302 fixtures, zero absences); C0 matched sealed probabilities with maximum numeric difference0.0. For the restricted 194-fixture/two-month ASSESS, alpha-.5 log loss was C0 **0.597410939**, Core18/C0-D **0.597575949**, Core17 **0.597779176**, Slim18 **0.599706162**, fresh18 **0.597587741**, fresh20 **0.597645916**. None point-beat C0 in this cohort; not a general ranking or prospective result.

“Core” has two historically distinct meanings: registry/manifest **C0-D/Core18** is the same current artifact; Track-1 `student_w32_kd` **Core17** is a separate 17-token student. Slim18 is single-seed with its own Beta calibration. Named-matchup slope surrogates and cached serving transforms are not genuine neural inference. This full bounded research CLI has run; no live adapter, source activation, promotion, or serving change occurred.

The strict first-ready scored cohort is 302 fixtures (83 TRAIN,25 boundary-excluded,194 ASSESS); full1771 inventory is retained. The pre-overwrite 50-row CSV was recovered from DVC and matches the retained rewrite on all50 row IDs/order and recorded dates, participants, bookmaker labels, odds, C0/thesis probabilities, CLV, stakes, outcomes and bankroll fields; all50 hybrid probabilities/model-EV values were overwritten. Source-opportunity exact joins remain0 because canonical match/side/quote/book IDs are absent; this is not proof of 50 independently source-verified bets. The 43 trajectory IDs are support/coverage, not a standalone 43-ticket ledger. The Oct5 production export is a read-only transaction receipt, not a live query or record of issued predictions/wallet actions. Authenticated source remains0, true CLV null, optional EXP/rating not_evaluated, G0 blocked. No promotion, live wallet, financial certification or forward-date change.

### Completed raw-source C0-D reconstruction study — 2026-10-22

[`c0d_repro_20261021_01`](../data/05_model_input/c0_distilled_verification/c0d_repro_20261021_01/protocol.json) is a separately frozen new-bank experiment, not retrospective certification of the old anchor arrays. Six genuine student fits, zero C0/teacher refits, the same3,103 development-exposed assessment rows, native33-coordinate strictly prior raw-map summaries and fixed reference assembly.

| New research model | Log loss | Brier | Frozen decision |
|---|---:|---:|---|
| Fresh18-token | 0.534094945 | 0.180454469 | NOT_PROMOTED: primary interval crosses0; annual-A0 guards fail |
| Fresh20-token | 0.533925172 | 0.180375716 | NOT_PROMOTED: annual-A0 guards fail despite negative corrected primary interval |

Both pass all six C0 guards; both fail annual-A0 BO5/LPL/unseen-player noninferiority in the actual24-cell corrected family. The20-token match-weighted97.5% ΔLL interval versus C0 is [−0.002945750,−0.000047854], but its equal-month interval crosses0. Better point scores are a development signal, not promotion, bookmaker superiority or financial reliability. Full [results and guards](../data/08_reporting/c0_distilled_verification/c0d_repro_20261021_01/fresh/results.json), [saved-weight proof](../data/08_reporting/c0_distilled_verification/c0d_repro_20261021_01/fresh/saved_inference_verification.json), and [research limitations](RESEARCH.md).

The new weights/calibration stay under [`data/06_models/c0_distilled_verification/c0d_repro_20261021_01/fresh/`](../data/06_models/c0_distilled_verification/c0d_repro_20261021_01/fresh/). Actual public research inference is documented in [`scripts/README.md`](../scripts/README.md); no live roster adapter, registry activation, canonical replacement or deployment occurred. Original saved18/20-token probabilities also replay exactly, but their bank lineage remains unverified. The historical family table below retains its old scores/claims separately. “Core” is ambiguous: manifest/registry C0-D/Core18 is one artifact, while Track-1 `student_w32_kd` Core17 is a different student; Slim18 is single-seed with its own Beta calibration. The edge-recovery full-model and policy CLIs completed, and the bounded numerical audit passed within its documented scope; this is not financial or production certification.

### 1.1 Standard C0 Model Family (`C0`, `C0-D`, `C0-Slim` / `C0-D1`, `C0-Mini`)

The historical 18-token research variants add a 365-day anchor to the native recent-map representation. This is not the served 17-token C0, and the old anchor bank's causal lineage remains unverified. A separately versioned raw-map reconstruction is a new study, not retroactive reproduction of that bank.

> **[RESEARCH-ONLY / NOT SERVED / NOT ADOPTED]** The app serves legacy 17-token C0. C0-D registry labels are not operational models; the named-matchup CLI defaults to C0 and requires explicit `--allow-research-surrogate` for old BO slopes. Genuine frozen 18/20-token weights are exercised through `src.models.c0_distilled_replay`, separately from the fresh causal-bank study and without activating serving.

> **[WITHDRAWN / UNVERIFIED]** Old financial yields, Net CLV, and “Zero-Disappointment Guarantee” are not validated evidence. The reported 43-bet results are not an independently verifiable 43-ticket ledger: the 43 trajectory IDs are support/coverage, while 43-row claims were derived from/rescored using the 50-row ledger. The pre-overwrite 50-row version matches the retained rewrite on recorded chronology/cash fields, but all hybrid probabilities/model-EV were overwritten and exact source-opportunity joins remain0 because canonical IDs are missing; the Oct20 rewrite used ambiguous same-date/book/odds matching. Reported Sharpe is a t-statistic, not annualized Sharpe. Normalized LL 0.533600/0.533779 is look-ahead contaminated. The historical reports' claimed six guards remain unverified; a new frozen study does not certify those old claims.

More than 20 variants were compared on exposed 2026 assessment rows without multiplicity correction. C0-D 18T interval `[-0.002562, +0.000167]` and C0 18T interval `[-0.001195, +0.000039]` cross zero. No adoption or prospective edge is established.

Parameter and latency figures in the historical table below are retained report values, not new performance measurements. Genuine research CLI cold-load time includes input/weight closure verification and is not a resident-worker latency benchmark.

Standardized short codes, wire identifiers, neural parameter counts, inference latencies, and canonical 2026 assessment metrics ($N=3,103$, 9 monthly blocks):

| Short Code | Wire Identifier | Display Name | Neural Params | Inference Latency | LogLoss (2026) | Brier Score | ECE | Replacement Guards Status | Primary Role & Operational Status |
|---|---|---|---|---|---|---|---|---|---|
| **`C0`** | `Causal-C0`<br/>(`c0-year-18token-w32-e12-v1` / `c0-native-2026-w32-e12-v1`) | `PlayerHistory-Attention-Ensemble` | 19,914 (3 seeds `W32/E12`, 6,638/seed) | 848 ms | **0.534777** [RESEARCH-ONLY]<br/>(Legacy Pin: 0.535449) | **0.180842** [RESEARCH-ONLY]<br/>(Legacy: 0.181034) | **2.17%** [RESEARCH-ONLY]<br/>(Legacy: 2.29%) | Research 18T: 9 tails; served artifact is legacy 17-token C0 | **`ACTIVE_DEFAULT_REFERENCE`**<br/>Served local model: legacy 17-token C0 only.<br/>Config: [`conf/base/c0_serving.json`](../conf/base/c0_serving.json) |
| **`C0-D`** | `Causal-C0-D`<br/>(`c0-d-born-again-w32-v1`) | `BornAgain-Distilled-Attention-Ensemble` | 19,914 (3 seeds Born-Again `W32-KD`, 6,638/seed) | 848 ms | **0.534134** pure [RESEARCH-ONLY]<br/>(**0.533984** as `C0-D+` with Conn+ONS) | **0.180468** pure [RESEARCH-ONLY]<br/>(0.180381 with Conn) | **1.87%** [RESEARCH-ONLY] | **[UNVERIFIED]** Historical guards not certified; primary interval crosses zero | **`NOT_ADOPTED / RESEARCH_ONLY`**<br/>Original 18-token bank lineage unverified; genuine saved inference is not operational activation.<br/>Financial claims withdrawn / unverified. |
| **`C0-Slim`**<br/>(alias `C0-D1`) | `Causal-C0-Slim`<br/>(`c0-slim-1seed-w32-v1`) | `SingleSeed-Distilled-Attention` | **6,638** (1 seed `W32/E12`, 1/3rd size) | **278 ms**<br/>(**3.1x faster**) | **0.534656** standalone [RESEARCH-ONLY] | **0.180667** standalone [RESEARCH-ONLY] | **1.92%** [RESEARCH-ONLY] | **[UNVERIFIED]** Not computed by repository evaluation | **`NOT_ADOPTED / RESEARCH_ONLY`**<br/>Research candidate; not served.<br/>Financial claims unverified / by construction. |
| **`C0-Mini`**<br/>(alias `C0-Small`) | `Causal-C0-Mini`<br/>(`c0-mini-w16-kd-v1`) | `HalfWidth-Distilled-Attention` | **8,394** (`2,798`/seed, 58% smaller width) | 729 ms | **0.535845** [RESEARCH-ONLY]<br/>($\Delta\text{LL} = -0.000800$ vs `W16-Direct`) | 0.181188 [RESEARCH-ONLY] | **2.13%** [RESEARCH-ONLY] | **[UNVERIFIED]** Not computed by repository evaluation | **`NOT_ADOPTED / RESEARCH_ONLY`**<br/>Research candidate; not served. |

#### Architectural Specifications & Guard Details
- **`C0` (`PlayerHistory-Attention-Ensemble`, `Causal-C0`)**:
  - Historical 18-token research specification: 3 seeds (`20260910`, `20260911`, `20260912`), Width-32 / Epoch-12, 365-day anchor plus recent-map inputs, `19,914` neural params (`6,638`/seed). The served native C0 remains the separate history16/17-token artifact.
  - Metrics: LogLoss **0.534777**, Brier **0.180842**, ECE **2.17%**, Tails **9**, inference latency **848 ms**. Legacy 17-token `C0-16m` baseline (LogLoss `0.535449`, 12 tails) remains pinned for historical parity.
- **`C0-D` (`BornAgain-Distilled-Attention-Ensemble`, `Causal-C0-D`)**:
  - Specification: 19,914 neural params across 3 seeds Born-Again `W32-KD` distilled from the 9-checkpoint teacher ensemble with $\lambda = 0.60$ soft loss over the 18-token multi-horizon bank, **zero external tree/connector dependencies**.
  - Metrics: Development LL **0.534134**, Brier **0.180468**, ECE **1.87%**. The reported interval vs legacy C0 `[-0.002562, +0.000167]` crosses zero. The historical report's six-guard claims remain unverified; separate new-study guards do not repair that provenance.
  - Connector variant (`C0-D+`): historical point LL 0.533984; development-only, not served or adopted.
  - **20-token upgrade (`C0-D-20T`):** development LL 0.533954; normalized 0.533600 is **[WITHDRAWN: look-ahead]**. The development result is unadjusted, not an adoption result.
- **[WITHDRAWN / UNVERIFIED] Historical financial claims:** reported 43-row returns/Net CLV and “Zero-Disappointment Guarantee” are not validated; the 43 trajectory IDs do not establish an independent ticket ledger. The 43-row account was derived from/rescored using the 50-row ledger. Pre-overwrite versus rewritten 50-row comparison preserves recorded chronology/cash fields, but hybrid probabilities/model-EV were overwritten and source-opportunity exact joins remain0 due missing canonical IDs (audit §5). Reported Sharpe is a t-statistic.
- **`C0-Slim` / `C0-D1` (`SingleSeed-Distilled-Attention`, `Causal-C0-Slim`)**:
  - Specification: Single seed (`20260910`) Distilled 18-Token Width-32 model with **6,638 neural params** (exactly 1/3rd parameter count of the 3-seed ensemble), achieving **3.1x faster** inference latency (**278 ms** vs **848 ms**).
  - Metrics: Development LL 0.534656 (18-token) / 0.534621 (20-token), Brier 0.180667 / 0.180648, ECE 1.92%, Tails 13 (development-only; interval vs C0 crosses zero).
  - **[UNVERIFIED / BY CONSTRUCTION]** Financial claims in historical reports: Polish and Pinnacle yields/Net CLV are not validated; Pinnacle Net CLV is 0%/−12% by construction in cited code. Reported Sharpe is a t-statistic. Not evidence of profitability.
- **`C0-Mini`:** Development LL 0.535845. **[UNVERIFIED]** The claimed “passes all 6 guards” was not computed for that historical recipe.

- **Historical cached cross-regional input transform (`src/models/cross_region_input_normalizer.py`, research-only)**:
  - Off by default in named-matchup prediction; requires `--research-cross-region-norm`. Its global cache is not an authenticated as-of input or a validated correction, and is not used by the fresh causal-bank study.
  - Dynamically loads live `FamilyCalibratedGlicko2` (`ratings-v2`) international bridge posteriors (`families`, `tiers`) and active-player domestic pyramid medians (`players`) from `data/04_feature/live_c0_cache/regional_offsets_state.json` (rebuilt every 6 hours by `rebuild_regional_ratings` in `heavy_maintenance_cycle`).
  - **[WITHDRAWN: look-ahead]** Historical normalized LL/slice improvements used one current offset state retrospectively across 2024–2026; they are not causal validation (audit §3.1).

### 1.2 Decision-Time Gate Simplification & Data Orientation Invariants

**[WITHDRAWN as a description of the running system — audit 2026-10-21 §2]** This historical research protocol does not describe the implemented policy. The repaired local hybrid uses α = 0.50 and keeps the dynamic minimum EV, maximum EV, DRO and other configured gates. EXP-039 is optional and explicitly not_evaluated when absent. Local code/CLI regression and the new frozen replay exercise true leave-one-out from at least two **other** books, divergence on pure C0 and actual-wallet/free-capital rules; these are not live-source or profitability certification. See the dated [research journal](RESEARCH.md).
1. **Historical LOO Market Shrinkage (α = 0.65):** the equation below is retained as a description of the former research protocol, not current behavior:
   $$\text{logit}(p_{\text{hybrid}}) = 0.35 \cdot \text{logit}(p_{\text{C0}}) + 0.65 \cdot \text{logit}(q_{\text{open, loo}, -k}(t))$$
2. **Historical DRO Gate:** the former fixed haircut formula is research description, not proof of a Wasserstein guarantee:
   $$\text{EV}_{\text{DRO}} = (p_{\text{hybrid}} - 0.025) \cdot (1 - \tau) \cdot O - 1 > 0$$
3. **Historical Divergence Gate:** the historical rule is not an implementation guarantee:
   $$|\text{logit}(p_{\text{C0}}) - \text{logit}(q_{\text{open, loo}})| \le 0.85$$
4. **Historical Odds Ceiling:** the research protocol recorded:
   $$\text{Odds} \le 4.00$$

**[WITHDRAWN — these filters are not retired; all are active in the running policy]**
- `Max EV <= 25%`: Active.
- `Min EV >= 5%`: Active.
- `Model-Consensus-Veto` (`EXP-039`): Active in policy; audit found an implementation defect at the audited commit, see §1.2 correction.

#### Historical counterfactual alignment proposal — not the current policy
The three joins/batching rules below are retained from the withdrawn historical ledger account, not mandated for the new frozen evaluation. Its 2h batches/12h freshness and float-coerced identity expressions do not define the current T−1h/300s protocol or certify historical alignment:
1. **Quote Alignment Join**: Join `data/08_reporting/c0_production_counterfactual/c0_production_20261006_01/quote_alignment.parquet` (`quote_identity_status == 'verified'`) and orient `odds_1, odds_2` strictly via boolean `raw_a_is_team1`.
2. **Identity Alignment Join**: Join `data/08_reporting/c0_production_counterfactual/c0_production_20261006_01/identities.parquet` (`identity_status == 'verified'`, `date_within_one_day != False`) and align model prediction probabilities (`year_anchor_distilled_predictions.parquet`) via `str(int(float(ident.team1_id))) == str(int(float(p26.team1_id)))`.
3. **Batch Execution & LOO Consensus**: Group evaluations strictly at the end of each 2-hour cron scrape batch (`bucket_2h = scraped_dt.dt.floor('2h')`) with a `freshness <= 12h` window and compute Leave-One-Out consensus $q_{\text{open, loo}, -k}$ across the other active bookmakers excluding the focal bookmaker.

These historical expressions are not a guarantee of correct identity, publication clocks or absence of leakage. The new execution binds exact canonical participant orientation, latest quote state, its own fixed origin/freshness, genuine native inputs and explicit absences; see the current frozen protocol and independent receipt.

---

## 2. Canonical Research Controls & Baselines

| Model Identifier | Target / Scope | Architecture & Method | Benchmarks | Role & Governance |
|---|---|---|---|---|
| **`WalkForward-Reference-Ensemble`** (`A0`) | Historical match winner | Annual walk-forward causal ensemble (7 base checkpoints across 3 experts; quarterly cross-fitted calibration). Wire ID: `p_full` / `Annual Causal A0`. | Full 11,550 series (2024–2026): **LogLoss 0.551246**, Brier **0.187103**.<br/>Common 9,907 series: **0.559920**. | **`KEEP_CONTROL` (Canonical Benchmark Reference)**<br/>Pinned in [`conf/base/research_benchmark.json`](../conf/base/research_benchmark.json). Evaluated via `scripts/run_model_benchmark.py --suite --doctor`. |
| **`ElasticNet-TeamForm-Baseline`** (`EXP-039`) | Thesis reference & macro veto guard | Map-level symmetric calibrated ElasticNet with $W20$ team form + binomial BO3/BO5 series projection. Wire ID: `p_039` / `exp-039`. | Common 9,907 series: LogLoss **0.567337** (16 deep tail losses). Missing 1,643 series lacking historical W20. | **`KEEP_CONTROL` (Immutable Thesis Baseline)**<br/>Preserved in `betting_app/models/sym_cal_lr_elasticnet_w20_binomial_pipeline.joblib`. Evaluated as independent sanity check in Model-Consensus-Veto. |
| **`Player-Glicko2-Rating-Baseline`** (`Glicko-2`) | Rating baseline | Match-level player rating engine with rating deviation (RD), volatility tracking, and logistic link. Wire ID: `p_glicko`. | Full 11,550 series: LogLoss **0.575888**, Brier **0.196548**.<br/>Common 9,907 series: **0.577998**. | **`KEEP_CONTROL`**<br/>Baseline rating control; evaluated in benchmark manifest. |
| **`Player-Elo-Rating-Baseline`** (`Elo`) | Rating baseline | Match-level calibrated Elo with player-level K-factor optimization. Wire ID: `p_elo`. | Full 11,550 series: LogLoss **0.588946**, Brier **0.201930**.<br/>Common 9,907 series: **0.591198**. | **`KEEP_CONTROL`**<br/>Baseline rating control; evaluated in benchmark manifest. |

---

## 3. Preserved Standalone Prop Model Recovery Bundle

The standalone prop model is preserved in a byte-exact, self-contained recovery archive:

- **Location:** `data/06_models/model_preservation/prop_distribution_20261007_01.zip` (63.4 MB, 154 members)
- **SHA-256:** `a115b26576b90b32997536aa9e1ee7eb1228dd6864bfa53e16e671b72982a232`
- **Selected Model Hash:** `9260e43dcd926a662c75a46d3e747f0629757fbedb061207c9234542570d110b`
- **Verification CLI:**
  ```bash
  .venv/bin/python scripts/preserve_prop_model.py verify \
    --bundle data/06_models/model_preservation/prop_distribution_20261007_01.zip \
    --sha256 a115b26576b90b32997536aa9e1ee7eb1228dd6864bfa53e16e671b72982a232
  ```

---

## 4. Completed Empirical Studies & Archived Research

> **Audit correction (2026-10-21):** C0-D/18T/20T is DEVELOPMENT_CANDIDATE, not adopted: 18T intervals cross zero, no prospective result exists, and nominally excluding-zero comparisons are unadjusted development results. The app serves only legacy 17-token C0; historical “ADOPTED” rows below are withdrawn. [Audit](../review-reports/codebase_logic_audit_20261021.md)

All historical studies are listed below with their recorded disposition; evidence has not been independently reproduced here.

| Study Identifier | Research Objective | Outcome & Scientific Verdict | Evidence & Full Report |
|---|---|---|---|
| `c0_upgrade_and_distillation_20261017` | Multi-track C0 upgrade & knowledge distillation: Track 1 Knowledge Distillation (W16/W32 KD students), Track 2 Role-Aligned Lane & Query Attention, Track 3 Continuous Champion Archetypes + 15% UNK Dropout, Track 4 Disagreement-Gated Mixture + Online Newton Step Recalibration, followed by 18-Token Multi-Horizon sequence consolidation (`16 Recent Maps + 365-Day Whole-Year Anchor Token + Empty Token`). | **[WITHDRAWN / NOT ADOPTED]** Distilled family standardization: historical 18-token architecture development candidate (LogLoss **0.534777**, 9 tails). Pure 3-seed 18-token distilled candidate **`C0-D`** (LL **0.534134**, ECE **1.87%**; financials unverified / withdrawn), **`C0-Slim`** / **`C0-D1`** (LL **0.534656**, `278 ms`, `6,638` params; financials unverified / by construction), and **`C0-Mini`** (LL **0.535845**, 58% smaller width). The app serves only legacy 17-token C0. | [`data/08_reporting/c0_upgrade_and_distillation_20261017/`](../data/08_reporting/c0_upgrade_and_distillation_20261017/)<br/>[`docs/RESEARCH.md`](RESEARCH.md#2026-10-18--c0-knowledge-distillation--multi-track-upgrade-execution-c0-d-c0-slim-c0-mini) |
| `optimal_synergy_c0_thesis_20261012` | Historical C0 hybrid ledger (reported α=0.65; documentation inconsistent). | **NOT ADOPTED / UNVERIFIED**: 50 bets, yield and Net CLV are not reproducible repo evidence; see independent audit. | [`reports/optimal_synergy_c0_thesis_simulation_20261012.md`](../reports/optimal_synergy_c0_thesis_simulation_20261012.md) |
| `oddsportal_multiyear_20261012` | Historical multi-year simulation. | **UNVERIFIED / NOT VALIDATION**: repo multi-year code bets at de-vigged fair odds; reported Net CLV is 0%/−12% by construction, and C0 years overlap TRAIN. | [`reports/oddsportal_open_close_simulation_20261012.md`](../reports/oddsportal_open_close_simulation_20261012.md) |
| `c0_monthly_refit_20261011` | Historical quote census and model comparison. | **UNVERIFIED**: point LL/Raw CLV differences are not prospective edge evidence. | [`reports/monthly_refit_c0_production_odds_since_may_20261011.md`](../reports/monthly_refit_c0_production_odds_since_may_20261011.md) |
| `c0_tournament_copula_20261007_01` | Player-shared Gaussian copula over complete tournament Monte Carlo paths. | **NOT_ADOPTED**: CAL path NLL slightly improves, but champion log-loss and placement RPS degrade. CAL guards fail. | [`reports/reliable_tournament_simulation_research_20261006.md`](../reports/reliable_tournament_simulation_research_20261006.md) §14 |
| `champion_history_20261007_01` | Multitask representation learning for joint winner and prop prediction. | **NOT_ADOPTED**: CAL-selected winner LogLoss 0.546321 vs C0 0.535449. Failed replacement gates. | [`reports/champion_history_prediction_20261007.md`](../reports/champion_history_prediction_20261007.md) |
| `c0_meta_stable_20261007_01` | 180-day player support profiles with 30-day role-meta features. | **NOT_ADOPTED**: Both calibration amplitudes zero; representations yield identity C0 probabilities. | [`docs/RESEARCH.md`](RESEARCH.md#2026-10-07--c0-meta-stable-follow-up) |
| `c0_ev_loss_20261006_01` | Training sports betting heads with after-tax return-MSE objective vs likelihood. | **NOT_ADOPTED**: Multiplicity-adjusted risk CI $[-0.0008, +0.0016]$ crosses zero; after-cost guards fail. | [`reports/c0_selected_betting_reliability_research_and_audit_20261006.md`](../reports/c0_selected_betting_reliability_research_and_audit_20261006.md) §11 |
| `native_capacity_20261004_01` | Scaling model capacity to Width 64 and Epoch 24 + knowledge distillation. | **NOT_ADOPTED**: Bonferroni confidence intervals cross zero; distillation skipped by evaluation gate. | [`reports/model_targets_capacity_and_distillation_research_20261004.md`](../reports/model_targets_capacity_and_distillation_research_20261004.md) |
| `c0_raw_connector_20261002_033658` | Learned specialist connector combining C0 with fresh logistic/tree ensemble. | **NOT_ADOPTED**: Primary LL contrasts negative, but BO5 and source-unseen replacement guards fail. | [`reports/plan09_c0_connector_and_forward_evidence_20261002.md`](../reports/plan09_c0_connector_and_forward_evidence_20261002.md) |
| `reliability_tradeoff_20261001` | Comparing Native-6 vs Native-12 epochs for calibration-first trade-off. | **NOT_ADOPTED**: Native-6 LogLoss worsens by $+0.000867$; RMS calibration gain below required $0.5$ pp. | [`reports/noisy_data_reliability_research_20260929.md`](../reports/noisy_data_reliability_research_20260929.md) §13 |
| `a0_data_focus_20260929_192443` | High-tier / market-tier neural filtering and history token dropout. | **NOT_ADOPTED** (Challengers): Uniform control is retained as the **active Native C0 baseline**. | [`reports/bookmaker_noise_and_provider_training_20260927.md`](../reports/bookmaker_noise_and_provider_training_20260927.md) §10 |
| `native_refit_20260929_122617` | Exponential time decay (H365) vs uniform weighting on neural history. | **NOT_ADOPTED**: H365 worsens performance across all cadences. Retain uniform history weighting. | [`reports/bookmaker_noise_and_provider_training_20260927.md`](../reports/bookmaker_noise_and_provider_training_20260927.md) §9 |

---

## 5. Developer Navigation & Documentation Map

- **Comprehensive Empirical Journal:** [`docs/RESEARCH.md`](RESEARCH.md) — Exhaustive statistical receipts, CI bounds, and verification logs.
- **Architectural Specification & Contracts:** [`docs/PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md) — Data contracts, schema definitions, and process isolation.
- **Master Research Reports Index:** [`reports/README.md`](../reports/README.md) — Index of all 40+ research papers and econometric audits.
- **Application Backend Contracts:** [`betting_app/README.md`](../betting_app/README.md) — API routes, scrapers, models, and local test instructions.
- **Executable Script Reference:** [`scripts/README.md`](../scripts/README.md) — Command-line interface definitions and usage guides.
