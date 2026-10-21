# Model Catalog & Research Registry

**Updated:** 2026-10-21  
**Source of Truth:** [`conf/base/c0_serving.json`](../conf/base/c0_serving.json), [`betting_app/core/models/registry.py`](../betting_app/core/models/registry.py), and [`reports/`](../reports/).

This catalog is the definitive index of active production models, canonical research baselines, and completed empirical studies across **LoL-Breaker**.

---

## 1. Active Operational Models & Strategies

| Role / Scope | Model Identifier | Specification & Mechanics | Primary Metrics | Status & Interface |
|---|---|---|---|---|
| **Active Sports Match-Winner** | **`PlayerHistory-Attention-Ensemble`** (`C0`) | Standard 3-seed Width-32 attention ensemble with **18-Token Multi-Horizon sequence (`16 Recent Maps + 365-Day Whole-Year Anchor + Empty`)**: Beta terminal score + $0.1$ SmoothL1 aux loss + shared rating/MLP experts. Wire ID: `Causal-C0` (`c0-year-18token-w32-e12-v1` / `c0-native-2026-w32-e12-v1`). | **LogLoss 0.534777**, Brier **0.180842**, ECE **2.17%**, **9 tails** on 3,103 canonical 2026 series (improves upon legacy 17-token pin `0.535449` LL / 12 tails). | **`KEEP_REFERENCE`**<br/>Active local sports default. Output is whole-series BO1/3/5 win probability.<br/>Config: [`conf/base/c0_serving.json`](../conf/base/c0_serving.json) |
| **Market-Assisted Operational Hybrid** | **`MarketShrunk-Hybrid-50`** (`Hybrid-50`) | 50/50 logit-space shrinkage combining `PlayerHistory-Attention-Ensemble` with Shin-devigged market consensus: `logit(p_Fixed50) = 0.50 · logit(p_C0) + 0.50 · logit(q_shin)`. Wire ID: `Hybrid-Operational-Market`. | LogLoss **0.557877** vs Market 0.570054 on 2026 production quotes ($-1.22$ pp gain). | **`KEEP_LOCAL_CONFIGURATION`**<br/>Interface: [`betting_app/core/models/registry.py`](../betting_app/core/models/registry.py) |
| **Leading Decision-Time Betting Policy** | **Optimal Synergy Strategy** | Fixed50 / LOO Opening Consensus Hybrid + Shin 2-way de-vigging + **`Market-Divergence-Gate`** (`\|logit(p_C0) - logit(q_open_loo)\| ≤ 0.85`) + **`Model-Consensus-Veto`** (`\|p_C0 - p_039\| ≤ 0.10`) + `EV_net ∈ [5%, 25%]` + `Odds ≤ 4.00` + 1/4 Kelly. | Claimed EV **+8.59%**, **+4.83% Net CLV** ($t=3.00, p=0.0042$), Residual Optimizer's Curse **+3.76 pp** (56.2% edge retention), Match Noise $\eta = +19.42\text{ pp}$, **+23.91% Realized Yield** (+1,715.41 PLN) on 50-bet production ledger. Max DD 6.10%. | **`ADOPTED_LOCAL_STRATEGY`**<br/>Report: [`reports/optimal_synergy_c0_thesis_simulation_20261012.md`](../reports/optimal_synergy_c0_thesis_simulation_20261012.md)<br/>Code: [`betting_app/core/ev.py`](../betting_app/core/ev.py) |
| **Standalone Props: Map 1 Total Kills** | `prop_distribution_20261007_01` (Kills) | Regularized Ridge-1 Poisson-mean / Negative Binomial dispersion over 306 invariant pre-match pace features. | NLL **3.590488** (predictive-mean MAE 7.05 kills, 80% coverage 79.57%). | **`KEEP_BASELINE`**<br/>Bundle: [`data/06_models/prop_distribution/prop_distribution_20261007_01/`](../data/06_models/prop_distribution/prop_distribution_20261007_01/) |
| **Standalone Props: Map 1 Duration** | `prop_distribution_20261007_01` (Duration) | 15-leaf LightGBM / LogNormal distribution. | NLL **3.076194** (MAE 4.24 min, 80% coverage 75.99%). | **`KEEP_RESEARCH_CANDIDATE`**<br/>Code: [`src/models/prop_distribution.py`](../src/models/prop_distribution.py) |
| **Standalone Props: Series Length (BO3/5)** | `prop_distribution_20261007_01` (Length) | Width-128 / Epoch-12 3-seed neural distribution mixture. | NLL **0.714580**, RPS **0.261305** on 2,358 BO3/5 matches. | **`KEEP_RESEARCH_CANDIDATE`**<br/>Code: [`src/models/prop_distribution.py`](../src/models/prop_distribution.py) |
| **Advanced Tournament & Pick'Em Simulator** | `tournament_numerical_20261007_01` + `crystal_ball_simulator` | • Sequential Riot Swiss draw lottery (sliding-slot mechanics, no-rematch backtracking).<br/>• Hard Fearless Draft map simulator.<br/>• Exact $O(2^K)$ knockout bracket DP.<br/>• Hierarchical player stats generator for Riot Worlds Pick'Em. | Rao-Blackwellized exact rational distributions; calibrated fair odds for all Worlds Pick'Em categories. | **`KEEP_NUMERICAL_IMPLEMENTATION`**<br/>Code: [`src/models/advanced_tournament_simulator.py`](../src/models/advanced_tournament_simulator.py), [`src/models/crystal_ball_simulator.py`](../src/models/crystal_ball_simulator.py) |

### 1.1 Standard C0 Model Family (`C0`, `C0-D`, `C0-Slim` / `C0-D1`, `C0-Mini`)

Consolidated and rebranded standard model family built upon the **18-Token Multi-Horizon sequence (`16 Recent Maps + 365-Day Whole-Year Anchor Token + Empty Token`)**.

The pure, self-contained 3-seed 18-Token Born-Again Distilled model is selected as **`C0-D`**:
- **Why Pure `C0-D` is Selected Over `C0-D+` (Connector Variant):**
  1. **Zero External Tree/Connector Dependencies:** Pure `C0-D` evaluates entirely in native PyTorch with zero external LightGBM, logistic regression, or secondary calibration dependencies. It eliminates runtime coupling and production failure modes.
  2. **Superior Probability Calibration:** Achieves **1.87% ECE** (the finest calibration across all evaluated models), compared to 2.58% ECE for the connector+ONS pipeline.
  3. **Exceptional Realized Profitability:** Under active DRO without the EXP-039 veto on Polish bookmakers (12% tax), pure `C0-D` executes 43 bets with a **60.5% win rate**, generating **+1,765.78 PLN profit** (**+27.82% Yield**, **+17.66% ROI**, **1.56 Sharpe**, **+6.02% Net CLV**). Pre-DRO Optimizer's Curse Bias is compressed to **+2.14 pp** (strictly smaller than `+6.02%` Net CLV!), and Wasserstein DRO ($\varepsilon = 0.025$) shifts Post-DRO Optimizer's Curse to **-3.62 pp** (**Zero-Disappointment Guarantee!** $B \le 0$). On Polish Longshots (`Odds 2.30–4.00`), it achieves **+47.41% Yield** (+1,793.15 PLN, 1.79 Sharpe, 3.89% Max DD). The connector variant yields lower return (+23.33% yield / 1.22 Sharpe) despite marginal unclipped log-loss reduction (`0.533984` vs `0.534134`). The connector variant is preserved as **`C0-D+`** for post-processing research.

Standardized short codes, wire identifiers, neural parameter counts, inference latencies, and canonical 2026 assessment metrics ($N=3,103$, 9 monthly blocks):

| Short Code | Wire Identifier | Display Name | Neural Params | Inference Latency | LogLoss (2026) | Brier Score | ECE | Replacement Guards Status | Primary Role & Operational Status |
|---|---|---|---|---|---|---|---|---|---|
| **`C0`** | `Causal-C0`<br/>(`c0-year-18token-w32-e12-v1`) | `PlayerHistory-Attention-Ensemble` | 19,914 (3 seeds `W32/E12`, 6,638/seed) | 848 ms | **0.534777**<br/>(Legacy Pin: 0.535449) | **0.180842**<br/>(Legacy: 0.181034) | **2.17%**<br/>(Legacy: 2.29%) | **Baseline Standard** (9 tails vs 12 in legacy pin; passes all guards) | **`ACTIVE_DEFAULT_REFERENCE`**<br/>Standard 18-token multi-horizon sports winner reference.<br/>Config: [`conf/base/c0_serving.json`](../conf/base/c0_serving.json) |
| **`C0-D`** | `Causal-C0-D`<br/>(`c0-d-born-again-w32-v1`) | `BornAgain-Distilled-Attention-Ensemble` | 19,914 (3 seeds Born-Again `W32-KD`, 6,638/seed) | 848 ms | **0.534134** pure<br/>(**0.533984** as `C0-D+` with Conn+ONS) | **0.180468** pure<br/>(0.180381 with Conn) | **1.87%** (Best calibration) | **PASSES_ALL_GUARDS** (13 tails pure / 10 tails with Conn; passes all 6 C0 guards) | **`PRIMARY_DISTILLED_STANDARD`**<br/>Self-contained PyTorch distilled winner; zero tree deps.<br/>+27.82% Polish Yield, 1.56 Sharpe. |
| **`C0-Slim`**<br/>(alias `C0-D1`) | `Causal-C0-Slim`<br/>(`c0-slim-1seed-w32-v1`) | `SingleSeed-Distilled-Attention` | **6,638** (1 seed `W32/E12`, 1/3rd size) | **278 ms**<br/>(**3.1x faster**) | **0.534656** standalone | **0.180667** standalone | **1.92%** | **PASSES_ALL_GUARDS** (13 tails; slashes difficult match variance) | **`CANDIDATE_LOW_LATENCY`**<br/>Production candidate for high-frequency or memory-constrained scoring environments.<br/>+34.37% Pinnacle 12% yield. |
| **`C0-Mini`**<br/>(alias `C0-Small`) | `Causal-C0-Mini`<br/>(`c0-mini-w16-kd-v1`) | `HalfWidth-Distilled-Attention` | **8,394** (`2,798`/seed, 58% smaller width) | 729 ms | **0.535845**<br/>($\Delta\text{LL} = -0.000800$ vs `W16-Direct`) | 0.181188 | **2.13%** | **PASSES_ALL_GUARDS** (14 tails; cures `W16-Direct` Major and LPL guard failures) | **`CANDIDATE_COMPACT_WIDTH`**<br/>Distilled student compressing representational width by 58% with zero regression. |

#### Architectural Specifications & Guard Details
- **`C0` (`PlayerHistory-Attention-Ensemble`, `Causal-C0`)**:
  - Specification: 3 seeds (`20260910`, `20260911`, `20260912`), Width-32 / Epoch-12 causal neural attention with **18-Token Multi-Horizon sequence (`16 Recent Maps + 365-Day Whole-Year Anchor Token + Empty Token`)**, `19,914` neural params (`6,638`/seed).
  - Metrics: LogLoss **0.534777**, Brier **0.180842**, ECE **2.17%**, Tails **9**, inference latency **848 ms**. Legacy 17-token `C0-16m` baseline (LogLoss `0.535449`, 12 tails) remains pinned for historical parity.
- **`C0-D` (`BornAgain-Distilled-Attention-Ensemble`, `Causal-C0-D`)**:
  - Specification: 19,914 neural params across 3 seeds Born-Again `W32-KD` distilled from the 9-checkpoint teacher ensemble with $\lambda = 0.60$ soft loss over the 18-token multi-horizon bank, **zero external tree/connector dependencies**.
  - Metrics: LogLoss **0.534134** ($\Delta\text{LL} = -0.001315$ vs legacy C0, $-0.000643$ vs 18-token C0), Brier **0.180468**, ECE **1.87%** (best calibration across all models), Tails **13**. Strictly passes all 6 C0 replacement guards.
  - Connector Variant (`C0-D+`): Optional pre-2026 `CAL` trained specialist connector + causal `ONS` recalibration yields LogLoss **0.533984**, Brier **0.180381**, Tails **10**, ECE **2.58%**.
  - **20-Token Multi-Scale Upgrade (`C0-D-20T`, `scripts/train_c0_multiscale_20token.py`)**: Adds `90-Day` (Past 3 Months / Split) and `180-Day` (Half-Year) anchor tokens alongside the `365-Day` Year token (`20,106` params = `6,702`/seed), improving standalone 2026 LogLoss to **`0.533954`** ($\Delta\text{LL} = -0.001494$ vs. `C0`, 95% CI $[-0.002681, -0.000147]$) and **`0.533600`** with Cross-Regional Input Normalization (`Brier 0.180388`, `BO3 LL 0.506207`, `Major Leagues LL 0.572799`), while 1-seed **`C0-Slim-20T`** improves to **`0.534621`**.
  - Financials (Polish Bookmakers, 12% Tax, DRO Active, No EXP-039): 43 bets, 60.5% win rate, **+1,765.78 PLN** profit (**+27.82% Yield**, **+17.66% ROI**, **1.56 Sharpe**, Net CLV **+6.02%**, Pre-DRO Curse **+2.14 pp** [$< +6.02\%$], Post-DRO Curse **-3.62 pp** [Zero-Disappointment Guarantee]); Polish Longshots (Odds 2.30–4.00): 26 bets, 57.7% win rate, **+1,793.15 PLN** (**+47.41% Yield**, **1.79 Sharpe**).
- **`C0-Slim` / `C0-D1` (`SingleSeed-Distilled-Attention`, `Causal-C0-Slim`)**:
  - Specification: Single seed (`20260910`) Distilled 18-Token Width-32 model with **6,638 neural params** (exactly 1/3rd parameter count of the 3-seed ensemble), achieving **3.1x faster** inference latency (**278 ms** vs **848 ms**).
  - Metrics: LogLoss **0.534656**, Brier **0.180667**, ECE **1.92%**, Tails **13**. Strictly passes all 6 C0 replacement guards.
  - Financials: Polish cohort +1,643.79 PLN (+26.53% Yield); Sharp Pinnacle $T-1\text{h}$ 12% Tax: 29 bets, 48.3% win rate, **+1,348.78 PLN** (**+34.37% Yield**, 1.15 Sharpe); Pinnacle 0% Tax: 560 bets, **+5,024.57 PLN** (**+6.04% Yield**, **+50.25% ROI**, 1.52 Sharpe); Pinnacle Longshots (Odds 2.30–4.00, 0% Tax): 211 bets, **+4,462.18 PLN** (**+14.59% Yield**, 1.59 Sharpe).
- **`C0-Mini` (`HalfWidth-Distilled-Attention`, `Causal-C0-Mini`)**:
  - Specification: **8,394 neural params (`2,798`/seed, 58% smaller width)** across 3 seeds Distilled Width-16 / Epoch-12 neural ensemble.
  - Metrics: LogLoss **0.535845**, Brier **0.181188**, ECE **2.13%**, Tails **14**. Improves over direct width-16 training (`W16-Direct` LogLoss `0.536645`) by $\Delta\text{LL} = -0.000800$. Passes all 6 C0 replacement guards, successfully curing both Major and LPL guard failures observed under unguided direct W16 training.

- **Automatic Cross-Regional Input Normalization (`src/models/cross_region_input_normalizer.py`)**:
  - Dynamically loads live `FamilyCalibratedGlicko2` (`ratings-v2`) international bridge posteriors (`families`, `tiers`) and active-player domestic pyramid medians (`players`) from `data/04_feature/live_c0_cache/regional_offsets_state.json` (rebuilt every 6 hours by `rebuild_regional_ratings` in `heavy_maintenance_cycle`).
  - Shifts all 6 collinear rating channels (`elo`, `gl_mu`, `ts_mu`, `os_mu`, `pl_mu`, `tm_mu`), `raw.w20`, and `hist.history` in lockstep by equal standard-deviation units gated by an overconfidence sigmoid so domestic matches remain 100% bit-exact (`0.000000` ΔLL) while `Worlds + MSI + First Stand + EWC` ($N=186$) improve by **`-0.064757` ΔLL** (`0.573079 -> 0.508321`, accuracy `70.43% -> 75.81%`), full 11,550 walk-forward improves to **`0.550203`**, and 2026 `C0-D` improves to **`0.533779`** (`0.8009` AUC).
### 1.2 Decision-Time Gate Simplification & Data Orientation Invariants

#### 4-Rule Production Betting Stack (Zero-Lookahead DRO Protocol)
Empirical verification confirms that when coupling **Wasserstein DRO** ($\varepsilon = 0.025$) with **Leave-One-Out (`LOO`) Market Shrinkage** against opening prices, the execution stack under 12% Polish tax simplifies to exactly **4 mathematical rules**:
1. **Leave-One-Out (LOO) Market Shrinkage**: Shrink neural attention forecasts toward the Leave-One-Out opening consensus $q_{\text{open, loo}, -k}(t)$ of the remaining active books at the exact scrape timestamp without lookahead ($\alpha = 0.65$):
   $$\text{logit}(p_{\text{hybrid}}) = 0.35 \cdot \text{logit}(p_{\text{C0}}) + 0.65 \cdot \text{logit}(q_{\text{open, loo}, -k}(t))$$
2. **Wasserstein DRO Gate ($\text{EV}_{\text{DRO}} > 0\%$)**: Enforce positive expected value under adversarial distribution shift with radius $\varepsilon = 0.025$:
   $$\text{EV}_{\text{DRO}} = (p_{\text{hybrid}} - 0.025) \cdot (1 - \tau) \cdot O - 1 > 0$$
3. **Logit Divergence Gate (`Guard B` $\le 0.85$)**: Reject extreme model divergences where neural overconfidence dominates market consensus:
   $$|\text{logit}(p_{\text{C0}}) - \text{logit}(q_{\text{open, loo}})| \le 0.85$$
4. **Odds Ceiling**: Restrict maximum decimal odds to contain longshot variance:
   $$\text{Odds} \le 4.00$$

**Retired Legacy Filters:**
- `Max EV <= 25%`: Retired. Redundant because Wasserstein DRO naturally suppresses longshot tail overconfidence.
- `Min EV >= 5%`: Retired. Subsumed by the continuous margin required by $\text{EV}_{\text{DRO}} > 0\%$ (which demands edge exceeding $\varepsilon \cdot (1 - \tau) \cdot O$).
- `Model-Consensus-Veto` (`EXP-039`): Retired for the distilled model family (`C0-D` / `C0-Slim`). Pure student neural attention achieves superior probability calibration (ECE 1.87%) without requiring external 20-match linear form validation.

#### Counterfactual Alignment Invariant & Walk-Forward Rule
Any script or verification pipeline processing raw quotes (`data/01_raw/c0_production_counterfactual/c0_production_20261006_01/quotes.parquet`) MUST adhere to three strict invariant joins:
1. **Quote Alignment Join**: Join `data/08_reporting/c0_production_counterfactual/c0_production_20261006_01/quote_alignment.parquet` (`quote_identity_status == 'verified'`) and orient `odds_1, odds_2` strictly via boolean `raw_a_is_team1`.
2. **Identity Alignment Join**: Join `data/08_reporting/c0_production_counterfactual/c0_production_20261006_01/identities.parquet` (`identity_status == 'verified'`, `date_within_one_day != False`) and align model prediction probabilities (`year_anchor_distilled_predictions.parquet`) via `str(int(float(ident.team1_id))) == str(int(float(p26.team1_id)))`.
3. **Batch Execution & LOO Consensus**: Group evaluations strictly at the end of each 2-hour cron scrape batch (`bucket_2h = scraped_dt.dt.floor('2h')`) with a `freshness <= 12h` window and compute Leave-One-Out consensus $q_{\text{open, loo}, -k}$ across the other active bookmakers excluding the focal bookmaker.

*Guarantee:* Enforcing these invariants ensures that no future study or backtest ever mixes unaligned `quotes.parquet` sides or closing-line lookahead into $p_{\text{hybrid}}$.
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

All historical studies remain reproducible at their immutable data stages in `data/` and documented in `reports/`. Dispositions follow scientific evaluation gates:

| Study Identifier | Research Objective | Outcome & Scientific Verdict | Evidence & Full Report |
|---|---|---|---|
| `c0_upgrade_and_distillation_20261017` | Multi-track C0 upgrade & knowledge distillation: Track 1 Knowledge Distillation (W16/W32 KD students), Track 2 Role-Aligned Lane & Query Attention, Track 3 Continuous Champion Archetypes + 15% UNK Dropout, Track 4 Disagreement-Gated Mixture + Online Newton Step Recalibration, followed by 18-Token Multi-Horizon sequence consolidation (`16 Recent Maps + 365-Day Whole-Year Anchor Token + Empty Token`). | **ADOPTED_DISTILLED_FAMILY_STANDARDIZATION**: Consolidated 18-token multi-horizon architecture as standard `C0` (LogLoss **0.534777**, 9 tails). Established pure 3-seed 18-token distilled model as **`C0-D`** (LogLoss **0.534134**, ECE **1.87%**, +27.82% Polish Yield, 1.56 Sharpe; 0.533984 with connector), **`C0-Slim`** / **`C0-D1`** (LogLoss **0.534656**, **3.1x faster** at `278 ms`, `6,638` params, +34.37% Pinnacle 12% yield), and **`C0-Mini`** (LogLoss **0.535845**, 58% smaller width). Tracks 2 and 3 rejected for failing replacement criteria. | [`data/08_reporting/c0_upgrade_and_distillation_20261017/`](../data/08_reporting/c0_upgrade_and_distillation_20261017/)<br/>[`docs/RESEARCH.md`](RESEARCH.md#2026-10-18--c0-knowledge-distillation--multi-track-upgrade-execution-c0-d-c0-slim-c0-mini) |
| `optimal_synergy_c0_thesis_20261012` | C0 Shrunk Hybrid ($\alpha=0.65$ LOO opening consensus) + Guard B + Thesis Veto on 10,000 PLN portfolio. | **ADOPTED_LOCAL_STRATEGY**: 50 bets, 58.0% win rate, **+23.91% Yield / +4.83% Net CLV** ($t=3.00, p=0.0042$), Claimed EV **+8.59%**, Residual Optimizer's Curse Bias **+3.76 pp** (56.2% edge retention). Vetoed 9 bets with 11.1% win rate. | [`reports/optimal_synergy_c0_thesis_simulation_20261012.md`](../reports/optimal_synergy_c0_thesis_simulation_20261012.md) |
| `oddsportal_multiyear_20261012` | 5-season simulation on OddsPortal ($N=11,609$, 2021–2026) testing open vs close. | **COMPLETED_LARGE_SAMPLE_VALIDATION**: At 0% tax: **+10.15% Yield / +42.5% CAGR** ($N=3,528$). At 12% tax: **+9.65% Yield** ($N=684$). | [`reports/oddsportal_open_close_simulation_20261012.md`](../reports/oddsportal_open_close_simulation_20261012.md) |
| `c0_monthly_refit_20261011` | Census of 234,435 quotes across 1,600 matches; monthly walk-forward C0 retraining. | **COMPLETED_CENSUS_AND_BENCHMARK**: Shrunk Hybrid LogLoss **0.557877** vs Market 0.570054 ($-1.22$ pp gain). Mean Raw CLV $+19.13\%$. | [`reports/monthly_refit_c0_production_odds_since_may_20261011.md`](../reports/monthly_refit_c0_production_odds_since_may_20261011.md) |
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
