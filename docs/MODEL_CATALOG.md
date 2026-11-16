# Model catalog

This catalog separates the locally configured match model from historical controls and research specialists. **Local configuration is not a deployment, a prospective validation or a financial certification.** For methods and evidence, start with the [project report](PROJECT_REPORT.md) and [research journal](RESEARCH.md).

## Main model and local policy

| Name | Identity and role | Scope |
|---|---|---|
| **C20R** | Documentation name for the three-seed fresh20 player-history ensemble with regional input adjustment. | Local runtime name `Causal-C0`; serving version `c0-native-fresh20-2026-10-21-v1-regional-v1`. Predicts the whole BO1/3/5 series without market odds. |
| **Hybrid-50** | Combines the C20R sports prediction with an admitted market consensus once, with equal weights in logit space. | Separate market-assisted identity `Hybrid-Operational-Market`; focal-book qualification requires at least two other distinct fresh bookmakers. No profitability claim. |
| **C0** | Historical native 17-token match-winner model. | Reference in the factorial study; not the current fresh20+regional default. |

C20R is a **display name**, not a new runtime selector or artifact ID. The [serving configuration](../conf/base/c0_serving.json) and [registry](../betting_app/core/models/registry.py) specify the local contract. Fresh20 uses 19 history input slots plus one model token; regional state is reconstructed from prior released history. The separate historical cached regional normalizer is not the same as that input path. See the [application guide](../betting_app/README.md) for input, roster and policy boundaries.

### Historical match-quality comparison

The [factorial audit](../reports/c0_regional_factorial_audit_20261022.md) compares four saved configurations without new fits. On 3,083 common supported 2026 series, C20R scored log loss **0.533023** and Brier **0.180030**, versus historical C0 **0.536947** and **0.181576**. Twenty of the 3,103 requested rows lack regional support. The combined paired comparison favors C20R on the study's reported intervals, but BO5 and LPL replacement guards failed. The decision was **NOT_ADOPTED_GUARDS_FAILED**; the exposed development cohort is not a prospective holdout. The later local configuration change does not alter that study decision.

## Reference models

| Name | Role and distinction | Source |
|---|---|---|
| **A0** | Annual walk-forward ensemble, the canonical historical research reference; not the local C20R artifact. | [Benchmark manifest](../conf/base/research_benchmark.json) |
| **EXP-039** | Frozen symmetric calibrated ElasticNet team-form baseline with W20 history and series projection. Missing a comparison is not agreement. | [Model catalog context and scripts](../scripts/README.md) |
| **Glicko-2 / Elo** | Player-rating controls for historical match comparisons. | [Benchmark manifest](../conf/base/research_benchmark.json) |
| **C0-D, C0-Slim, C0-Mini** | Historical distilled research variants, distinct from C20R and its saved fresh20 bundle. Original anchor-bank lineage and old financial claims are not certified. | [Code audit](../review-reports/codebase_logic_audit_20261021.md), [research journal](RESEARCH.md) |

## Specialists and research tools

Metrics below are retained historical study measurements, not prospective application performance. Local bundles under `data/` are not generally present in Git; consult the [reproduction guide](REPRODUCING.md) for prerequisites.

| Model / study | Scope and status | Sources |
|---|---|---|
| `prop_distribution_20261007_01` | Standalone map 1 kills: `KEEP_BASELINE`, NLL 3.590488; map 1 duration and BO3/5 series length: `KEEP_RESEARCH_CANDIDATE`, NLL 3.076194 and 0.714580 respectively. | [Model implementation](../src/models/prop_distribution.py); local recovery archive `data/06_models/model_preservation/prop_distribution_20261007_01.zip` |
| `prop_distribution_20261027_01` | Prop v2, `A3_combined_full` / `A1_enriched_features`: 414-feature and CAL-mixture research, `DEVELOPMENT_CANDIDATE_RETAINED`. Kills/duration/length NLL 3.579557 / 3.069120 / 0.710849; A1 length 0.710557. | `src/models/prop_distribution_v2.py`; local bundle `data/06_models/prop_distribution/prop_distribution_20261027_01/` |
| `draft_intelligence_20261027_01` | `Draft_Ensemble` and `CausalDraftState` pick/ban and post-draft winrate: `DEVELOPMENT_CANDIDATE_RETAINED`, post-draft scope. Map 1 LL 0.607013; post-Map-1-draft series LL 0.570846. | `scripts/predict_draft.py`; local bundle `data/06_models/draft_intelligence/draft_intelligence_20261027_01/` |
| `c0_predictive_power_20261103_01` | Six-track offline historical development: no general replacement; scoreline is a development candidate, event law fails CAL, stakes not evaluable. | `src/models/predictive_power_common.py`; `reports/c0_predictive_power_20261103.md` |
| `tournament_numerical_20261007_01` and related tools | Tournament and Pick'Em research; numerical methods and bracket scenarios do not establish forecast calibration or official draw equivalence. | [Tournament simulator](../src/models/advanced_tournament_simulator.py), [Crystal Ball simulator](../src/models/crystal_ball_simulator.py), [tournament research](../reports/reliable_tournament_simulation_research_20261006.md) |

## Study status and evidence

| Study | Result and boundary | Evidence |
|---|---|---|
| Fresh20 + regional factorial | Lowest point scores among the four cells on supported 2026 rows; formal replacement guards fail. | [Factorial audit](../reports/c0_regional_factorial_audit_20261022.md) |
| Fresh20 raw-source reconstruction | Separate frozen new-bank experiment; the fresh20 candidate failed annual-A0 guards and was not promoted by that study. | [Research journal](RESEARCH.md), [scripts index](../scripts/README.md) |
| Selected-bet reliability and policy | Historical simulations and policy replays are not authenticated live execution or evidence of guaranteed returns. | [Reliability audit](../reports/c0_selected_betting_reliability_research_and_audit_20261006.md), [edge-recovery audit](../reports/c0_edge_recovery_audit_20261022.md) |
| Tournament copula | Candidate failed its CAL guards; not adopted. | [Tournament research](../reports/reliable_tournament_simulation_research_20261006.md) |

The [documentation index](README.md) links architecture, data, reproduction and historical reports. Dated reports describe their own study and may contain later-corrected claims; use their scope and the research journal rather than treating older status as current.
