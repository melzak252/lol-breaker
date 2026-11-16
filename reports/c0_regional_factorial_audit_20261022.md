# C0 × regional × fresh20 factorial audit — 2026-10-22

**Decision: COMPLETED / NOT ADOPTED — guard requirements failed; a financial edge is not established.** The completed study supports neither promotion nor a claim that betting is globally impossible. This is a fixed historical research comparison, not a live/official forecast or authenticated execution study.

**Scope:** Four frozen configurations (C0, C0+regional, fresh20, fresh20+regional); no new model or calibration fits, source requests, fresh production-database query, deployment, source activation or betting. Historical comparisons use 2026 data; financial results use the retained production export and fixed protocol. Long-term positive gross study results remain unchanged, but are not proof of profitable deployment.

## Decision and headline findings

The accepted neural/historical run is `c0_region_factorial_20261022_01_integrity_v4`; finance and corrected projection use v7. Fresh20+regional has the lowest observed common-cohort LL **.5330227337** and Brier **.1800299882**. Its five-contrast-adjusted 99% comparison with C0 favors the combination under match-weighted and equal-month summaries. Isolated regional and fresh20 contrasts cross zero; BO5/LPL replacement guards fail. Regionalization changes some TRAIN selections, but not the funded ASSESS tickets or outcomes in this financial sample. **NOT_ADOPTED_GUARDS_FAILED**: neither a reliable financial edge nor global impossibility follows.

## Frozen factorial and model identities

The four cells are a 2×2 comparison: existing serving-lineage C0 versus the actual saved fresh20 model, each evaluated without or with the separately reconstructed regional factor. Fresh20 is the actual saved **20-token model with 19 input positions**, not a model architecture inferred from a directory label: `data/06_models/c0_distilled_verification/c0d_repro_20261021_01/fresh/20t/`, with three seed checkpoints (`20260910`, `20260911`, `20260912`) and its saved bundle/calibration. The bundle manifest binds checkpoint hashes and calibration. The regional variants apply the same frozen regional law to their model inputs.

| Cell | Model identity | Regional component |
|---|---|---|
| C0 | Existing C0 native saved artifact and native inference | Off |
| C0+regional | Same C0 artifact and inference | On, causal reconstructed state |
| fresh20 | Actual saved fresh20 20-token model; 19 input positions | Off |
| fresh20+regional | Same fresh20 artifact and inference | On, causal reconstructed state |

The execution binding and protocol are the identity authority; model roots and hashes are in `execution_binding.json` and `bundle_manifest.json`. No slope surrogate or C0D alias was substituted. C0 serving model/configuration, manifest, registry and original weights were not changed.

## Completed Worlds exercise

The v4 Worlds stage completed in **91.86 seconds**, with all **32 requested predictions** (eight matches × four configurations). Its query, prediction and proof files are byte-identical to successful v3 Worlds outputs; query SHA-256 is `3c4fbc353b8dd10b11555b8835369c04164f7ddf1ae826f9d723c815cc2ccd02`. The independent v3 anchor proof reports **700,833 checks, zero failures, 18,483 membership occurrences and 240 numeric comparisons, maximum error 0**, binding 700,290 cached player-map records. The v4 artifact-equivalence receipt reuses this proof; it is not another independent experiment.

The table is preserved as requested. Each probability is the first named team's win probability, not blue-side probability; the request is eight best-of-one pairings, IDs 146609–146616.

| First team vs opponent | C0 | C0+regional | fresh20 | fresh20+regional |
|---|---:|---:|---:|---:|
| Gen.G vs Invictus Gaming | 73.43% | 74.33% | 70.02% | 71.06% |
| Team Liquid vs CTBC Flying Oyster | 63.04% | 68.58% | 63.54% | 69.35% |
| Anyone's Legend vs Karmine Corp | 38.16% | 55.75% | 40.53% | 58.90% |
| G2 Esports vs Dplus KIA | 57.33% | 31.29% | 58.73% | 31.38% |
| Hanwha Life Esports vs Top Esports | 65.33% | 66.71% | 64.12% | 65.57% |
| Los Grandes vs T1 | 26.57% | 15.55% | 26.78% | 15.31% |
| Team Secret Whales vs LYON | 60.24% | 44.42% | 62.43% | 45.99% |
| Bilibili Gaming vs Movistar KOI | 64.53% | 72.33% | 65.65% | 73.65% |

The frozen request cutoff is 2026-10-20; newest used history day is 2026-10-18. Match starts on these cached snapshots are placeholders (including the 2026-10-21 request start), not official schedules or issuance timestamps. Snapshot player IDs are a saved roster scenario; authentic roster announcements/clocks were not verified. These are conditional input scenarios, not live or bettable forecasts. **No Worlds outcome accuracy claim is made.**

## Methods and data construction

Raw anchors use native 33-wide raw-map vectors and the TRAIN-fitted `HistoryNormalizer`. A map enters a player's horizon only when its whole series has effectively released strictly before the cutoff and `0 < cutoff_day − played_day < horizon_days`, for 365/180/90 days. Each eligible player-map record counts once; this is not one observation or equal weight per series. Summaries use full retained history, not reconstruction from the latest 16 maps. Corrected canonical releases and latest-map-day release scenarios remain explicit.

Fresh20 uses the actual saved public-inference route: three masked horizon-summary positions followed by 16 recent-map positions; the model adds its internal empty token, giving 20 tokens. Masks distinguish real support from empty positions. This architecture is defined in `scripts/train_c0_multiscale_20token.py`, not inferred from the artifact name. No refit or recalibration was performed.

Regional inputs use `FamilyCalibratedGlicko2` with the existing defaults and complete-series scores, first-natural-map rosters, and strictly prior release-day updates. Later substitutes are not unioned into an invented roster. Current cached offsets are diagnostic only, never a retrospective fallback; unsupported families remain explicit absences. The existing `normalize_cross_region_batch` **input-feature transform** (`z_per_100gl=.28`) precedes inference through the unchanged saved model; it is not a post-hoc probability adjustment or odds blend. Active rows retain the existing float16 quantization; zero-difference rows remain unchanged. The heuristic coefficients were development-exposed: causal timing removes event-time look-ahead, not recipe-selection optimism.

The Worlds binding freezes source/model/config/code bytes, protocol, parameters, cutoff, roster scenarios and environment. The v4 raw-history adapter uses 42,177 reconstructed series, 70,029 source maps and 700,290 player-map records, sourced from full raw history plus corrected release map; regional replay source is `raw_history_causal_family_calibrated_replay`. Cached Worlds records are separately hash-bound and not canonical daily inputs.

## Historical results: 3,103 standalone coverage vs 3,083 paired support

The denominator is **3,103 canonical 2026 target rows**. All years were exposed; this is not an independent holdout, and only 2026 saved weights are origin-eligible (no fresh20 inference claim for 2021–2025). Native C0 and fresh20 have all 3,103 predictions. Regional variants each have 3,083; **20 explicit regional absences**. Exact common support is 3,083 rows for all four cells. The common-support table below is paired and comparable; do not substitute it for standalone coverage.

| Configuration / scope | n | Log loss | Brier | ECE | Accuracy | AUC |
|---|---:|---:|---:|---:|---:|---:|
| C0, common support | 3,083 | 0.5369474649936095 | 0.18157644260330946 | 2.32673565% | 0.71942913 | 0.79803737 |
| C0+regional, common support | 3,083 | 0.5348164317926817 | 0.18077686190731554 | 2.25631754% | 0.72169964 | 0.79993721 |
| fresh20, common support | 3,083 | 0.5353911533229386 | 0.18090585859596142 | 2.12261243% | 0.72137528 | 0.79958210 |
| fresh20+regional, common support | 3,083 | 0.5330227336604842 | 0.1800299882441574 | 1.92875198% | 0.72397016 | 0.80165250 |
| C0, all native standalone rows | 3,103 | 0.5354487005519922 | — | — | — | — |
| fresh20, all native standalone rows | 3,103 | 0.5339251721044982 | — | — | — | — |

Full-coverage standalone values are shown only for native C0 and fresh20; no standalone regional-cell metric over 3,103 is imputed for its 20 absent rows. The independently reconstructed full-coverage verifier reports the same C0 log loss and separately provides full scores; its stated limit is that full per-configuration metrics were reconstructed rather than compared with absent producer-summary objects.

Uncertainty uses 5,000 paired UTC-calendar-month bootstrap draws, seed 20261022 and nine months. Delta is candidate minus reference; negative is favorable. **Fresh20 − C0 uses all 3,103 native-covered fixtures; the other four contrasts use 3,083 pairwise-supported fixtures.** Every pair uses identical IDs and orientations. The five predeclared log-loss contrasts have adjusted 99% intervals; equal-month estimates expose weighting sensitivity. These intervals do not adjust for all prior adaptive historical research.

| Contrast | ΔLL weighted | Weighted 99% interval | ΔLL equal-month | Equal-month 99% interval |
|---|---:|---:|---:|---:|
| C0+regional − C0 | −0.0021310332 | [−0.0079977949, 0.0001714272] | −0.0040015982 | [−0.0122204740, 0.0000426390] |
| fresh20+regional − fresh20 | −0.0023684197 | [−0.0087180684, 0.00013531699] | −0.0044034829 | [−0.0132738837, 0.0000102211] |
| fresh20 − C0 | −0.0015235284 | [−0.0032152529, 0.00028465883] | −0.0020428582 | [−0.0065356983, 0.0012053928] |
| fresh20+regional − C0+regional | −0.0017936981 | [−0.0034320269, −0.0002369253] | −0.0024690512 | [−0.0067236411, 0.0002889171] |
| fresh20+regional − C0 | −0.0039247313 | [−0.0090369653, −0.0015190076] | −0.0064706493 | [−0.0133571424, −0.0018530494] |

The combined fresh20+regional vs C0 weighted and equal-month intervals are below zero here; fresh20+regional vs C0+regional has a weighted interval below zero but an equal-month interval crossing zero. Isolated regional-only and fresh20-only 99% intervals cross zero under both reported estimates. This evidence is conditional on the exposed 2026 cohort and previously exposed regional heuristic; it does not establish independent generalization or causal recipe benefit.

The six actual guard profiles are **BO1, BO3, BO5, LPL, Major and Unseen_Players**. They compare two candidates against C0 and annual A0: 24 contrasts, confidence `1 − .05/24`, margin .005, and the predeclared equal-month upper bound. The source-unseen profile in `src/analysis/lineup_law_reliability.py:594-597,842-860` is the same pinned `unknown_players > 0` membership, not a seventh independent profile: 334 full / 314 regional-supported rows, with no unknown metadata. Fresh20+regional fails BO5 upper bounds **.0100737177764 vs C0 / .009103684381 vs annual A0** and LPL **.007017027796 / .01978699424**, all above .005. All 24 guard calculations were independently replayed; maximum numeric discrepancy **1.38778e−17**.

## Financial cohort, fixed timings, and model cells

Finance continuation v7 ran 32 cases in 95.35 seconds, bound to the exact v4 daily parquet and historical protocol/stage hashes, and reused NN outputs (no duplicate NN run). It uses retained production-export data, not a fresh current-DB query. Full 1,771-fixture accounting is TRAIN 794 / ASSESS 846 / boundary-excluded 131. ASSESS is the outcome summary below; TRAIN rows remain included in machine evidence. A separate first-ready projection cohort has 302 fixtures (83/194/25); it is not to be mixed with the 1,771 denominator.

Selection qualification and wallet simulation are distinct: the same frozen source q/prices and gates, each model's raw series probability, earliest eligible causal state in a 48-hour window (never future maximum), quote freshness ≤300 seconds and ≥2 other books. Frozen settings are sports alpha .5, temperature 1, min net EV .05, Wasserstein epsilon .025, 12% full-winning-payout tax and bound `BASE_POLICY`. Each independent wallet starts at 10,000 PLN, quarter Kelly, 2–150 PLN stakes, 25% exposure cap and one bet per fixture. Regional-pair ASSESS outcomes are identical; TRAIN outcomes differ.

The compact table reports all four timings for every model family. `bets (net profit; ROI on stake turnover)` includes all accepted bets in the ASSESS partition. Values for regional pairs are identical to their corresponding nonregional cell; this does not make the pair independent evidence. No timing/model winner is selected.

| Model | First-ready | Quote-triggered first eligible | UTC even-hour 2h first eligible | UTC odd-hour 2h first eligible |
|---|---|---|---|---|
| C0 | 3 (+343.495715 PLN; 98.4192%) | 15 (+255.561741; 13.1689%) | 2 (−254.731191; −100%) | 12 (+268.727968; 17.2501%) |
| C0+regional | 3 (+343.495715 PLN; 98.4192%) | 15 (+255.561741; 13.1689%) | 2 (−254.731191; −100%) | 12 (+268.727968; 17.2501%) |
| fresh20 | 2 (+173.966516 PLN; 63.1949%) | 8 (−304.102137; −28.3714%) | 2 (−269.772942; −100%) | 6 (−64.622873; −7.7670%) |
| fresh20+regional | 2 (+173.966516 PLN; 63.1949%) | 8 (−304.102137; −28.3714%) | 2 (−269.772942; −100%) | 6 (−64.622873; −7.7670%) |

All ASSESS selections in these rows settled; pending count is zero. This does not negate unknown source authentication/fills or external execution uncertainty. The 32 wallet cases share fixture opportunities across models/timings, so they are not 32 independent samples. Baseline parity passed for all eight C0 timing/partition cases, including ticket identities and 1e−8 PLN money checks. Financial outcomes are simulated portfolio-policy results, not pure fixed-stake P&L: separate fixed-ticket resized-stake effects are available in the saved artifact and must not be conflated with wallet policy.

TRAIN diagnostics below are **bets / net P&L (PLN)**. They are not additional ASSESS evidence and were not used to select a new policy. Regionalization removes some TRAIN tickets; therefore ASSESS equivalence must not be generalized to all 32 cases.

| Model | First-ready | Quote-triggered | UTC-even 2h | UTC-odd 2h |
|---|---:|---:|---:|---:|
| C0 | 2 / +31.02 | 4 / +236.71 | 0 / +0.00 | 3 / +348.75 |
| C0+regional | 2 / +31.02 | 3 / +348.75 | 0 / +0.00 | 3 / +348.75 |
| fresh20 | 3 / +324.30 | 5 / +504.11 | 0 / +0.00 | 4 / +209.50 |
| fresh20+regional | 3 / +324.30 | 4 / +610.91 | 0 / +0.00 | 3 / +313.31 |


The 1,771-fixture universe remains accounted across TRAIN 794 / ASSESS 846 / boundary-excluded 131. `unavailable_daily_probability_rows = 0` refers only to the 1,126 usable same-day model-state rows out of 4,983 requested native daily states—not complete coverage of all 4,983 states or all fixtures. In particular, this value does not establish coverage of the full 1,771-fixture native universe.

The independent wallet audit counts **86 repeated ledger entries** in the 5–10% EV band across models, timings and partitions; these are not 86 independent bets. For **C0 quote-triggered ASSESS**, the 12 near-band selections have mean EV overstatement **19.8405 pp**, stake-weighted overstatement **21.3068 pp**, and realized mean unit return **−12.3667%**. The overall 15-bet stake ROI is **13.1689%**, with nominal within-case 95% interval **[2.0254%, 15.9165%]**. Near-band overstatement interval **[11.9622, 106.5017] pp** does not fit inside the required **[−2,+2] pp** reliability band. These are only two ASSESS months. The key `search_adjusted_roi_ci` uses `search_family_size=1`: it is not adjusted across 16 ASSESS cases or prior adaptive research.

Financial-bootstrap draws with undefined conditional ROI/EV-error remain undefined, not replaced or dropped. Selection overlaps and gate attribution are machine-recorded in `financial_selection_overlaps.parquet` and `financial_event_attribution.parquet`; repeated cell/timing cases are shared-opportunity comparisons, not independent bets. No current-DB source availability, authenticated CLV/fill, or true execution uncertainty is established.

## Execution history, recovery and integrity limits

V2 stopped for a same-ID display-punctuation mismatch and incomplete raw-series participant headers. V3 was rejected because its fresh20 inference assignment was missing (3,103 NaNs) and it selected the Worlds cache for canonical daily history. These were defects in the new audit runner, not changes to the existing native model or weights. Corrected v4 completed genuine inference: C0 reference maximum error **0**, fresh20 maximum **5.206107323019182e−9** over all 3,103 rows. All **1,126** daily C0 and fresh20 probabilities exactly reproduce the earlier genuine saved states.

Financial recovery continued through v5/v6 and accepted v7; those failed attempts are retained execution history, not scientific results. V7’s corrected first-ready projection admits 910/910 rows with strict source-day, forecast and query bounds. Exact-clock match count is zero because all legitimate states occur earlier on the same day. The old v4 first-ready file with 910 absent rows is retired and must not be treated as accepted output.

Independent v7 numerical verification: **73 checks, zero failures**, includes 32 Decimal wallet checks and metric/CI/source-identity comparisons. It independently reconstructed common-support metrics within at most 1.11e−16; limitations: it does not verify neural checkpoint internals, independently rebuild source opportunity/quote gates, or authenticate external issuance/fills. The separate guard/daily verifier passed 24 guard numeric comparisons (maximum error 1.38778e−17), checks all six actual profiles, and confirms daily parity. Protected-code check reports **191/191** unchanged. Final source regression reports **204 passed, one existing warning**; scope is the final source regression run, not proof of every production behavior. The v3 anchor proof's 700,833 checks and 240 numeric anchor comparisons are specific to anchors/membership, not forecast correctness. No verifier establishes live binding or new source authenticity.

No neural/calibration fits, source requests, database queries, deployment, promotion, source activation or future Worlds outcomes occurred. All specified protocol settings were fixed; alpha, DRO and thresholds were not tuned to current profit. Positive gross long-term study results remain separate and unchanged. This audit neither establishes a current financial edge nor proves betting impossible in general.

## Evidence paths

- v4 frozen protocol/binding/stage receipt: `data/05_model_input/c0_region_factorial/c0_region_factorial_20261022_01_integrity_v4/{protocol.json,execution_binding.json,stage_cache/worlds.json}`.
- v4 outputs: `data/07_model_output/c0_region_factorial/c0_region_factorial_20261022_01_integrity_v4/{canonical_predictions.parquet,daily_model_predictions.parquet,worlds_predictions.parquet,worlds_query.pkl,worlds_proofs.json}`.
- v4 completed scores/contrasts: `data/08_reporting/c0_region_factorial/c0_region_factorial_20261022_01_integrity_v4/{historical_scores.json,historical_contrasts.json,historical_slices.json}`.
- v7 frozen protocol/binding and outputs: `data/05_model_input/c0_region_factorial/c0_region_factorial_20261022_01_integrity_v7/{protocol.json,execution_binding.json,stage_cache/}` and `data/08_reporting/c0_region_factorial/c0_region_factorial_20261022_01_integrity_v7/{financial_summaries.parquet,financial_wallets.json,financial_receipt.json,financial_baseline_parity.json,financial_selection_overlaps.parquet,financial_fixed_tickets_resized_stakes_effects.parquet,financial_paired_monthly_contrasts.parquet}`.
- Corrected accepted projection only: `data/07_model_output/c0_region_factorial/c0_region_factorial_20261022_01_integrity_v7/{first_ready_predictions.parquet,first_ready_proofs.json}`.
- Independent v7 proofs: `data/08_reporting/c0_region_factorial/c0_region_factorial_20261022_01_integrity_v7_verification/verification/{numerical_verification_final.json,guard_and_daily_verification.json,protected_after.json}`.
- v3 anchor proof and v4-v3 equivalence binding: `data/08_reporting/c0_region_factorial/c0_region_factorial_20261022_01_integrity_v3_verification/verification/anchor_verification.json`; `data/08_reporting/c0_region_factorial/c0_region_factorial_20261022_01_integrity_v4_verification/verification/worlds_proof_binding.json`.
- Earlier recovery evidence: v5 failed finance attempt in `data/08_reporting/c0_region_factorial/c0_region_factorial_20261022_01_integrity_v5_verification/verification/`; v6 historical verification and failed v6 baseline source in `data/08_reporting/c0_region_factorial/c0_region_factorial_20261022_01_integrity_v6_verification/verification/`.
- Saved fresh20 manifest: `data/06_models/c0_distilled_verification/c0d_repro_20261021_01/fresh/20t/bundle_manifest.json`.
- The retained production-export provenance is separate from current database state; no fresh production DB query or live-binding proof is claimed.
