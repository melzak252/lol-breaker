# C0 edge recovery audit — 2026-10-22

**Run:** `c0_edge_recovery_20261022_01`  
**Scope:** historical development audit only. No source permission, deployment, policy activation, live database access, or new base-model training was performed.  
**Policy status:** `COMPLETED_DESCRIPTIVE_HISTORICAL_AUDIT`; all 350 frozen configurations (1,400 config-by-timing combinations) were evaluated. **No betting policy was selected**: all four timing families failed the frozen earlier-only training support requirement. Assessment results below are descriptive and none is recommended.  
**Decision:** no certified recoverable financial edge or production-ready profit strategy is established by these data. This is not a claim that edge is globally impossible.

## Executive findings

1. **The proper-score blend yields a bounded alpha candidate, not a profit policy.** Earlier-only strict TRAIN selected sports weight `alpha=0.2853137251` (therefore market weight about 71.47%), from 83 fixtures across three months. On the 194-fixture ASSESS block, the fitted blend scored slightly better than alpha .5 in log loss and Brier score, but every year in this work is development-exposed and the assessment spans only August–September. The alpha was not promoted and was not selected on financial outcomes.
2. **The policy-selection rule returned no policy.** It required at least 20 settled TRAIN bets and two active TRAIN months. The best supported bet counts across the four timing families were only 6, 15, 0, and 13. Therefore there is no selected TRAIN policy, no valid selected-policy contrast, and no ASSESS winner to recommend. A few assessment cells have positive realized returns; the frozen protocol explicitly makes those exploratory, with a 1,400-cell timing/configuration search and only two ASSESS months.
3. **The assessment outcomes depend sharply on clock, alpha, odds, and gates.** The unchanged alpha .5 baseline produced three first-ready bets (+343.50 PLN), 15 quote-triggered bets (+255.56), two even-hour 2h bets (−254.73), and 12 odd-hour 2h bets (+268.73). These are tiny, separate historical portfolios, not validated prospective returns. With pure C0 (alpha 1) the same baseline gate produced losses on all four clocks; market-only matched control (alpha 0) placed no ASSESS bet under the sports-dependent baseline controls.
4. **Relaxing or changing one gate can expose positive assessment points, but no such point is selection-supported.** The 140-cell core grid and deduplicated one-factor sensitivities were evaluated on the complete fixture denominator, without using ASSESS to select a policy. The best assessment point is hindsight by construction. Positive and negative outcomes are scattered across the grid; no finite-sample result supports a universal no-edge claim or a production recommendation.
5. **The retained historical 50-bet spreadsheet has a confirmed probability/EV overwrite.** An exact-hash-bound pre-overwrite copy preserves the earlier values. All 50 bets' `p_hybrid` and `model_ev_pct` were subsequently rewritten using approximate same-date/bookmaker/odds matches and a post-hoc net-CLV (future-close-informed) alpha scan, while all other columns, including outcomes, stakes, profits and bankroll fields, remained unchanged. The original ticket generator and exact match/side/quote/bookmaker identifiers remain unrecovered. The stored +1,715.41 PLN is therefore not independently attributable to exact tickets or certified edge; the overwrite changed claims, not the recorded profit ledger.
6. **Timing cadence plausibly explains why fixed two-hour replay diverges sharply, without proving causation.** Of 248,735 raw quotes, 202,646 (81.47%) arrived at even UTC hours, with a further 150,242 concentrated at minute :55 and 15,872 at :56. The odd-hour grid had 4,805 native decision keys / 44,966 event rows versus only 243 / 2,580 on the even-hour grid before financial gates. All 19 baseline quote-triggered accepted bets were verified actual snapshot states (`is_snapshot=true`, `is_grid=false`). This is a strong snapshot-cadence/TTL alignment explanation for loss of opportunity coverage at a fixed even-hour phase, not a causal P&L attribution or evidence of actual manual execution. The 2h schedule can therefore miss fresh quote updates systematically; do not infer that blind TTL loosening or post-hoc odd-hour deployment recovers edge.
7. **Real model comparisons do not establish a replacement.** Six distinct saved artifacts were scored on the same 194 ASSESS fixtures. The C0-D / legacy Core18 naming collision is explicitly separated from the genuinely different historical 17T “Core” student. At alpha .5, none of the five alternatives has a lower ASSESS log loss than C0; short-run financial results are mixed and do not imply general model ranking. No weights, calibration, registry, manifest, or serving policy were changed.
8. **A real old Shin utility defect was fixed and tested, but is unrelated to these financial results.** The previous `betting_guards_a_to_d.devig_shin` implementation was wrong for asymmetric two-outcome odds. The current frozen financial audit uses multiplicative q, not that helper, so the correction is not claimed to recover any historical profit.

## Scope, clocks, and denominators

The binding design is `data/05_model_input/c0_edge_recovery/c0_edge_recovery_20261022_01/protocol_v2.json` (design revision 2). It defines the fixed 2026-08-01 TRAIN/ASSESS cutoff, 2026-10-01 assessment end, first-ready cohort, alpha search, policy grid, timing rules, and portfolio mechanics. Every historical observation is treated as development-exposed; the later period is a held-out assessment block only within the frozen historical experiment, not a genuinely unseen future holdout.

- Full accounting universe: 1,771 fixture rows; 1,126 usable same-day native daily C0 states over 400 fixtures.
- First-ready, native-covered scoring cohort: 302 fixtures and 910 focal bookmaker rows; one selected fixture probability is counted once overall (focal rows receive `1/focal_count` weights). All 302 have known winners.
- Strict TRAIN: 83 fixtures, three active months, 249 focal rows. Boundary-excluded: 25 fixtures / 76 rows. ASSESS: 194 fixtures, two active months (171 August and 23 September), 585 rows. The other 1,469 fixtures have no first-ready state for matched scoring.
- Full cash denominator: 794 TRAIN fixtures, 846 ASSESS fixtures, 57 outside the two fixed windows, 74 with unknown start; 879/1,771 outcomes are unknown. Unavailable fixtures remain in the ledger as absences/abstentions rather than being dropped from the denominator.
- Timing methods: first-ready; first qualifying observed quote state (not future maximum EV); fixed UTC even-hour 2h; fixed UTC odd-hour 2h. Every decision is strictly before recorded start and within the frozen 48-hour window, at most one bet per canonical fixture. Quote eligibility requires an observed state, quote age at most 300 seconds, and at least two other books. No timing phase was chosen on ASSESS.

The fixed portfolio starts with 10,000 PLN, applies a 12% tax to winning payouts (`after-tax payout = stake × odds × 0.88`; net profit on a win is that after-tax payout minus the original stake), chronological capital feedback, quarter-Kelly sizing, a 150 PLN maximum stake, and a per-bet limit of 25% of available free capital; unresolved stakes are reserved. The wallet/ledger SSOT uses continuous decimal-PLN stake amounts rather than cent-rounded stakes. A source-internal comparison of 925 ledger rows and 210 wallets reported exact cash parity (zero error) and maximum stake delta 4.8e−13 PLN; that establishes arithmetic agreement with the SSOT, not cent-rounded ticket precision, manual fillability, market impact or source latency. Financial returns below are simulated from snapshots and outcomes, not authentic issued bets, fills, wallets, or current live opportunities.

## Alpha selection and predictive quality

The blend is `sigmoid(alpha*logit(p_C0) + (1-alpha)*logit(q_focal_LOO))`, temperature 1 and existing epsilon `1e-6`. Alpha is the **sports/C0 weight**: historical “65% market” corresponds to sports alpha .35, not .65. The series probability is used once, after BO conversion; there is no second shrink or BO projection.

### Earlier-only alpha fit

The equal-fixture-weighted series-logloss optimum on strict TRAIN is alpha `0.28531372507769104`; TRAIN log loss 0.5315484984450976. Support is 83 fixtures/three months, above the frozen 50-fixture/two-month minimum. A separately named release-clock-only diagnostic selected alpha `0.4280513766` using 91 fixtures/three months, log loss 0.5199719265; this uses a less conservative clock and does not replace the result-recorded-label primary fit. Neither result changes serving configuration.

Selected C0 blend predictive metrics (fixture-weighted; `alpha/score_curves.csv`):

| Partition / blend | Fixtures | Rows | Log loss | Brier |
|---|---:|---:|---:|---:|
| TRAIN, market-only alpha 0 | 83 | 249 | 0.5341043 | 0.1801828 |
| TRAIN, alpha .5 | 83 | 249 | 0.5329340 | 0.1806345 |
| TRAIN, fitted alpha .2853 | 83 | 249 | 0.5315485 | 0.1796542 |
| ASSESS, market-only alpha 0 | 194 | 585 | 0.5973682 | 0.2063971 |
| ASSESS, alpha .5 | 194 | 585 | 0.5974109 | 0.2077458 |
| ASSESS, fitted alpha .2853 | 194 | 585 | 0.5953613 | 0.2062420 |
| ASSESS, pure C0 alpha 1 | 194 | 585 | 0.6130263 | 0.2152210 |

The paired intervals are in `alpha/paired_monthly_intervals.json`: 5,000 paired full-month resamples, with undefined draws retained. On ASSESS, the strict TRAIN alpha vs alpha .5 contrast is −0.0050914 equal-month log loss (95% interval −0.0090785 to −0.0011043) and −0.0020497 match-weighted log loss (same interval −0.0090785 to −0.0011043); equal-month Brier is −0.0030662 (95% interval −0.0051142 to −0.0010182) and match-weighted Brier is −0.0015038 (same interval). There are only two ASSESS months and those dates were exposed during 2026 development. These intervals describe this limited retrospective cohort; they do not make the alpha prospective or certify money-making value. The `*_ci95_descriptive` columns in `score_curves.csv` are IID-fixture normal-SE descriptions, not monthly intervals.

The completed independent receipt `independent/independent.json` reproduces 21 alpha LL/Brier values to maximum absolute errors 3.33e−16 / 1.39e−16. Alpha root parameter difference is 1.32657e−8; derivative at reported alpha is 8.12248e−10, maximum probability difference 3.52351e−9, and objective regret 1.11022e−16. The corrected objective-regret tolerance passes at 1e−10; reported and root objectives are effectively identical. The earlier verifier failure and its byte-exact source remain archived. This is bounded numeric agreement for saved alpha scores and optimizer objective, not a prospective alpha claim or financial certification; the broader completed independent receipt and limits are summarized below.

## Real model identities and matched scores

The real model CLI regenerated predictions from the actual saved artifacts and raw causal inputs (no model training). `models/model_audit_receipt.json` records 1,126/1,126 native states, 910/910 focal rows and 302/302 fixtures, zero query absences, and exact C0 parity against the sealed reference (maximum difference 0). The raw history adapter used 41,915 series, 69,370 maps, 693,700 player-map vectors, and 7,056 players. The audit is research-only; source activation and deployment are false.

| Actual saved artifact | Identity / lineage note | ASSESS LL at alpha .5 | ASSESS LL at alpha 1 |
|---|---|---:|---:|
| C0 | Current C0 | 0.597410939 | 0.613026264 |
| Legacy C0-D / Core18 | One shared registry artifact, not two models; legacy positional lineage unverified | 0.597575949 | 0.613374092 |
| Historical Core17T | Separate real Track-1 `student_w32_kd` artifact, also called Core in an older study; legacy positional lineage unverified | 0.597779176 | 0.613990775 |
| Slim18 | Actual single-seed saved model with its own Beta calibration; legacy positional lineage unverified | 0.599706162 | 0.617857730 |
| Fresh18 | Separate three-seed fresh raw-causal research model | 0.597587741 | 0.613404587 |
| Fresh20 | Separate three-seed fresh raw-causal research model | 0.597645916 | 0.613395404 |

No alternative has point-better LL than C0 at alpha .5 on these 194 fixtures; differences are very small for the Core18/fresh alternatives, and two months are not a general model ranking. The corresponding raw-alpha scores are not all served probabilities: alpha 1 is included as a control. Do not compare the prior separate canonical 3,103-fixture fresh20 result (positive primary, but failing its A0 guard) directly to this 194-fixture ASSESS table as model deterioration; cohorts and tests differ.

Locked baseline-gate simulated ASSESS results by model and clock are also mixed. At alpha .5, in order first-ready / quote-triggered / even-hour / odd-hour:

| Model | Bets | Net P&L (PLN) | Stake ROI |
|---|---|---|---|
| C0 | 3 / 15 / 2 / 12 | +343.50 / +255.56 / −254.73 / +268.73 | +98.42% / +13.17% / −100% / +17.25% |
| Legacy C0-D/Core18 | 2 / 9 / 2 / 7 | +176.11 / −381.19 / −253.99 / −158.54 | +65.68% / −33.45% / −100% / −17.29% |
| Core17T | 2 / 9 / 2 / 7 | +162.08 / −412.77 / −256.37 / −181.45 | +60.12% / −35.63% / −100% / −19.57% |
| Slim18 | 2 / 10 / 2 / 7 | +155.29 / +103.78 / −279.10 / +139.58 | +54.47% / +8.14% / −100% / +15.26% |
| Fresh18 | 2 / 9 / 2 / 7 | +172.93 / −381.33 / −253.78 / −158.86 | +64.76% / −33.58% / −100% / −17.40% |
| Fresh20 | 2 / 8 / 2 / 6 | +173.97 / −304.10 / −269.77 / −64.62 | +63.19% / −28.37% / −100% / −7.77% |

Rows are fixed matched controls (`policy/locked_model_policy_comparisons.parquet`), not model-specific training or selected strategies. They should not be read as a stable ranking or independent financial validation.

## Frozen financial search and selection outcome

The core surface contains 7 sports alpha values × 5 net minimum-EV thresholds × 4 DRO epsilon values = 140 cells, each assessed on all four fixed timing policies. The unchanged baseline is alpha .5, minimum net EV .05, epsilon .025, plus the source policy's existing gates. Duplicate configurations were removed. The run evaluated 350 primary configurations (1,400 primary config-by-timing combinations). Three diagnostic-only source-safety controls are reported separately in their 24-row table. The 2,810 main case-summary rows comprise 2,800 primary config/timing/partition rows plus eight proper-score-alpha fit-comparator rows and two noncausal-oracle rows. Results live in `policy/results.json`, `case_summaries.parquet`, `gate_support.parquet`, and `policy_contrasts.json`.

Policy choice used strict TRAIN only: at least 20 settled bets across at least two active months, ranked by net profit per full preselection TRAIN fixture universe, with frozen tie-breaks. Results:

| Fixed timing | Maximum eligible-run TRAIN support (settled bets) | Minimum required | TRAIN active months at that support | Selection |
|---|---:|---:|---:|---|
| First-ready | 6 | 20 | No eligible configuration | NOT_SELECTED_INSUFFICIENT_SUPPORT |
| Quote-triggered, first eligible state | 15 | 20 | No eligible configuration | NOT_SELECTED_INSUFFICIENT_SUPPORT |
| UTC even-hour 2h | 0 | 20 | No eligible configuration | NOT_SELECTED_INSUFFICIENT_SUPPORT |
| UTC odd-hour 2h | 13 | 20 | No eligible configuration | NOT_SELECTED_INSUFFICIENT_SUPPORT |

The protocol also requires two active months; no timing family approached the bet minimum. All selected-policy contrasts versus baseline, matched-market alpha 0 and pure-C0 alpha 1 are therefore `null` with `NOT_SELECTED_INSUFFICIENT_SUPPORT`; there is no confirmatory selected-policy inference. Every ASSESS cell remains descriptive. In particular, the top assessment cell is not a recommendation or evidence of a chosen policy.

### Unchanged-gate alpha comparison: ASSESS only

Every row uses the exact same baseline gates, full ASSESS fixture denominator of 846, 12% tax and capital mechanics. Stake ROI is net P&L divided by turnover (stake), not bankroll ROI.

| Sports alpha | First-ready bets / net PLN / stake ROI | Quote-triggered bets / net PLN / stake ROI | Even 2h bets / net PLN / stake ROI | Odd 2h bets / net PLN / stake ROI |
|---:|---|---|---|---|
| 0 | 0 / 0 / n.a. | 0 / 0 / n.a. | 0 / 0 / n.a. | 0 / 0 / n.a. |
| .25 | 0 / 0 / n.a. | 0 / 0 / n.a. | 0 / 0 / n.a. | 0 / 0 / n.a. |
| .35 | 0 / 0 / n.a. | 1 / −87.72 / −100% | 1 / −87.72 / −100% | 0 / 0 / n.a. |
| .50 | 3 / +343.50 / +98.42% | 15 / +255.56 / +13.17% | 2 / −254.73 / −100% | 12 / +268.73 / +17.25% |
| .65 | 5 / +175.32 / +23.38% | 13 / −139.74 / −8.12% | 2 / −271.20 / −100% | 11 / +67.95 / +4.52% |
| .75 | 5 / −247.03 / −34.84% | 13 / −8.42 / −0.47% | 1 / −150 / −100% | 12 / +133.57 / +8.05% |
| 1 | 8 / −738.00 / −61.50% | 15 / −386.28 / −17.91% | 2 / −79.56 / −26.52% | 11 / −833.42 / −52.61% |

This table is intentionally not an assessment-alpha selection. The positive first-ready and odd-hour alpha .5 points, for example, consist of 3 and 12 bets across two ASSESS months, respectively, and were not selected by TRAIN support. The market-only matched alpha 0 control's zero bets mean the sport-dependent baseline gates did not permit bets there; it does not mean no market quotes exist.

### DRO, EV and the remaining gates

The full `case_summaries.parquet` is the numeric record for all cells. Across the 560 core ASSESS timing/configuration rows, 131 have positive realized P&L and 175 have zero bets; positive rows reflect a broad searched surface, not 131 independent winning trials. For epsilon .01 with net minimum EV at least .05, 84 ASSESS timing/configuration rows were tested: 25 happened to be positive, 22 had no bets, and realized P&L ranged from −738 PLN to +752.51 PLN. The latter maximum is a hindsight cell: alpha .5, minimum net EV .10, epsilon .01, quote-triggered timing, seven bets, two active months. The comparable best point is equally descriptive at epsilon .025 (same seven-bet result). At alpha .5 / quote-triggered timing, epsilon 0 and .01 each resulted in 16 bets/+246.98 PLN (+14.24% stake ROI); epsilon .025 gave 15/+255.56 (+13.17%); epsilon .05 gave 1/−150. Such nonmonotonic realized results and minute samples do not select epsilon.

The protocol explicitly treats epsilon as a downside stress, `p_stress=max(0,p_nominal−epsilon)`, and requires both a nominal and stressed net-EV gate. It is not a fitted ambiguity set, a per-bet guarantee, or proof of a probabilistic DRO bound. The frozen redundancy check follows directly from the tax-adjusted EV formula: reducing probability by epsilon reduces nominal net EV by at most `epsilon × odds × (1−tax)`. With epsilon .01, odds at most 4, and 12% tax, the maximum reduction is .0352; therefore any nominal net EV at least .05 still has stressed net EV at least .0148 > 0. This explains why epsilon 0 and .01 are redundant under those exact conditions. It does not make the condition universal beyond odds<=4 or minimum net EV>=.05.

At each alpha, the frozen one-factor comparisons changed one of the existing source gates around the baseline; the alternatives were max odds 3/6/unbounded, max EV .15/unbounded, logit disagreement .5/1.25/unbounded, Tier-1 market gap .05/.12/unbounded, BO1 market gap .08/.16/unbounded, dynamic hurdle off, universal rating guard off, longshot threshold 3/4, max longshot EV .15/unbounded, BO1 max odds 3/4, and coinflip-market-gap cap .05/.10. Availability-only sensitivities tested rating disagreement .05/.10/.15/unbounded and negative CLV drift −.03/−.015/0/unbounded. Exact deduplicated variants are in `protocol_v2.json` and every result is in `case_summaries.parquet`.

The primary-only family counts below filter `case_summaries.parquet` to ASSESS, the four timing policies named in the protocol, and primary configurations only; proper-fit comparators and the two oracle rows are excluded. This yields 1,400 primary ASSESS configuration-by-timing rows, 311 with positive realized P&L. For each factor family, the table reports positive rows / total timing rows and the largest raw ASSESS P&L observed. These are not selected estimates; the full family was searched and every metric is subject to the assessment search.

| Sensitivity family | Positive ASSESS rows / total | Largest ASSESS P&L observed (PLN) |
|---|---:|---:|
| Core 140 configurations | 131 / 560 | +752.51 |
| Max odds | 25 / 84 | +751.87 |
| Max logit disagreement | 18 / 84 | +475.32 |
| Max Tier-1 market gap | 14 / 84 | +533.93 |
| Max BO1 market gap | 16 / 84 | +343.50 |
| Max coinflip market gap | 18 / 56 | +343.50 |
| Max net-EV cap | 11 / 56 | +343.50 |
| Max longshot EV cap | 11 / 56 | +343.50 |
| Longshot odds threshold | 12 / 56 | +343.50 |
| BO1 max odds | 14 / 56 | +388.07 |
| Dynamic hurdle off | 5 / 28 | +343.50 |
| Universal rating guard off | 6 / 28 | +343.50 |
| Rating-disagreement availability | 18 / 84 | +343.50 |
| Negative-CLV drift availability | 12 / 84 | +343.50 |

There is no claim that these gate families caused old returns or that a threshold can be chosen from ASSESS. Real EXP/rating availability is missing; the rating gates are not evidence of a fitted rating guard. Same-cutoff drift is zero by design, so negative-closing-drift thresholds are non-identifiable and cannot be interpreted as CLV filters.

### Source-safety-only ablation, market-only extension, and quote denominators

Three joint diagnostic controls at alpha 0, .5 and 1 switched off the statistical/model-veto family as a whole, including disagreement/rating/market-gap gates, DRO stress (epsilon=0), the dynamic hurdle, and max-EV caps. They retained minimum net EV .05 and maximum odds 4, plus identity/clock/300-second TTL/two-other-books, tax, one-fixture, bankroll and settlement safety. They are not selection-eligible and are not a candidate production policy. At alpha .5, ASSESS first-ready had 7 bets/+147.36 PLN; quote-triggered had 21/+38.46; even-hour had 2/−153.26; odd-hour had 19/+121.34. At alpha 1, quote-triggered had 51/−752.88 PLN. At alpha 0, quote-triggered had 2/+56.71. Removing this combined statistical gate family increases opportunities but does not yield a selected/supportable strategy. Because these are joint controls, the incremental effect of any single removed gate cannot be inferred from their P&L.

The separate market-only all-quote extension is not a C0-padded control. Across the 1,771 fixture cash denominator it had 245,702 observed events, including 116,532 with no native probability; its odd-grid projection contained 69,394 market events. The quote-readiness panel contained 545 fixtures; the separate broader raw-quote coverage set comprised 1,371 fixtures without usable C0 daily states. Its cash comparison explicitly uses market-only probabilities and disables sports-base-dependent disagreement/rating/Tier/BO1/coinflip guards instead of fabricating them. Quote-triggered TRAIN had one −150 PLN settled bet; ASSESS had one two-wallet pending fixture (144791) with 150 PLN reserved in each independent wallet and no settled P&L. The canonical native label is known (`y=0`, a loss if accepted), but `native_effective_release_day` is absent, `label_available_at` and `settle_at` are `NaT`, and the native release clock is unknown; under the frozen contract that label is not certifiably settled, so ROI is undefined—not a zero-return or a win. Odd-hour had one −121.37 TRAIN bet and the same pending ASSESS fixture in two wallets; first-ready and even-hour had zero. These sparse unresolved market-only results cannot be compared as if they share the matched C0 candidate universe. No C0 imputation, cash-clock fallback, or close/CLV substitute.

### Near-EV band and uncertainty

The report keeps the protocol's near-EV ±2 percentage-point containment criterion distinct from realized positive value. It passed 0 of 2,810 summarized case rows. At the fixed alpha .5 baseline quote-triggered ASSESS row, 12 of 15 bets fell in the near-EV band; their mean nominal-minus-realized return error was +19.8405 pp (stake-weighted +21.3068 pp), and the status is `FAIL`. Its monthly error interval is +11.9622 to +106.5017 pp, but it is based on only two ASSESS months, not a certified population uncertainty interval. The same row's near-band realized return mean was −12.3667% (`NON_POSITIVE`).

The baseline first-ready ASSESS near-band subset had two bets with +54.44% mean realized return (`POSITIVE`), but the containment status is `INSUFFICIENT_SUPPORT`; monthly error resamples are undefined in 1,243/5,000 draws. Positive realized value is therefore not evidence that the ±2pp calibration gate passes. Even-hour near-band support also fails as insufficient; odd-hour's nine near-band bets have negative mean realized return. Across the 2,810 rows the criterion never passes, and sparse months often null otherwise tempting intervals. `policy_contrasts.json` records no selected-policy contrast because selection support failed; there is no valid selected alpha-vs-control inference.

## How earlier edge claims disappeared: timing, money, and measurement

### Earlier timing runs are separate evidence

The prior fixed UTC even-hour 2h run selected two ASSESS bets (145,808 / 145,814) for −254.731191 PLN; across 200 fixed random-grid schedules the median was −150 PLN. The old max-model-EV snapshot oracle yielded 21 bets, eight wins, +319.226759 PLN and 10.66% stake ROI, but chooses a future/highest-EV snapshot and is explicitly noncausal. These are a prior run, not rows from the current 350-configuration study and not evidence to tune the current grid. The current odd-hour/even-hour phases are both fixed and evaluated separately.

The separate quote-cadence audit materially sharpens, but does not settle, the timing explanation (`quote_cadence_audit.json`, `timing_event_provenance.json`). Among 248,735 raw quotes, 202,646 (81.47%) fell at even UTC hours; 46,089 fell at odd UTC hours. The modal quote minute was :55 (150,242 rows), with another 15,872 at :56. Before financial gates, the exact odd-hour grid retained 4,805 native decision keys / 44,966 event rows, compared with 243 keys / 2,580 rows for the native-admitted even-hour grid (the quote-cadence artifact separately records 3,754 raw even-grid rows). Since a 300-second freshness limit expires well before the next fixed even-hour tick for a large share of :55 updates, a fixed two-hour phase is poorly aligned with this recorded quote cadence. This provides a plausible sampling/TTL-alignment mechanism for why fixed-phase replay can lose or alter opportunities without any model change; it does not prove that timing alone caused a past yield difference.

Crucially, all 19 alpha .5 baseline quote-triggered accepted bets (4 TRAIN and 15 ASSESS) are verified in `timing_event_provenance.json` as actual retained snapshot events (`is_snapshot=true`, `is_grid=false`). Thus the positive 15-bet quote-triggered ASSESS result is not a fabricated odd-grid interpolation. The generic quote-triggered implementation can also consider retained grid states, so these event flags describe the actual selected rows, not an assertion that no other state type was eligible. Neither snapshot receipt nor historical timestamps prove an operator could observe or fill every quote at that moment. The appropriate next test is forward event-arrival / observation-latency capture with exact quote identity, not post-hoc selection of an odd UTC deployment phase or blind relaxation of TTL.

The five historical T−1h ledger entries have decision time exactly one hour before recorded start. Ages at decision: fixture 62,960: 293s (fresh); 145,061: 4,761s; 145,062: 3,858s; 145,454: 25,462s; 145,808: 786s. Thus the current 300-second TTL would reject four of these five historical T−1h quote states. The historical T−1h profit was +226.516527 PLN across five rows. Only fixture 145,808 overlaps the separate fixed 2h pair (145,808 / 145,814), which lost; the two-bet pair's total was −254.731191 PLN. The 2h subset is not a fresh replay of the same five: even-hour timing, stale-age eligibility and first-observed-state requirements change cohort and price. The freshest T−1h case 62,960 is absent from the fixed 2h pair. Quotes can also qualify earlier than a fixed 2h tick: the current first-eligible quote-triggered alpha .5 ASSESS policy selected 15 actual snapshot bets. These bounded facts are consistent with cadence/TTL effects but do not recover the user's claimed 80% drop in prior bet count. In particular, no causal 50-to-10 count reconstruction is possible from the old 50-row CSV, which lacks exact fixture, side, bookmaker and quote IDs. Neither stale-age counts nor cadence alone establish that timing caused the claimed count drop.

### Historical 50 ledger and overwrite provenance

The exact versions are:

- Pre-overwrite: `data/01_raw/c0_edge_recovery/c0_edge_recovery_20261022_01/legacy_50_preoverwrite.csv`, SHA-256 `d4b960e204b99ae7288cfbb3912eac0b3296a9b08666a1d2d0dff9b8dc9cd434`.
- Retained overwritten CSV: `data/08_reporting/optimal_strategy_simulation_ledger.csv`, SHA-256 `b51ee7b41d666a6c96de88d1fe25d669a640bbccd1cdf75f585c545c79dfd90a`.
- Exact comparison: `data/08_reporting/c0_edge_recovery/c0_edge_recovery_20261022_01/recovery/legacy_50_version_comparison.json`; provenance summary: `recovery/legacy_50_history.json`.

All 50 row ordinals/order match. Pre-overwrite mean model EV was 13.4051269%, min 5.2716%, and 50/50 fell in the stated 5–25% EV band. The retained post-overwrite mean was 8.58879272%, minimum −7.816031%, and only 31/50 fell in the 5–25% band. The diff says only `p_hybrid` and `model_ev_pct` changed; date, teams, odds, outcomes, stakes, profit, bankroll and drawdown did not. Both versions report 1,715.41 PLN profit and 7,173.15 PLN turnover, 29 wins, 23.914319% yield, 17.1541% initial-capital return and 1.575807 profit factor. These are recorded ledger totals, not independently recovered execution profit.

A Stage05 `historical_probability_overwrite_receipt.json` and `legacy_50_history.json` trace a later inline overwrite dated 2026-10-20 01:27:10.572Z (session `01a14799-6fad-7000-82d4-c449dbd3bfe4`, message `f3cd3c70`, operation `call_409441`). It scanned sports alpha [.50,.60,.65,.70,.75] against known net CLV in the retained ledger, making the scan close-informed using future-close information. This evidence does not establish that winner labels were used to fit or select alpha. The overwrite then used the first same-date/bookmaker quote within odds ±.02, not exact canonical fixture, side, orientation or quote IDs, and fell back to own q or `1/(odds*1.055)` if there was no approximate match. This is not the original creator/selection algorithm; that remains unknown. The overwrite is post-hoc evidence about the columns, not a valid re-score of the old tickets. No original canonical match ID, selected side ID, bookmaker ID or quote ID is present for any of the 50; exact-ID attribution is zero for all rows (`legacy_50_exact_id_attribution.csv`). No fuzzy name/date/probability join is substituted.

The pre-overwrite mean 13.41% vs overwritten 8.59% is **not model deterioration** and neither percentage is a validated true edge: the later blend rewrite used non-exact matches and a post-hoc close-informed alpha scan. The rewrite also does not prove that the original bets lost, because profit/stakes/outcomes stayed put; their identities and causal probabilities remain unverified. The retained CSV's own mean closing odds (1.9628) differs from the earlier report's 2.06; the CSV and financial JSON agree on profit/turnover to tiny rounding but differ from the Oct-12 published report by +3.74 PLN profit, −20.89 PLN turnover and −0.0972 mean close odds. Which source represented the original row-level ledger cannot be resolved from this evidence.

The purported 43-bet C0-D ledger is not an independent 43-ticket list. Forty-three distinct canonical IDs with model-A trajectory support are separate from an imported 50-row ledger filtered by team/date ±1 day and probability tolerance; the old 43-ticket identity is not retained. Existing 21-bet C0-D-Core Cohort-B optimizer output also has unverified saved-output provenance. Current code's later A-selection/B-disjointness does not retroactively establish how that old output was generated. Therefore neither 43 nor 21 supplies exact recovered historical ticket profit.

### CLV, tax, DRO, calibration and statistical claims

The old retained 50 has no missing-close rows, but it lacks independent close quote IDs/timestamps and exact bet identities. Stored raw/net CLV means of 19.129216% / 4.83371% (45/50 raw-positive, 33/50 net-positive) cannot serve as independently verified closing-line value. The prior source audit identifies other construction defects across the older studies: same-consensus fair price reused as its CLV reference (CLV by construction); T−1h placed prices taken from close; some legacy paths defaulted missing close to p=.5; a futures quote-selected view; known outcome/side orientation in some generators; IID bet-level inference for clustered rows; a “Sharpe” calculation that was t-statistic-like rather than annualized; and a claimed 99% “guarantee” based on a conditional normal approximation. These remain historical claims from preserved audits, not outputs of this new policy run. Current close coverage is zero, CLV is null, and no missing close is replaced with .5.

The historical “Zero-Disappointment Guarantee” subtracts a stake-weighted average error; it is not a per-bet DRO guarantee. The current stress threshold is a deterministic `max(0,p−epsilon)` scenario, not a calibrated distributional ambiguity set. Tax is 12% in this frozen cash simulation, but tax-adjusted return cannot repair missing ticket identity, stale prices or selection bias. A nominal EV hurdle and positive observed P&L remain distinct from uncertainty/near-EV criteria.

## Production source snapshot and source-safety boundary

Receipt: `data/08_reporting/c0_production_counterfactual/c0_production_20261006_01/export_receipt.json`. It records a read-only transaction (`transaction_read_only=on`) to `192.168.1.17`, extracted at 2026-10-05 22:13:08.452166 UTC. The repeatable-read snapshot exported eight selected tables, including 1,771 matches, 248,735 quotes, 11 bookmaker references, results/mappings/events/clocks; there were no prediction rows. The export is observed production-source data, but the C0 predictions in this study are reconstructed from saved daily states and actual saved weights. The DB export does not authenticate that a prediction or bet was issued, which wallet was owned, whether a quoted price was fillable, or exact scheduled opening/start/closing times. Quote-embedded schedules were not exported; minimum canonical/event clocks are conservative scenarios. Initial unbounded raw-payload export exceeded local memory and the schedule projection timed out and rolled back; the final numeric/name/clock projection used a fresh read-only snapshot. No production write occurred and no new live query was made.

The policy separates market-only coverage from native-covered model scoring and never imputes C0 probabilities into absent-native fixtures. Live source authorization remains absent; CLV remains unavailable; EXP/rating gates remain `NOT_EVALUATED`; G0 remains blocked. The prior 10/10 readiness/attempt journal is preserved as a historical result and does not provide source authorization. Oct-31 readiness and Nov-1 start dates are unchanged. Nothing in this historical audit changes those operational states.

## Permanent repairs, audit defects, and verification status
### Verified existing invariant (unchanged in this audit)
Application series BO conversion is applied once at the model-family boundary; blend uses the series probability once, and A/B outputs complement. This was verified as existing behavior; no BO-related serving change or repair is attributed to this work.

### Confirmed permanent repairs
- Production policy SSOT rejects own settlement at or before decision; unknown latest prediction fails closed; the standalone CLI uses the real owned-wallet resolver. The CLI wallet smoke is a controlled in-process fixture only: wallet cash 1,000 plus 200 open stakes yields 1,200 bankroll and 30.12 suggested stake; explicit 2,200 override doubles it to 60.24; no wallet yields zero suggestions. It is not a backtest.
- Real old Shin defect: `src/analysis/betting_guards_a_to_d.devig_shin` used `z=sum(1/odds)-1` in the Shin formula and was wrong for asymmetric odds. At odds 1.2/4 it returned qA=.7908162523 vs independent/root Shin .7916666667. The exact two-outcome additive solution was repaired. Archived pre-repair source is `pre_repair_source/betting_guards_a_to_d.py`, SHA-256 `2d3ac50a16f1127f99b9d2815bb0ed0c3aadb4fc1ed067bc4a175a4a341c0a02`; four targeted regressions were red before the fix, parent-reported root-smoke maximum error 2.19e−14, and the affected test suite passed afterward. Existing near-zero/negative-vig proportional fallback is unchanged. Current edge-audit financial q is multiplicative and does not use this helper; no profit recovery is attributed to the correction.

### New audit preflight defects are not historical product defects

Earlier preflight/verifier issues were in the new audit wrappers, not historical serving policies: an original oracle omitted 25 boundary-excluded focal rows; a strict alpha verifier compared an argmin to an exact root under metric tolerance instead of objective regret; an initial mechanics preflight exposed source-shape/clock assumptions; the first policy run failed a TypeError after 1,206 seconds before a repaired runner passed; and a timing-proof preflight treated the known stored `y=0` label for fixture 144791 as if its settlement clock were available. In fact, `native_effective_release_day` is absent and both `label_available_at` and `settle_at` are `NaT` (`native_release_clock_unknown`): two 150 PLN entries across independent wallets remain pending under the frozen contract, despite the stored label indicating a loss if accepted. Other verifier-only preflight mistakes involved wrong native-artifact fields, an integer-versus-`is_live` assertion, confusing the ±2pp near-band tolerance with the 5–10pp selector band, NumPy JSON serialization/scalar helpers, and raw-outcome value versus label availability clock. These were corrected in the oracle/verifier layer only; formal financial outputs, gates and tolerances were not changed. The clock-preflight issue is a verifier assumption, not a financial model change, and its result is not financial evidence. Preservation differs by artifact: failed-run logs and bindings (including the original `execution_binding_policy.json` hash and tool trace) remain available, but the exact original source copy for the first failed financial runner is unavailable. Named byte-exact verifier/preflight sources are preserved separately, including the later clock-preflight script; do not confuse that preserved verifier source with the first financial runner source. The corrected alpha oracle uses the frozen cohort and objective-regret criterion. No base-model fits were run; the only formal alpha fit was the single strict TRAIN scalar estimate, replayed by the same optimizer in the two financial attempts and one clock diagnostic. No search grid or scoring tolerance was changed to improve P&L. None of these new audit-harness failures is evidence that production historically made those mistakes, and the earlier alpha-oracle failure is not converted into a false PASS by editing the archived artifact.

### Independent numeric verification: bounded PASS

The three retained independent receipts are complete: `independent/independent.json` is `COMPLETE`; `independent/quote_clock_verification.json` and `independent/monthly_statistics_verification.json` are `PASS`; each reports an empty `failed_objects` list. The core checker independently compared 21 alpha score rows and 126 locked-model score rows (147 score rows total), over the 910-row focal scoring surface, and replayed the retained wallet ledgers. The score discrepancies are at floating-point noise (alpha maximum log-loss error 3.33e−16 and Brier error 1.39e−16; model maximum log-loss error 4.44e−16 and Brier error 1.94e−16). Its TRAIN alpha oracle reports objective regret 1.11e−16, parameter delta 1.3266e−8, and maximum probability delta 3.5235e−9.

The 210 retained Decimal wallets comprise 925 ledger rows: 923 settled and two pending entries for fixture 144791 whose known stored `y=0` is not settled under the frozen label-release-clock rule. Wallet replay has zero cash-accounting error and maximum stake delta 4.7954e−13 PLN (maximum model-EV delta 3.2432e−16). The separate quote verifier checked 260 unique accepted source-quote tuples across the 210 wallets, with zero failed tuples and parity across eight baseline C0 model/wallet pairs. The monthly verifier recomputed 2,986 monthly aggregate rows and 48 alpha paired-interval objects; maximum interval arithmetic error was 1.2195e−15, with the wallet near-band and cash-month joins checked. These establish reproducibility of the retained score, wallet, accepted-quote and summary arithmetic—not prospective performance or source authenticity.

Coverage is deliberately bounded: not all primary configurations were independently wallet-replayed; non-selected quote opportunities were not independently reconstructed; model weights/internals, actual production clocks, source authenticity and live execution are not certified. The ledger uses continuous decimal PLN, not proof of cent-rounded filled tickets. The monthly verifier explicitly says it checks aggregate bootstrap arithmetic and detailed wallet joins, not every primary wallet; the quote verifier covers accepted retained bets only. No model, alpha, or policy is promoted by these proofs.

Parent-run execution checks also reported for this run: 122 affected tests passed (including actual native namespace/tensor paths); the 13 mechanics regression group passed on repeat; Ruff F821 passed across the five audit modules; 129/129 protected-source hash closures remained unchanged; two independent static reviews reported no required findings. One pre-existing `httpx` warning remains, and this was not a full-application verification. These checks establish bounded implementation hygiene, not profit certification or authentic production execution.

## Final decision and concrete next action

**Claimable:** a strict earlier-only scalar alpha can be fitted on 83 fixtures; its historical proper-score contrast is measurable and modest; actual saved models and frozen gate/timing surfaces can be reconstructed and compared; the old 50-row reported hybrid probabilities/EV were later rewritten under a non-exact, post-hoc close-informed procedure; several old financial claims cannot be authenticated because identity, clock, execution and independent close evidence are missing. The overwrite evidence does not establish winner-label fitting or selection. In the fixed historical ASSESS run, different small portfolios show both gains and losses.

**Not claimable:** a promoted alpha; an eligible TRAIN policy; a selected assessment winner; an independent future profit edge; true CLV; authenticated historical bet tickets or wallet outcomes; a global proof that no edge exists; permission to activate a source; that the Shin repair recovers old profit; or that freshness alone explains the claimed 80% drop in prior bet count.

**Next action:** keep serving policy/weights/manifest unchanged and G0 blocked. For any future authorization decision, first obtain prospective, timestamped, immutable records containing canonical fixture ID, selected side, bookmaker and quote ID, decision/settlement/start clocks, exact prediction and available market inputs, owned wallet/open stakes, submitted/fill outcome, and an independent close quote. Then evaluate a frozen policy on a genuinely future holdout with at least the predeclared TRAIN support and monthly uncertainty requirements. Do not widen the historical grid or promote the best ASSESS cell.

## Reproducibility and evidence paths

Binding protocol: `data/05_model_input/c0_edge_recovery/c0_edge_recovery_20261022_01/protocol_v2.json`. Run outputs: `data/08_reporting/c0_edge_recovery/c0_edge_recovery_20261022_01/`.

Commands for the actual audit CLIs:

```bash
python -m src.analysis.c0_edge_alpha_audit --protocol data/05_model_input/c0_edge_recovery/c0_edge_recovery_20261022_01/protocol_v2.json --prepare
python -m src.analysis.c0_edge_alpha_audit --protocol data/05_model_input/c0_edge_recovery/c0_edge_recovery_20261022_01/protocol_v2.json --run
python -m src.models.c0_edge_model_audit --protocol data/05_model_input/c0_edge_recovery/c0_edge_recovery_20261022_01/protocol_v2.json
python -m src.analysis.c0_edge_policy_audit --protocol data/05_model_input/c0_edge_recovery/c0_edge_recovery_20261022_01/protocol_v2.json
python -m src.analysis.c0_edge_mechanics_audit --protocol data/05_model_input/c0_edge_recovery/c0_edge_recovery_20261022_01/protocol_v2.json
python -m src.analysis.c0_edge_recovery_attribution --protocol data/05_model_input/c0_edge_recovery/c0_edge_recovery_20261022_01/protocol_v2.json
```

Independent proof receipt commands (run monthly and quote checks before the combined core receipt):

```bash
python data/08_reporting/c0_edge_recovery/c0_edge_recovery_20261022_01/independent/verify_monthly_statistics.py
python data/08_reporting/c0_edge_recovery/c0_edge_recovery_20261022_01/independent/verify_quote_clocks.py
python data/08_reporting/c0_edge_recovery/c0_edge_recovery_20261022_01/independent/verify_edge_recovery.py
```

These verifiers consume retained outputs; they are not a request to rerun the historical alpha/model/policy CLIs.

Key evidence: alpha results `alpha/alpha_audit.json`, `alpha/score_curves.csv`, `alpha/paired_monthly_intervals.json`; actual model receipt and scores `models/model_audit_receipt.json`, `models/model_alpha_scores.csv`; policy selection/results `policy/results.json`, `policy/selection_ids.json`, `policy/policy_contrasts.json`, `policy/case_summaries.parquet`, `policy/locked_model_policy_comparisons.parquet`, `policy/market_only_coverage.json`, `policy/market_only_all_quote_summaries.parquet`, `policy/source_safety_only_controls.parquet`; timing evidence `quote_cadence_audit.json`, `timing_event_provenance.json`; extracted finance/DRO values `financial_summary_extract.json`, `dro_gate_extract.json`; mechanics `mechanics/current_mechanics_audit.json`, `mechanics/shin_reconciliation.json`, `mechanics/standalone_cli_wallet_smoke.json`; historical reconciliation `recovery/legacy_50_version_comparison.json`, `recovery/legacy_50_history.json`, `recovery/legacy_50_exact_id_attribution.csv`, `recovery/financial_claim_status_matrix.csv`; independent checker artifacts `independent/independent.json` and archived initial failure `independent/independent.preflight.json`.
