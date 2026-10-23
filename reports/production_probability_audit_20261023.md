# Production odds versus model probabilities — 2026-10-23

## Decision

**COMPLETED / HISTORICAL_CONDITIONAL_COMPARISON. Superiority over bookmakers is not established for the selected local fresh20+regional model, overall or at any tested lead-time interval.** This is not evidence of equivalence or proof that profitable betting is impossible. It does not establish financial edge, source authorization or production readiness.

The production API advertises original 16-history C0; the selected local model actually exercised here is genuine fresh20+regional. The exported prediction history contains older model identities, not saved C0/fresh20 forecasts. Do not attribute the old tabular/EXP081 results below to current fresh20 or claim this is recorded live fresh20 performance.

Run: `production_probability_20261023_01`. No model/calibrator fitting, alpha/threshold/DRO search, scraping, deployment, migrations, betting, scheduler execution or production writes occurred.

## 1. Source acquisition and inventory

Read-only access to `192.168.1.17`; primary tables were exported in one repeatable-read, read-only transaction. Every retained row was exported without date, outcome, live, EV or recommendation filtering. The exact primary transaction clock/LSN was not captured. The separately observed server clock **2026-10-22T23:45:52.630382Z** is a later bound, not the original transaction identity. Subsequent live counts grew; this is a complete retained snapshot, not a claim of capturing an ongoing stream indefinitely.

| Export | Rows |
|---|---:|
| Basic odds snapshots | 273,194 |
| Outcome-price observations | 106,288 |
| Prop odds snapshots | 55 |
| Canonical predictions | 791,360 |
| Canonical matches | 1,883 |
| GOL.GG matches | 42,492 |
| GOL.GG games | 70,517 |
| GOL.GG game-player rows | 705,165 |

All **27 primary tables** and **10 separate quarantine/repair tables** were independently checked against receipt row counts, byte counts and SHA-256 digests: **3,482,854,592 bytes**, including **5,397 quarantined rows**. Quarantined duplicate/repair observations are preserved separately, not treated as additional clean independent quotes. Accounts, wallets and unrelated request logs were not needed for probability scoring and were not exported.

Raw basic odds span 2026-05-28 through 2026-10-22. This is not a 2021–2026 production-odds study; the older OddsPortal/OddsPapi work remains separate.

Evidence:

- `data/01_raw/production_probability/production_probability_20261023_01/export_receipt.json`
- `data/01_raw/production_probability/production_probability_20261023_01/followup_runtime_identity.json`
- `data/01_raw/production_probability/production_probability_20261023_01/archive_odds/archive_export_receipt.json`
- `data/08_reporting/production_probability/production_probability_20261023_01/verification/raw_export_closure.json`

### Exact outcome pairing

Event IDs in outcome rows are external bookmaker event IDs, not internal event-table primary keys. Exact event, market, scrape-run and observation-clock grouping produced **52,631 binary A/B pairs** from **105,262 observations**. Another **1,026 observations** lack exact event metadata and remain explicitly unpaired.

Of those pairs, **42,770** reproduce an existing quote state with an explicit recorded non-live flag; they are duplicate representations, not new evidence. The remaining **9,861** have unknown source live flags and are not silently classified as pre-match. All 106,288 constituent observations have pairing-accounting rows. The 55 prop rows are inventoried, not compared with series-win probabilities.

## 2. Model identity and native replay

The retained `/matches/models/active` response reports:

- Operational: `Causal-C0:c0-native-2026-w32-e12-v1`, feature version `native-c0-history16-v1`.
- Hybrid: `Hybrid-Operational-Market:c0-native-2026-w32-e12-v1-a0.50-t1.00`, logit shrinkage, alpha 0.5 and temperature 1.

This is an advertised runtime configuration, not an independently traced successful production inference. It does **not** report the selected local fresh20+regional identity.

The actual local replay uses the saved model under `data/06_models/c0_distilled_verification/c0d_repro_20261021_01/fresh/20t`, its genuine three-seed assembly, and exactly one causal regional input transformation. Twenty tokens means **19 observed positions: three 365/180/90-day summary anchors plus 16 recent maps, and one learned token**; it does not mean 20 recent maps. Regional transformation retains the full FLOAT32 19-position input contract. Final probabilities are already series probabilities; no second BO projection or probability blend was added.

The new audit history preserves the frozen **41,915 canonical series** and appends **263 complete DB-only series** using the existing serving extension frontier, date >=2026-09-08. Another **314 older DB-only IDs** remain ledgered outside that frontier. The resulting **42,178-series** source reaches 2026-10-22 and passed the native history parser. SHA-256: `403fae54ada0865b811c5108a3280b03791b8962b8ebef740359055249fcf807`.

The replay requests strict prior-UTC-day, exact-competition-context rosters. It never substitutes actual target lineups or current mutable rosters when prior support is absent. The completed 4,308 fixture-day requests produced:

| Native state outcome | Requests |
|---|---:|
| Available fresh20+regional probability | 2,186 |
| Both prior rosters missing | 1,679 |
| Prior B roster missing | 167 |
| Prior A roster missing | 152 |
| Request not pre-start | 122 |
| Target already released before cutoff | 2 |

The full successful replay took **700.20 seconds**, including 46.62 seconds regional replay and 602.92 seconds feature construction/inference. Earlier failed preflight attempts are retained in the execution record; they are not additional fitted recipes.

Paired outcome inventory added 13 fixture-day keys beyond the original request manifest. **None has an eligible scored source observation**, so no eligible native request was omitted. Ten fractional-clock basic quotes exposed an old mixed-format parser omission; five have known scored outcomes and already use earlier same-day states, while the other five have unknown outcomes. The parser is repaired; no scored start bound changes under the corrected mixed-format parser.

## 3. Comparison contract

For each individual bookmaker, same-clock dual-sided decimal odds are converted to proportional no-vig implied probabilities:

\[
q_A=\frac{1/o_A}{1/o_A+1/o_B},\qquad q_B=1-q_A.
\]

Raw implied probabilities and overround remain in the audit. Canonical team A is the outcome orientation, including fields historically named `p_team1`/`p_team2`. Reversed bookmaker sides are aligned before scoring.

- Select the latest recorded forecast at or before each quote; unavailable/invalid latest states do not revive older states.
- Require declared input cutoff <= forecast clock <= quote clock < the minimum retained declared scheduled-start bound.
- Aware schedules are normalized to UTC. Legacy UTC-naive upcoming schedules are additional conservative UTC-assumed bounds, not authenticated actual start times. Every one of the 100,949 primary matched quotes is also before the separately reconstructed minimum available **aware** canonical/event schedule.
- Fresh20 states are counterfactual forecasts at the earliest represented daily quote clock, retained only for later quotes in that same UTC day. No backward clock relabeling, cross-day carry or unverified cache fanout.
- Actual model generation now remains separate from the historical scenario clock. These are not issued historical fresh20 forecasts.
- First/last endpoints mean the literal retained pre-start source state, not certified market OPEN/CLOSE. An invalid or unavailable endpoint is withheld, not replaced with a more favorable earlier observation. Endpoint flags remain in all-row accounting.
- Score each actual quote's probability **before** averaging. Average losses within fixture/book/lead interval, then equal books within fixture, then equal fixtures. Overall scores pool intervals by quote counts within each fixture/book first. The comparator is average individual-book loss, **not synchronized market-consensus probability**.
- Log loss is unclipped; exact zero/one errors remain infinite rather than being dropped or softened. Brier is a second primary probability metric.
- Paired monthly-block bootstrap: 5,000 draws, seed 20261023, six observed calendar months overall. Intervals are nominal pointwise historical intervals, not adjustment for all adaptive research or horizon comparisons.

No EV, odds cap, tax, DRO, Kelly, recommendation or financial-admission gate filters the probability comparison. Authentication limits are reported separately; the conditional study is not relabeled as G0-cleared execution.

## 4. Result adjudication

A preliminary positional comparison of canonical winners with current DB score headers found 69 disagreements, including 43 replay-scored fixtures. Positional score columns were not a reliable independent winner oracle: **42 of those 43 scored cases agree with canonical winners after map results are aligned by immutable team ID**.

Example: source series 80216's frozen map IDs identify ThunderTalk as winner, while current DB header names are reversed relative to its positional score values. This does not establish that the canonical winner or the native prediction was wrong.

The map-ID audit checked 659 mapped, known-result fixtures, including **all 483 initially scored fixtures**. One contradiction remains: canonical fixture **144805**, source series **80456**. Its three quote rows are withheld from the primary cohort; the recorded winner and every model probability remain unchanged. No label was chosen by its effect on loss or payout.

The complete original canonical-label pass remains available: 483 fixtures/100,952 quote pairs, model/book LL **0.570155746/0.570910303**. It is a diagnostic, not the source-consistent primary result. The unresolved fixture was never scored in the timestamp-only stored-production cohort, so its exclusion does not change those stored scores.

## 5. Selected local fresh20+regional versus production prices

Primary source-consistent cohort: **482 fixtures and 100,949 quote comparisons**, target schedules 2026-05-29–2026-10-22. These are repeated observations on 482 fixtures, not 100,949 independent matches.

| Metric | Fresh20+regional | Paired bookmaker | Model minus book | Monthly 95% interval |
|---|---:|---:|---:|---|
| Log loss | 0.570465578 | 0.571467723 | -0.001002146 | [-0.034766911, +0.012521124] |
| Brier | 0.196392756 | 0.195336403 | +0.001056353 | [-0.012892779, +0.006932242] |

Lower is better. LL is slightly better pointwise; Brier is slightly worse pointwise. Both intervals cross zero. The equal-month LL difference is -0.012406795, interval [-0.032959640, +0.007663856], also crossing zero.

### Lead-time intervals

Intervals refer to the retained declared scheduled-start bound, not authenticated actual starts. Cohorts differ by interval and fixtures may appear in multiple intervals.

| Hours before bound | Fixtures | Quote pairs | Model LL | Book LL | Delta LL | Monthly 95% delta interval |
|---|---:|---:|---:|---:|---:|---|
| [0,2) | 394 | 2,503 | 0.574267 | 0.570170 | +0.004098 | [-0.031226, +0.020106] |
| [2,6) | 420 | 5,322 | 0.565438 | 0.563854 | +0.001584 | [-0.030238, +0.016123] |
| [6,12) | 412 | 7,450 | 0.559592 | 0.562185 | -0.002593 | [-0.040208, +0.018949] |
| [12,24) | 379 | 11,387 | 0.551566 | 0.558476 | -0.006910 | [-0.046742, +0.014443] |
| [24,48) | 315 | 16,759 | 0.556545 | 0.557576 | -0.001031 | [-0.034420, +0.011928] |
| [48,96) | 282 | 26,487 | 0.558171 | 0.558224 | -0.000053 | [-0.053378, +0.014430] |
| [96,infinity) | 206 | 31,041 | 0.567650 | 0.572986 | -0.005336 | [-0.074856, +0.006472] |

**No interval establishes superiority.** Favorable early point estimates do not select a best betting time. The >=96h cohort has five months; the other intervals have six.

A 67-block source tournament-name/year sensitivity gives overall LL interval **[-0.020328268, +0.018771748]** and Brier interval [-0.007244712, +0.009673873], again crossing zero. These are source-label proxies, not authenticated independent tournament editions.

The scored books are betclic, betfan, efortuna, lebull, pinnacle, sts, superbet and totalbet. Per-book matched scores and intervals are in `fresh20_regional_comparison_v3/headline_summary.json`. Pinnacle has only **five fixtures/one month**; it cannot establish a sharp-market advantage.

### Coverage, not financial gate attrition

The final source ledger contains **178,094 eligible known-outcome price rows on 964 fixtures**. Fresh20+regional scores 100,949 rows on 482 of those fixtures. Most model absences come from unavailable prior exact-context rosters, not EV/DRO gates. This does not prove that an actually announced roster could not support a forecast; no historical announcement evidence is invented.

Source exclusions retain 46,511 unknown-result rows, 42,770 duplicate representations, 37,977 unresolved-side rows, 9,861 live-status-unknown rows, 6,920 not-yet-started fixtures' rows, 1,571 not-prestart rows, 1,280 missing-clock rows, 838 invalid two-sided prices and three unresolved-result rows. Counts are source rows, not independent matches.

## 6. Saved production predictions: timestamp-only versus date-only scenario

All 791,360 source prediction rows are preserved. Duplicate analysis found 1,173 repeated fixture/model/clock states across 2,640 rows. Equivalent copies are collapsed; **677 conflicting same-clock states are withheld** rather than choosing a convenient probability. The ASOF state ledger contains 789,893 rows.

Its cutoff metadata is:

| Cutoff representation | Unique states |
|---|---:|
| Aware timestamp | 3,088 |
| Date only | 657,594 |
| Null | 105,800 |
| Literal text `NaN` | 23,411 |

Only 25 forecast clocks are after the export bound; future timestamps are not the main coverage problem. None of these normalized states supplies a recognized actual-generation timestamp. Stored predicted clocks and limited historical/backfill markers are declarations, not authenticated issuance.

### Timestamp-only recorded-clock cohort

All 29 model/feature identities were examined. Only three identities have eligible comparisons: two old Hybrid-Thesis variants and exp-039, with **2,449 quote-model pairs across 36 distinct fixtures**. This is not a suitable record of current C0/fresh20 production quality. The complete model-specific accounting remains under `stored_comparison_v3/`.

### Separate declared-date upper-bound scenario

It would be misleading to call every date-only record a permanently unusable probability. A separate, explicitly conditional study assumes a recorded cutoff date is the latest included UTC day and uses the **following midnight** as its input-availability upper bound. A forecast earlier than that bound remains ineligible; null/malformed cutoffs remain invalid. Raw probabilities, outcomes, prediction clocks and strict-cohort results are unchanged. This neither authenticates actual release/issuance nor clears financial admission.

This scenario yields **266,980 quote-model pairs across 18 nonempty model/feature identities**. These are overlapping model comparisons, not 266,980 distinct bets or matches. Representative operational identities are shown by their declared role, not selected by profit; every nonempty identity is retained in `stored_declared_date_scenario_v1/headline_summary.json`.

| Recorded identity / feature contract | Fixtures | Months | Model LL | Paired book LL | Delta LL | Monthly 95% interval |
|---|---:|---:|---:|---:|---:|---|
| Weekly tabular / player-team v0.2, latest-full | 626 | 4 | 0.693784 | 0.570610 | +0.123174 | [+0.114228, +0.138332] |
| Weekly tabular / player-team v0.3, ratings-v2 | 34 | 1 | 0.634442 | 0.510737 | +0.123705 | Not estimable |
| Weekly tabular / symmetric, latest-full | 135 | 2 | 0.675466 | 0.583365 | +0.092101 | [+0.076514, +0.193416] |
| Saved EXP081 sports / symmetric | 50 | 1 | 0.789821 | 0.609536 | +0.180285 | Not estimable |
| Saved EXP081 hybrid, alpha .5 / T .8 | 50 | 1 | 0.706789 | 0.609536 | +0.097254 | Not estimable |
| Saved EXP081 hybrid, alpha .5 / T 1 | 78 | 2 | 0.614722 | 0.607169 | +0.007553 | [-0.008863, +0.066457] |

The bad saved tabular/EXP081 results are **not fresh20 results**. Feature populations differ; do not rank models by comparing LL across these rows. Hybrid predictions already incorporate market information and are not odds-free bookmaker challengers. One small A1-hybrid identity has a favorable 14-fixture point estimate, but only one month and no estimable monthly interval; it does not overturn the withdrawn A1 claims or select a replacement. No alpha or temperature is selected here.

## 7. Verification and retained failures

Actually exercised:

- All 27 primary/10 quarantine exports: full row/byte/digest closure, zero mismatches.
- Full genuine native replay: 4,308 requested daily states, 2,186 available probabilities; complete native source parser execution.
- Six actual public native calls: BO1/3/5 at two same-day clocks. Post-regional input hashes exactly match bulk-replay inputs; public probabilities differ from saved bulk outputs by at most **1.660808429670979e-8**. Each two-clock input/probability pair is exact. This is integration parity; both paths reuse the native forward implementation, not independent neural arithmetic certification.
- The current replay wrapper also actually ran three BO1/3/5 cases after the interface repair: both complementary probabilities, valid status, UTC-day-start cutoff and same-day scope export correctly; input hashes match and saved-probability max error is **1.4946320070485797e-8**. This is a three-case interface smoke, not another assessment recipe.
- All **100,949** primary no-vig values independently reconstructed as a closed-form odds ratio: maximum error **2.220446049250313e-16**. All matched quote/model/cutoff/day/start inequalities checked.
- Independent explicit monthly resampling reproduces both overall LL/Brier interval endpoints exactly; 67 source-label proxy blocks checked separately.
- Clarified final protocol and parsed-clock optimization reproduce every source-consistent fixture-panel value exactly.
- **49 focused audit plus existing benchmark tests passed**, one existing Starlette/httpx deprecation warning. This is not the full application suite or migrated-database certification.
- Actual existing benchmark CLI executed for stored timestamp-only, declared-date and fresh20+regional cohorts. Latest final protocol SHA-256: `ecd7bc46e7e6fc404831141bd44aa17b3259611d3d9c4cd2c8a9195b53fd076a`.
- The actual CLI rejects a non-5000 bootstrap override with exit2 before data access; its prior silently ignored override is covered by a failing-before/passing-after regression.

Retained preflight failures include adding legacy DB-only IDs outside the existing extension frontier (exposing malformed old series 9118), a post-start request reaching validation, an undefined replay-state-marker import, an undefined lead-bin constant, an omitted bookmaker column and timezone-dtype handling for an all-null cutoff column. Two initial reversed-side unit-test expectations were wrong: raw-B odds 2 against raw-A odds 4 imply canonical-A probability 2/3, not 1/3. Those oracle expectations were corrected without changing the correct no-vig implementation. Initial reviewer backend quota failures were followed by two completed read-only reviews; their cutoff/interface/day-bound findings were resolved and endpoint semantics clarified. No failure led to another fit, outcome-picked label or probability clipping.

## 8. Artifacts and next action

Final roots:

- Raw export: `data/01_raw/production_probability/production_probability_20261023_01/`
- Audit history/pairing: `data/02_intermediate/production_probability/production_probability_20261023_01/`
- Source-consistent inputs: `data/03_primary/production_probability/production_probability_20261023_01/integrity_v7/`
- Declared-date inputs: sibling `declared_date_scenario_v1/`
- Native outputs: `data/07_model_output/production_probability/production_probability_20261023_01/integrity_v4/`
- Final current-model comparison: `data/08_reporting/production_probability/production_probability_20261023_01/fresh20_regional_comparison_v3/`
- Recorded timestamp-only comparison: sibling `stored_comparison_v3/`
- Recorded date-scenario comparison: sibling `stored_declared_date_scenario_v1/`
- Independent proofs: sibling `verification/`

The existing runner is documented in [scripts/README.md](../scripts/README.md). Occupied namespaces must not be rerun or overwritten. Raw datasets and generated artifacts remain local; no commit, push or production cutover was performed.

**Next:** reconcile the one unresolved result and upstream score-header orientation before relying on those header fields; retain actual timestamped native inputs/artifact identity, forecast generation/issuance and paired quote states for a genuinely forward comparison. Any deployment of the selected local fresh20+regional identity requires an explicit production cutover and health/inference checks. This audit does not authorize that cutover or a new threshold/shrinkage search. Preserve the existing failed replacement guards, canonical benchmark, financial G0, prior studies and frozen forward dates.
