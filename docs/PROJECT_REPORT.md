# Project report

LoL-Breaker develops player-based pre-match League of Legends series probabilities and evaluates
whether they add information beyond bookmaker prices. Market-assisted forecasts, tournament/prop
research and the manual betting application have separate scopes. No automatic bets are placed.

## Current conclusion

The model has historical predictive signal, but **general bookmaker superiority and live betting
profitability are unestablished**. Earlier-price point advantages do not establish superiority to
later markets. Fixed market shrinkage helps point scores on one matched panel, but its primary
improvement intervals include zero. The standalone replacement study still fails BO5/LPL guards.
All assessed years are development-exposed; this report changes neither that status nor deployment.

## Three market datasets

| Source | Retained source | Model-comparison fixtures | Role |
|---|---|---|---|
| OddsPortal, 2021–2026 | 13,473 events; 125,322 endpoint quote rows | 9,132 first / 9,805 last | Annual year-eligible sports model versus the selected focal book's fair price |
| Pinnacle, 2026 | 2,182 fixtures / 4,364 endpoints | 1,040 first / 993 last | Canonical-oriented C20R versus Pinnacle at that endpoint |
| Production-collected Polish prices | 482 model-scorable fixtures / 100,949 records | 400 with 51,208 fresh entry-LOO records | Repeated recorded snapshots, independent other-book comparison and opportunity diagnostics |

Source periods differ from usable model periods. OddsPortal mapped fixtures span
2021-01-09–2026-09-07. Pinnacle source fixtures span
2026-01-14–2026-09-28; model-paired fixtures span
2026-01-17–2026-09-07. The Polish bound analysis spans fixture starts
2026-05-29–2026-10-22, not later captures or today.
The 482 Polish fixtures are not a claim that every fixture in the original production collection
has a usable C20R/market pair. The underlying GOL.GG sports corpus and standalone benchmark are
separate from these three market datasets.

The datasets may share fixtures/provider lineage. Do not sum them as independent replication,
compare absolute scores across cohorts as a model ranking, or treat quote counts as match counts.

### Coverage accounting

- OddsPortal: 13,338 source events have retained endpoint quotes;
  135 retained source events have none.
  1,708 source events lack canonical mapping. The quoted
  ledger has 11,659 distinct mapped fixtures before quality admission.
- OddsPortal's first / last known timed records are 42,435 / 41,439, not independent matches.
  The quality protocol takes one earliest eligible focal-book observation per fixture/endpoint;
  this yields 9,132 / 9,805 scored fixtures. It does **not** average all bookmaker losses.
- Pinnacle retains all 2,182 source fixtures. Missing identities, labels, prior rosters and invalid
  endpoint timing remain explicit. Eighty-eight last endpoints fail conservative prestart admission;
  not all were otherwise model-scorable. Main paired counts are 1,040 / 993, not the raw inventory.
- Polish entry LOO preserves the full 100,949-case accounting: 51,208 matched / 49,741 excluded,
  over 400 / 482 fixtures. The detailed reasons appear in the market-consensus section below.

## Matched comparison rules

LL is unclipped binary log loss; Brier is `(p-y)^2`. All contrasts use identical fixtures,
labels, orientations and endpoint admission. Raw sports forecasts contain no odds. Later prices
are assessment references, never inputs to earlier shrinkage or bet qualification.

OddsPortal uses the focal book's multiplicative no-vig probability. Pinnacle uses its own
multiplicative no-vig probability. Polish production LOO excludes the selected book and requires
at least two other fresh books; it averages fair logits, not fair probabilities. These benchmarks
are not silently substituted for one another.

First observed is not certified market opening; last retained is not necessarily the actual final
quote. Endpoint populations differ. A causal claim about market learning or optimal timing would
require the exact common fixture intersection and explicit input-availability treatment.

## Multi-year model versus focal bookmakers

Each saved endpoint is first filtered for timed-price eligibility and available p/q, then sorted
by quote clock, bookmaker and source event, taking one observation per canonical fixture. Known
binary labels are scored afterward. This preserves the original saved quality contract, including
the choice of the earliest eligible book among last-retained book endpoints. It is not a synthetic
multi-book consensus or a claim of the last observable market price.

| Endpoint | Fixtures | Sports LL / Brier | Focal-book LL / Brier | Sports−market ΔLL, 95% CI |
|---|---:|---:|---:|---|
| First observed | 9,132 | 0.583438 / 0.199523 | 0.586991 / 0.200994 | -0.003554 [-0.008530, +0.001372] |
| Last retained | 9,805 | 0.582592 / 0.199201 | 0.577945 / 0.197117 | +0.004647 [+0.000931, +0.008507] |

Year-eligible annual fresh20+regional models are used for 2021–2026, not a single 2026 checkpoint
applied to its training history. Prior exact-context rosters and daily releases remain historical
availability scenarios, not authenticated issuance.

The paired ratio bootstrap retains all 70 preselection months, January 2021–October 2026:
5,000 draws, seed 20261111. Equal-active-month point differences are
-0.011198 first and
-0.008130 last;
those are different estimands from the primary equal-fixture result. The first main interval
includes zero; the last nominal equal-fixture interval favors the market. Neither is prospective
confirmation or corrected for the entire adaptive research program.

## Direct Pinnacle comparison

| Endpoint | Fixtures | C20R LL / Brier | Pinnacle LL / Brier | ΔLL, nominal 95% CI | ΔLL, four-contrast 98.75% CI |
|---|---:|---:|---:|---|---|
| First observed | 1,040 | 0.558802 / 0.189400 | 0.578937 / 0.197194 | -0.020135 [-0.037267, -0.004424] | -0.020135 [-0.042401, +0.001336] |
| Last retained | 993 | 0.560559 / 0.190288 | 0.548674 / 0.184349 | +0.011885 [-0.010596, +0.027691] | +0.011885 [-0.017119, +0.031245] |

The first endpoint has a model point advantage, but its adjusted interval and equal-month
confirmation do not establish a general edge. Pinnacle has the lower last-endpoint point loss.
The saved decision remains **NO_GENERAL_PINNACLE_EDGE_ESTABLISHED**.

Blocks are the original **Pinnacle scheduled-start UTC months**, not GOL.GG canonical dates or
conservative earliest-start timestamps. Two fixtures cross a month boundary between those fields.
The initial replay validation wrongly required equality to canonical-date months; that verifier
assumption was corrected. Original cohorts, source labels, probabilities, gates, tolerances and
saved intervals were not changed. There are nine blocks, 5,000 draws, seed 20261112.

### Polish prices versus Pinnacle

Only Fortuna and Superbet from the same provider are represented, with an equal arithmetic mean
of their individually de-vigged probabilities. Source updates must precede the Pinnacle cutoff;
this source's primary comparison has no age cap. It is not the fresh logit-LOO production panel.

| Endpoint | Paired fixtures | Polish consensus LL | Pinnacle LL | Polish−Pinnacle ΔLL, nominal 95% CI |
|---|---:|---:|---:|---|
| First observed | 123 | 0.623161 | 0.631456 | -0.008295 [-0.011061, +0.022959] |
| Last retained | 112 | 0.570379 | 0.572639 | -0.002260 [-0.004293, -0.001818] |

The separately scored three-way intersection is smaller:

| Endpoint | Identical fixtures | C20R LL | Pinnacle LL | Fortuna/Superbet LL |
|---|---:|---:|---:|---:|
| First observed | 100 | 0.672136 | 0.658414 | 0.648693 |
| Last retained | 89 | 0.636062 | 0.596665 | 0.592267 |

Only two months support these Polish comparisons. They do not establish the quality of the whole
Polish market, a UK-versus-Poland price equivalence, or general superiority from a narrow overlap.

## Standalone sports model

**C20R** names the configured `c0-native-fresh20-2026-10-21-v1-regional-v1` artifact: a three-seed fresh20
player-history ensemble plus prior regional context, with 19 history input slots and 1 learned token.
Its recipe uses causal 90/180/365-day player/map summaries and optional organization context.
It outputs sports-only whole-series BO1/3/5 probabilities. This display name changes no runtime identity;
[model catalog](MODEL_CATALOG.md#main-model-and-local-policy) and [serving config](../conf/base/c0_serving.json).

The frozen split has 36,205 pre-July-2025 TRAIN rows, 1,328 2025H2 CAL rows and
3,103 development-exposed 2026 ASSESS rows. Regional coverage is
99.36%; 20 absences are not imputed.

| Model | Cohort | N | Log loss | Brier | ECE-10 |
|---|---|---:|---:|---:|---:|
| Historical C0 | Full assessment | 3,103 | 0.535449 | 0.181034 | 2.29% |
| Fresh20, no regional | Full assessment | 3,103 | 0.533925 | 0.180376 | 2.08% |
| Annual A0 | Full assessment | 3,103 | 0.536757 | 0.181573 | 2.20% |
| Historical C0 | Common supported | 3,083 | 0.536947 | 0.181576 | 2.33% |
| Fresh20, no regional | Common supported | 3,083 | 0.535391 | 0.180906 | 2.12% |
| Annual A0 | Common supported | 3,083 | 0.538211 | 0.182107 | 2.26% |
| C20R | Common supported | 3,083 | 0.533023 | 0.180030 | 1.93% |

The [four-cell factorial audit](../reports/c0_regional_factorial_audit_20261022.md) favors C20R on its
combined paired comparison, but required **BO5 and LPL replacement guards fail** against C0 and A0
at the +0.005 upper-bound margin. The study remains **not adopted**. Local configuration is not a
new research decision, deployment claim, independent holdout or tournament/financial certification.

## Market consensus and fixed shrinkage

The sealed panel fixes C20R's model identity and every focal-case input. It preserves the original
LOO convention: at least two distinct **other** books, equal-weight fair logits, contributor clocks
no later than the focal clock and ages at most 300 seconds. JSON-encoded contributors, clocks,
weights, ages and fair probabilities are decoded and checked; the panel is not rebuilt with an
arithmetic probability-consensus function.

Exactly three recipes are scored: C20R, other-book consensus and fixed Hybrid-50.
`p_H = expit(0.5*logit(p_C20R) + 0.5*logit(p_consensus))`; no alpha selection or fitting.
LL uses epsilon=0, without new clipping. Brier is `(p-y)^2`.
Losses—not probabilities—are averaged within fixture/book, then over books and fixtures.

| Recipe | Log loss | Brier | ΔLL vs consensus, nominal 95% CI |
|---|---:|---:|---|
| C20R | 0.585025 | 0.202815 | +0.004727 [-0.019232, +0.019035] |
| Other-book consensus | 0.580298 | 0.199347 | reference |
| Hybrid-50 | 0.576162 | 0.198601 | -0.004136 [-0.016001, +0.002928] |

Paired contrasts use exact fixture-ID joins and identical scheduled-start month labels;
all per-fixture differences must be finite before resampling.
Both estimands use 5,000 draws, seed `20261023`,
and nominal 95% intervals, **not wider adaptive-search correction**.

| Candidate−reference | Metric | Fixture-weighted Δ, 95% CI | Equal-month Δ, 95% CI |
|---|---|---|---|
| C20R−consensus | log_loss | +0.004727 [-0.019232, +0.019035] | -0.010149 [-0.025546, +0.007041] |
| C20R−consensus | brier | +0.003468 [-0.007654, +0.009749] | -0.003830 [-0.011454, +0.004023] |
| Hybrid−consensus | log_loss | -0.004136 [-0.016001, +0.002928] | -0.010217 [-0.017312, -0.002116] |
| Hybrid−consensus | brier | -0.000746 [-0.005893, +0.002225] | -0.003884 [-0.007247, -0.000241] |
| Hybrid−C20R | log_loss | -0.008863 [-0.016018, +0.003657] | -0.000068 [-0.009217, +0.009097] |
| Hybrid−C20R | brier | -0.004214 [-0.007553, +0.001853] | -0.000055 [-0.004329, +0.004260] |

Hybrid has lower point loss, but its main fixture-weighted intervals versus consensus and C20R
include zero. The equal-month Hybrid−consensus interval is negative; that different estimand
does not override the primary result or establish adoption.

### Consensus coverage

Complete focal accounting: **100,949 cases / 482 fixtures**.
Exactly 51,208 matched identities join one-to-one to the panel;
49,741 remain excluded with these reasons:

| Excluded reason | Cases |
|---|---:|
| `fewer_than_two_fresh_eligible_other_books` | 23,721 |
| `no_fresh_eligible_other_book_state` | 23,733 |
| `no_prior_other_book_state` | 2,287 |

Quote cases are not independent matches. Successful coverage is 1,496 fixture-book cells /
400 fixtures; excluded cases are not scored as zero loss.

### Forecast scores by time to scheduled start

| Lead hours | Fixture N | Consensus LL | C20R LL | Hybrid-50 LL |
|---|---:|---:|---:|---:|
| [0,2) | 291 | 0.591644 | 0.590224 | 0.585922 |
| [2,6) | 325 | 0.582057 | 0.582640 | 0.576787 |
| [6,12) | 328 | 0.576932 | 0.578070 | 0.571809 |
| [12,24) | 304 | 0.568443 | 0.572756 | 0.564460 |
| [24,48) | 249 | 0.575792 | 0.594537 | 0.578886 |
| [48,96) | 230 | 0.575465 | 0.586390 | 0.574458 |
| [96,inf) | 169 | 0.563134 | 0.565161 | 0.557140 |

All seven saved bands use the same within-profile weighting. Fixture cohorts overlap across bins;
counts are not additive. Scheduled-start leads do not establish actual-start accuracy, execution
or an optimal forecast/betting time. Per-band LL/Brier contrasts and equal-month intervals are in
[the aggregate snapshot](results/project_summary.json).

## Snapshot opportunities and unit returns

The exact saved sample contains 12,000 unique recorded fixture/timestamps across
471 fixtures: 2,000 without-replacement checks per lead-time band, seed 20261113.
Sampling is not conditioned on a future closing reference. The replay applies the unchanged zero-tax
baseline gates to offers recorded exactly at each sampled clock, then chooses the highest-EV
passing offer, with the original odds/book/side tie-breaks. It is not a complete carried-forward board.
Rating-disagreement and contemporaneous drift remain NOT_ASSESSED.

The earlier fixed portfolio is the original first eligible offer per fixture in [0,48) hours,
not the post-hoc earliest suggestion among sampled checks. The two views intentionally differ:

| View | Observations | Distinct fixtures | Mean expected EV | Realized unit yield, 95% CI |
|---|---:|---:|---:|---|
| First eligible [0,48)h | 82 | 82 | +10.54% | +11.76% [-6.63%, +42.41%] |
| Time-stratified snapshot checks | 451 | 74 | +11.01% | -4.99% [-21.96%, +17.69%] |

The snapshot selection has 451 suggestions / 12,000 checks
(3.76%), with 11,549 abstentions.
Its 451 known observations have 151 wins versus
184.92 expected from Hybrid-50 and 210.39
from raw sports probabilities. These are repeated observations on 74 outcomes,
not independent Bernoulli trials. The most repeated fixture contributes
61 observations; it is not discarded because of its result.

Zero-tax unit return is `y_selected*entry_odds-1`. Both views resample all six fixture-start months,
5,000 draws, seed 20261113; snapshots of a fixture stay in its month. Intervals measure uncertainty
under these historical weights, not new independent evidence or an estimate certified as true edge.
The positive fixed-policy result does not supersede the negative repeated-check result.

### Profiles by time before recorded start

Every band has 2,000 sampled checks. Fixture populations overlap across bands; do not add their
unique-fixture counts or choose a favorable band after observing outcomes. Per-band intervals,
medians, quantiles and missing-reference counts are in the aggregate snapshot.

| Lead | Checks | Suggestions / fixtures | Mean EV | Unit yield | Same-book raw mean (covered) | Fair LOO mean (covered) |
|---|---:|---:|---:|---:|---:|---:|
| 0–2h | 2,000 | 78 / 30 | +11.31% | +28.92% | +0.22% (68) | -3.72% (78) |
| 2–6h | 2,000 | 80 / 30 | +10.81% | +23.00% | +3.50% (72) | -2.74% (80) |
| 6–12h | 2,000 | 74 / 35 | +10.93% | -9.59% | +6.01% (70) | -1.78% (74) |
| 12–24h | 2,000 | 78 / 30 | +10.39% | -52.72% | +7.76% (75) | +0.15% (78) |
| 24–48h | 2,000 | 71 / 26 | +12.09% | -16.39% | +7.93% (66) | +2.31% (71) |
| 48h+ | 2,000 | 70 / 21 | +10.61% | -5.13% | +13.05% (70) | +4.69% (70) |

## Recorded-close CLV

The current reporting convention is **latest recorded strictly pre-start state per fixture/book,
without an age cap**. It does not require a closing quote strictly later than entry. Same entry
and closing state is allowed and yields zero raw same-book movement if the price is unchanged.
An invalid latest state stays missing; there is no fallback to an older favorable state.

- Same-book raw price CLV: `entry_odds / samebook_recorded_close_selected_odds - 1`.
- Fair other-book CLV: `entry_odds * q_closing_selected - 1`.
- For each peer, `q_A = odds_B/(odds_A+odds_B)`; consensus is expit of the equal-weight mean fair
  logits, excludes the focal book and needs at least two other eligible books.
- A peer's latest pre-start quote may predate entry. Its age remains visible, rather than imposing
  the earlier hidden requirement that at least one peer update be later than entry.
- Price CLV requires neither a known result nor an executed fill. Execution evidence is separate.
  The actual final available market quote and executable prices are not authenticated.
- Entry-time consensus still requires freshness; removing the **closing** age cap does not relax
  entry admission or feed future closing prices into prediction/selection.

| View | Same-book raw mean (covered) | Other-book fair mean, 95% CI (covered) | Fair median |
|---|---:|---|---:|
| First eligible [0,48)h | +8.32% (78/82) | +2.17% [-0.33%, +6.17%] (82/82) | +1.08% |
| Time-stratified snapshot checks | +6.43% (421/451) | -0.30% [-1.46%, +1.08%] (451/451) | -2.12% |

The literal latest-only convention covers all 451 sampled fair references.
The preceding exploratory result covered 440 because it additionally required a later peer update;
its −0.2143% mean is not relabeled as the new -0.30% mean.
No model, gate or selected event changed. Own-book coverage can remain missing even when two valid
other books support a fair closing reference.

### Price direction and medians

| View | Same-book shortened / tied / lengthened | Raw median | Fair positive / zero / negative | Fair median |
|---|---:|---:|---:|---:|
| First eligible [0,48)h | 49 / 18 / 11 | +6.59% | 48 / 0 / 34 | +1.08% |
| Time-stratified checks | 238 / 132 / 51 | +2.22% | 164 / 0 / 287 | -2.12% |

Positive raw CLV means the same book's selected-side odds shortened between entry and recorded
close. It does not assert monotonic movement at every intermediate observation. Other-book fair
CLV is a different benchmark, and neither closing consensus nor a Gaussian-looking disagreement
distribution reveals true win probability.

### Closing reference ages

Age is time **before recorded start**, not time since entry. LOO age statistics use the oldest
eligible contributor per selected observation, making the old-reference fraction explicit.

| View | Reference | Covered | Median age | p95 age | Maximum age | Age at least 2h |
|---|---|---:|---:|---:|---:|---:|
| First eligible [0,48)h | Same book | 78 | 0.24h | 8.67h | 41.07h | 14 |
| First eligible [0,48)h | Other books, oldest contributor | 82 | 1.08h | 56.15h | 160.25h | 34 |
| Time-stratified checks | Same book | 421 | 0.24h | 6.08h | 82.07h | 46 |
| Time-stratified checks | Other books, oldest contributor | 451 | 1.08h | 57.08h | 160.25h | 205 |

Quantiles are distributions of covered observations, not confidence intervals. Mean intervals use
all six fixture-start blocks; undefined denominator draws remain explicit and withhold the CI.
All means/medians are conditional on their own covered subset; missing references never become zero.

## Production-collected interval wallets

Retained secondary wallet evidence: these are the existing twelve saved portfolios, not new money simulations.
The primary zero-tax unit-return and latest-recorded CLV results are reported separately above.

The bound export contains **482 fixtures / 100,949 Polish-bookmaker quotes**, with fixture starts
from **2026-05-29 05:00 UTC to 2026-10-22 12:00 UTC**. Entry LOO eligibility retains
51,208 quotes / 400 fixtures. Start-month fixture counts are May 14, June 46, July 78,
August 232, September 94 and October 18; the edge months are partial.
These are **saved counterfactual C20R probabilities and simulated tickets**, not authenticated
forecast issuance, recorded bets, actual UK prices or data through today.

### Frozen selection and wallet assumptions

Exactly twelve existing cases are read: `logit` / `baseline`, tax 0 or 0.12, and
`h0_2`, `h2_6`, `h6_12`, `h12_24`, `h24_48`, `primary48`.
Each half-open interval uses its own first eligible recorded observation per fixture,
not a future-best quote, exact two-hour grid or post-score choice of the best interval.
The full [0,48) wallet overlaps the five narrower wallets; the tax scenarios also overlap.
Do not add ticket counts, profits or bootstrap evidence across them.

The sports share is fixed at 0.50 in logit space. Entry consensus requires at least two other
books, each no more than 300 seconds old at the entry clock. The frozen baseline retains
EV≥5%, short-odds dynamic hurdle (cutoff 2.2, penalty scale 0.08), DRO ε=0.025,
25% global/longshot EV caps, longshot cutoff 3.5, odds≤4, Tier-1 positive market gap≤8pp,
BO1 absolute gap≤12pp and unshrunk sports/market logit gap≤0.85.
Identity, non-live, pre-start and input-availability rules are not disabled.
Rating disagreement and contemporaneous negative CLV drift remain **NOT_ASSESSED**,
not successful complete-application safeguards.

Each portfolio starts with PLN10,000, quarter Kelly, minimum PLN2, maximum PLN150,
and stakes reserved until recorded result availability. Tax scenarios requalify and restake
independently on the same Polish prices. The archived `max_portfolio_exposure=0.25`
means a **per-new-ticket free-cash cap, not a 25% aggregate exposure limit**.
This replay validates retained stakes and Decimal payouts; it does not substitute today's
wallet engine or resimulate capital paths under changed risk rules.

### Results by entry lead

All 330 retained ticket rows across the twelve portfolios are settled; these are repeated
hypothetical selections, not 330 independent actual bets. ROI is P&L/stake turnover.

| Tax | Lead hours | Wins / bets | Turnover, PLN | P&L, PLN | ROI | Nominal 95% ROI interval |
|---|---|---:|---:|---:|---:|---|
| 0% | [0,2) | 12/31 | 4,019.13 | +83.50 | +2.08% | [-26.04%, +46.37%] |
| 0% | [2,6) | 18/39 | 5,227.40 | +1,289.76 | +24.67% | [-23.74%, +38.54%] |
| 0% | [6,12) | 21/49 | 6,700.40 | +962.59 | +14.37% | [-2.18%, +90.22%] |
| 0% | [12,24) | 16/44 | 6,124.41 | -515.87 | -8.42% | withheld (1/5,000 undefined) |
| 0% | [24,48) | 16/42 | 5,630.38 | +0.87 | +0.02% | [-22.71%, +38.01%] |
| 0% | Full [0,48) | 36/82 | 11,227.02 | +1,728.80 | +15.40% | [-4.22%, +48.59%] |
| 12% | [0,2) | 1/3 | 360.68 | +50.20 | +13.92% | withheld (92/5,000 undefined) |
| 12% | [2,6) | 2/5 | 605.16 | +74.10 | +12.24% | withheld (458/5,000 undefined) |
| 12% | [6,12) | 2/5 | 662.05 | +73.11 | +11.04% | withheld (77/5,000 undefined) |
| 12% | [12,24) | 0/8 | 1,055.75 | -1,055.75 | -100.00% | withheld (92/5,000 undefined) |
| 12% | [24,48) | 1/6 | 852.42 | -523.74 | -61.44% | withheld (444/5,000 undefined) |
| 12% | Full [0,48) | 5/16 | 2,154.85 | -397.69 | -18.46% | withheld (8/5,000 undefined) |

Intervals resample all six fixture-start UTC months, including zero-action months:
5,000 draws, seed `20261023`, NumPy's default integer dtype. A zero-denominator draw is
retained and withholds the interval; it is not deleted or converted to zero ROI.
These are retained-stake ratios, not new wallet-path simulations, and are not corrected
for the full historical timing/configuration search. Positive point ROI does not establish an edge.

### Production recorded-price references

These historical sensitivity rows retain the older strict-later and reference-age filters;
they are not the primary latest-recorded closing convention above. Executed fills were not recorded.
Price CLV does not require a fill; a captured quote is not an authenticated final market close. The sealed entry-reference panel is joined to each selected snapshot,
fixture, book, aware entry/start clocks and actual selected-side price.
Side-B results and prices are not flipped a second time.

This older sensitivity family allows at most 300 seconds before conservative recorded start;
60-minute and uncapped-age results below are explicitly separate sensitivities.
The latest recorded book state is used, not an outcome-picked state. Same-book prices must
be strictly later than entry and before start. LOO references need at least two distinct
other books, all contributor clocks before start and within the declared age limit, with
at least one contributor strictly after entry. Invalid latest states are not revived
by falling back to an older valid quote.

For actual selected-side odds `O` and later fair probability `q`:

- Raw same-book movement: `O / later_samebook_selected_odds - 1`.
- Gross other-book proxy: `O*q - 1`.
- Net other-book proxy: `(1-tax)*O*q - 1`.

The new replay decodes and validates reference contributor identities, clocks and ages,
then recomputes selected-side price formulas from the sealed fair-q panel.
It does **not** independently reconstruct every contributor's raw-price mean-logit consensus.
Closing references are assessment-only, not the entry-time shrinkage input or a timing oracle.

Full [0,48) portfolio sensitivity; every mean is conditional on its own covered subset:

| Tax | Maximum age before start | Same-book pairs / bets | Raw move | LOO pairs / bets | Gross LOO proxy | Net LOO proxy | Net LOO nominal 95% CI |
|---|---|---:|---:|---:|---:|---:|---|
| 0% | ≤5 min | 17/82 | +3.77% | 19/82 | +1.82% | +1.82% | withheld (8/5,000 undefined) |
| 0% | ≤60 min | 41/82 | +7.16% | 47/82 | +0.60% | +0.60% | [-0.99%, +9.04%] |
| 0% | Uncapped age | 74/82 | +8.77% | 80/82 | +2.30% | +2.30% | [-0.30%, +7.13%] |
| 12% | ≤5 min | 2/16 | +5.30% | 4/16 | +12.36% | -1.12% | withheld (458/5,000 undefined) |
| 12% | ≤60 min | 5/16 | +3.78% | 8/16 | +10.29% | -2.95% | withheld (458/5,000 undefined) |
| 12% | Uncapped age | 10/16 | +11.39% | 15/16 | +10.69% | -2.59% | withheld (8/5,000 undefined) |

Raw same-book movement is tax-independent; different tax rows select different tickets.
Gross fair-price movement must not be presented as a net tax-adjusted edge.
Uncapped age increases coverage by admitting stale references, not by recovering a certified close.
Missing references retain reasons and never become zero CLV; covered estimates are not extrapolated
to whole portfolios. JSON retains all twelve cases, three age limits, equal-ticket and
stake-weighted metrics, absences and interval statuses. These are legacy sensitivity definitions,
not the README's latest-recorded CLV convention.

## Historical financial policies

Annual year-eligible fresh20+regional probabilities, OddsPortal 2021–2026 first-observed endpoints.
The focal bookmaker's multiplicative no-vig q supplies the same-book 50/50 logit reference;
this is **not** the application's two-other-book hybrid or complete gate stack.
Selection uses EV≥5%, DRO ε=0.025 (a 2.5pp probability stress haircut), odds≤4 and the first
eligible retained quote, at most once per canonical fixture. Stakes are fixed at one unit.
Tax is a fraction and a winning ticket returns `O*(1-tax)-1`; a loss returns −1.

| Portfolio | Tax | Known / selected | Mean EV | Yield, 95% CI | EV−yield, 95% CI (pp) |
|---|---:|---:|---:|---|---|
| Sports only | 0% | 4,482 / 4,484 | 19.7235% | 6.0036% [+2.5435%, +9.3413%] | +13.7199 [+10.3584, +17.2568] |
| Focal-book 50/50 | 0% | 1,967 / 1,968 | 12.8988% | 13.6894% [+8.4207%, +18.9447%] | -0.7906 [-6.0611, +4.4008] |
| Focal-book 50/50, fixed original tickets | 12% | 1,967 / 1,968 | -0.6491% | 0.0466% [-4.5898%, +4.6714%] | -0.6957 [-5.3338, +3.8727] |
| Focal-book 50/50, requalified | 12% | 315 / 315 | 13.6441% | 22.8061% [+9.3066%, +36.4237%] | -9.1620 [-22.9312, +4.2959] |

The 0% sports / 0% shrinkage / 12% requalified policies have 2 / 1 / 0
unresolved selected outcomes. They remain in selected counts; only known binary labels with
`label_status=known` contribute to EV, expected wins, odds and return comparisons.
Unknown labels leave whole-portfolio P&L and ROI null, not zero.
The requalified 12% policy changes selections; its result is not a tax-only or shrinkage effect.

**Estimand.** Yield is profit per unit of known-label stake turnover, not bankroll growth.
All 70 original preselection UTC months, January 2021–October 2026, are resampled,
including zero-action months: 5,000 draws, seed `20261111`.
JSON retains complete return/error intervals, active-block counts, equal-active-month scope,
empty/valid replicate counts and statuses. Undefined bootstrap draws withhold the interval.
Neither nominal intervals nor archived source clocks establish executable prices, book activity,
fills or an independently confirmed edge.

### Same-ticket shrinkage diagnostic

On the **same 1,967 known zero-tax shrinkage tickets**, selected-side sports-only mean EV is
**33.5068%**, shrunk EV **12.8988%**, observed yield **13.6894%**.
Sports-only probability is derived from `p_c0` and the actual selected side, without reselection.
This is a forecast-error comparison; comparing the separately selected sports portfolio would
change the question. EV−yield is not an identified optimizer's-curse component.

### Fixed-original-ticket tax audit

Retained in this report and JSON, not part of the README's prediction-quality headline:
apply 12% tax to the original 1,968 zero-tax shrinkage tickets, keeping books, sides and entries.
Known outcomes: 1,967; unresolved: 1. Known-label profit is
**+0.9176 units**, yield **0.04665%**, 95% interval
**[-4.5898%, +4.6714%]**. Whole-portfolio return remains undefined.
This fixed-ticket audit must not be confused with the independently requalified 315-ticket policy.

### Near-band reliability

The band is strictly **(5%,10%] expected EV** on the identical known-label ticket mask.

| Portfolio | Known near-band tickets | Mean EV | Realized yield | EV−yield, 95% CI (pp) |
|---|---:|---:|---:|---|
| Sports only, 0% tax | 1,434 | 7.3445% | -0.0760% | +7.4205 [+3.1435, +11.4203] |
| Focal-book 50/50, 0% tax | 932 | 7.4796% | +6.5150% | +0.9646 [-8.0188, +9.4541] |
| Focal-book 50/50, requalified, 12% tax | 155 | 7.6171% | +25.8911% | -18.2740 [-36.4301, -0.5759] |

Neither shrinkage regime establishes the project's ±2pp containment target. Zero-tax intervals
are too wide; the taxed near-band interval is also not contained in that tolerance.
Positive realized yield is not proof that displayed EV is reliable within ±2pp.

## CLV and execution evidence

Retained **OddsPortal strictly-later price diagnostics**, not the primary latest-only closing rule.
Execution evidence is not recorded and is separate from price CLV. These historical pairs keep
their original strict-later/freshness admission; they are not relabeled as the new convention.
Each policy's actual selected entries are joined one-to-one to later endpoint rows using source,
event, canonical fixture and bookmaker, preserving side, entry clock and price. A match requires
the same ordered participants, strictly later quote, both quotes before the earliest recorded
start, valid two-side odds, both sides fresh within 300 seconds, and saved timed-price eligibility.
Naive/malformed clocks and duplicate endpoint identities fail, rather than becoming UTC or
selecting a favorable duplicate. Every unmatched selection retains an explicit reason.

Raw movement is `O_entry/O_later_selected-1`.
The net no-vig proxy is `(1-tax)*O_entry*q_later_selected-1`, with q recomputed from the paired
raw two-side odds. Price movement and no-vig/tax adjustment are distinct quantities.
The 315-ticket taxed selection is reconstructed independently; it never reuses zero-tax price pairs.

| Scenario | Price pairs / selected | Matched known / known selected | Raw move | Net proxy, 95% CI | Median later lead | Within 5 min |
|---|---:|---:|---:|---|---:|---:|
| 0% focal-book 50/50 | 1,709 / 1,968 | 1,708 / 1,967 | +12.6778% | +4.8198% [+3.5268%, +6.0323%] | 1.7064h | 111 |
| 12% requalified | 268 / 315 | 268 / 315 | +24.6815% | +2.0908% [-1.7274%, +5.9939%] | 1.4838h | 20 |

The price table above and the following EV/return comparison use the **same matched-known tickets**:

| Scenario | N | Mean expected EV | Realized yield | EV−proxy (pp) | Proxy−yield (pp) |
|---|---:|---:|---:|---:|---:|
| 0% focal-book 50/50 | 1,708 | 12.9732% | 14.3677% | +8.1534 | -9.5479 |
| 12% requalified | 268 | 14.1792% | 20.1167% | +12.0883 | -18.0259 |

These are not full-portfolio EV/yield values. Return minus later-price proxy is not identified
pure sporting luck; a later price is not true probability or guaranteed edge.
The complete 70-month finance bootstrap supplies unit-stake ratio intervals; JSON also
exposes active-month-only resampling and equal-active-month estimates separately.

| Scenario | Unmatched reason | Tickets |
|---|---|---:|
| 0% focal-book 50/50 | `at_or_after_earliest_known_start` | 6 |
| 0% focal-book 50/50 | `later_quote_fails_timed_price_rules` | 145 |
| 0% focal-book 50/50 | `same_clock_not_later` | 108 |
| 12% requalified | `at_or_after_earliest_known_start` | 2 |
| 12% requalified | `later_quote_fails_timed_price_rules` | 28 |
| 12% requalified | `same_clock_not_later` | 17 |

### Separate application-policy counterweight

The saved app-style factorial ASSESS wallets were 2 / +173.97 PLN first-ready,
8 / −304.10 PLN quote-triggered, 2 / −269.77 PLN UTC-even and 6 / −64.62 PLN UTC-odd.
These are prior reported sparse scenarios, **not recomputed by this aggregate replay**;
no primary configuration was selected because TRAIN support failed. They are neither additive
nor a timing recommendation. See [the research record](RESEARCH.md).

## Player versus organization Glicko-2

These are **ratings-only corrected day-batch controls**, not the W20-restricted ledger, canonical
player-only benchmark or a competition-family bridge substituted for organizations.
All 10,578 unique series have binary labels, valid BO1/3/5, dates, distinct team IDs and
all four saved probabilities. Invalid/missing one-sided predictions fail; separate convenient
model-specific intersections are not selected.

| Control | Probability | N | Log loss | Brier |
|---|---|---:|---:|---:|
| Player | Raw | 10,578 | 0.574052 | 0.195839 |
| Organization | Raw | 10,578 | 0.634898 | 0.217141 |
| Player | Prior-year calibrated | 10,578 | 0.574096 | 0.195826 |
| Organization | Prior-year calibrated | 10,578 | 0.607321 | 0.210434 |

Source protocol: `corrected-header-ratings-only-daybatch-20260908`.
The replay freezes the entire day before results, applies actual-date weekly Glicko decay and
updates actual map participants afterward from paired pre-map opponent states. Within-day updates
follow deterministic series-ID/source-map order, not authenticated event clocks. BO2 may update
state but is absent from binary scoring. Both levels share the independent-map series projection;
saved columns are **already BO-projected** and are never transformed or clipped again here.

Player and organization calibrated columns use separate positive zero-intercept slopes fitted
on the preceding calendar year. No slopes are refit by this command. Observed first-map rosters
and day-only clocks are retrospective scenarios, not announcement or issuance evidence.
No W20 or market inputs enter these rating controls.

| Player−organization | Metric | Series-weighted Δ, nominal 95% CI | Equal-month Δ, 95% CI |
|---|---|---|---|
| Raw | log_loss | -0.060845 [-0.079764, -0.047286] | -0.082312 [-0.102220, -0.063310] |
| Raw | brier | -0.021301 [-0.028548, -0.016047] | -0.028969 [-0.035824, -0.022329] |
| Calibrated | log_loss | -0.033225 [-0.045580, -0.024346] | -0.041800 [-0.057010, -0.026406] |
| Calibrated | brier | -0.014608 [-0.020133, -0.010697] | -0.018984 [-0.024785, -0.013274] |

Intervals use 30 match-date UTC months, 5,000 draws, seed `20261111`.
The calibrated comparison favors this player control historically; it is neither a cross-cohort
C20R ranking nor held-out/prospective promotion evidence.

## Reproduction and verification

The [saved-only guide](REPRODUCING.md) lists exact artifacts, source bindings, grouping rules and
commands. The [v4 aggregate](results/project_summary.json) contains full precision and current
source hashes. It is output, never a fallback input. No raw tickets, datasets or weights are published.

Current execution: **115 focused arithmetic tests passed**, one existing Starlette/httpx warning.
The actual CLI completed at `/tmp/c20r-readme-evidence-replay-3` with exclusive output.
The three market domains reproduce 28 annual, 90 Pinnacle and 528 Polish-panel numeric references;
the largest observed discrepancy is 1.11e−16, within the unchanged 1e−12 reference tolerance.
Twelve Glicko checks also reproduce. All **3,188 existing numeric fields** across the six retained
v3 evidence blocks remain identical; only hierarchy/metadata and new evidence domains changed.

The initial new replay failed before output because a Pinnacle block-validation assumption used
canonical dates instead of the original scheduled-start months. The original failure remains in
the session record; no source cohort, model, interval tolerance or frozen study was overwritten.
Existing wallet summaries are recomputed from saved stakes, not new bankroll simulations.

These checks do not independently certify neural internals, original source authenticity, roster
publication, actual starts, live issuance or fills. No training, inference, collection, database
access, deployment, commit or push occurred. Pre-existing unrelated research edits are preserved.
