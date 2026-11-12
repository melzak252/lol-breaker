# Project report

LoL-Breaker combines League of Legends sports probabilities, tournament/prop research and a local
odds-analysis application. It records manual betting decisions; **it does not place bets automatically**.
This report replays saved evidence, not training, forecast issuance or live execution.
All results are **historical development**, with no new adoption or profitability certification.

## Separate evidence domains

| Domain | Population | Period | Headline weighting / uncertainty |
|---|---|---|---|
| Standalone sports | 3,083 regional-supported / 3,103 canonical series | 2026-01-08–2026-09-08 | Equal series; existing frozen guard decision retained |
| Market prediction | 400 fixtures / 1,496 fixture-book cells / 51,208 cases | Scheduled-start months May–October 2026 | Mean quote loss within book; equal books within fixture; equal fixtures; 6 monthly blocks |
| Glicko-2 controls | 10,578 identical series, all four predictions present | 2024-01-01–2026-06-14 | Equal series; 30 match-date monthly blocks |
| Financial policies | Independently selected first-observed tickets, counts below | Mapped target dates 2021-01-09–2026-09-07 | Equal unit stakes; all 70 preselection UTC months January 2021–October 2026 |
| Production interval wallets | 482 source fixtures / 100,949 quotes; 12 fixed portfolios | Fixture starts 2026-05-29–2026-10-22 | Saved-stake ROI and conditional price proxies; all 6 start-month blocks, including zero-action months |

Do not rank C20R against ratings or market numbers from a different population.
The multi-year focal-book finance study is **not** the independent other-book market panel.
Financial portfolios overlap; their ticket counts and returns must not be summed.
Production wallets reuse the production market source; these are different estimands, not
independent replication. Their export does not include later production captures.

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

## Production-collected interval wallets

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

**True CLV remains null**, status `not_available_no_certified_close_or_fill`.
Roughly two-hour collection cannot authenticate actual start, the complete market close
or executable fills. The sealed entry-reference panel is joined to each selected snapshot,
fixture, book, aware entry/start clocks and actual selected-side price.
Side-B results and prices are not flipped a second time.

The default later reference allows at most 300 seconds before conservative recorded start;
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
to whole portfolios. The [README](../README.md#production-closing-price-proxies) shows the default
five-minute proxy by entry interval; JSON includes all twelve cases, three age limits, equal-ticket
and stake-weighted metrics, absences and complete interval statuses.

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

Retained in this report and JSON, omitted only from the README's headline table:
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

**Authenticated closing CLV is null; authenticated closes and executable fills are unavailable.**
Literal snapshot status: `not_available_no_certified_close_or_fill`.
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

The headline price table and the following EV/return comparison use the **same matched-known tickets**:

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

## Reproduction and verification

Use the [reproduction guide](REPRODUCING.md) and [aggregate snapshot](results/project_summary.json).
The saved-ledger CLI enforces exclusive output directories and fails missing/mismatched inputs;
it never substitutes the checked-in snapshot. Model/financial bindings and the consensus/ratings
receipts remain unchanged. Current consumed numerical source hashes are recorded separately.
No raw tickets, datasets or weights are published with the aggregate JSON.

Executed locally: **66 focused arithmetic tests passed**, one existing Starlette/httpx warning;
complete **v3** CLI replay passed. Twelve fixed portfolios / 330 saved tickets reproduce the
bound counts, stakes, payouts and ROI. A separate saved-data oracle agrees across **2,603
comparisons (1,935 numeric)**, maximum absolute error **1.14e-13**, covering wallet aggregates,
conditional same-book/LOO formulas, coverage and monthly intervals.
All five prior numeric evidence domains remain exactly unchanged, including the existing
528 consensus reference comparisons and 12 ratings checks. The checked-in JSON is byte-identical
to the generated summary.

Pre-publication source-join/Decimal-type and empty-reference schema errors were corrected in
replay/verifier code only, without changing frozen data, policies, models or tolerances.
The oracle's original source-identity failure receipt is retained locally, not counted as a pass.
These are bounded saved-evidence checks, not independent neural internals, complete raw-price
consensus reconstruction, source authentication, training replay or a full application suite.

Raw-data acquisition, causal feature construction, original fitting and forecast generation require
additional original artifacts. A clean clone does not supply these ignored/local inputs.
No live operation, deployment, collection, model change, commit or push is part of this work.
