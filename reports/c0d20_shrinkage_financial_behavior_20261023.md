# C0-D-20 shrinkage: superiority and financial behavior — 2026-10-23

**COMPLETED / SUPERIORITY_NOT_ESTABLISHED / FINANCIAL_EDGE_NOT_ESTABLISHED.** Fixed 50/50 logit shrinkage improves historical probability point scores but does not pass the two-comparator superiority criterion. Under the unchanged financial policy, its primary wallet loses **397.69 PLN on 16 bets, ROI -18.46%**. Model-only loses more money but has less-negative ROI. The slightly profitable alpha-0.75 cell is not a validated winning setting.

Run: `production_shrinkage_finance_20261023_01`.

- Frozen protocol: `data/05_model_input/production_shrinkage_finance/production_shrinkage_finance_20261023_01/protocol.json`.
- Outputs: `data/08_reporting/production_shrinkage_finance/production_shrinkage_finance_20261023_01/{probability,finance,verification}/`.
- Preceding [correlation and fixed-grid study](c0d20_shrinkage_correlation_20261023.md) remains separate and preserved.

## Scope and controls

Only genuine **`c0-native-fresh20-2026-10-21-v1-regional-v1`**, the selected C0-D-20 / fresh20+regional identity. EXP081/tabular predictions remain excluded. Reuses the retained read-only production export from `192.168.1.17` and exact saved counterfactual model probabilities. No new database export, external collection, neural inference, model/calibration fit, production write or deployment.

Alpha is the sports share in `sigmoid(alpha*logit(p_sports)+(1-alpha)*logit(q_market))`. The fixed grid is **0, 0.1, 0.25, 0.5, 0.75, 0.9, 1**; endpoints are exact. Alpha 0.5 is the primary candidate, not a newly selected optimum. Other weights and timings remain descriptive.

The market input excludes the focal bookmaker and uses at least two other distinct books, each latest retained valid state at or before the focal observation and at most 300 seconds old. Invalid latest states block older prices. Market probabilities are proportional no-vig probabilities, combined through the equal-weight mean of fair logits. Canonical series orientation, unchanged rosters/model-source exclusions and one BO projection remain inherited from the preceding audit. Historical publication, participant bindings and actual execution are not authenticated by this comparison.

## Probability superiority

Primary support is **51,208 LOO quotes /400 fixtures**, May–October 2026. Score each quote before averaging within fixture/book, then equal books within fixture and equal fixtures. No probability clipping or filtering by financial success.

Market-only /50:50/model-only LL is **0.580297896 /0.576161822 /0.585025169**; Brier is **0.199347083 /0.198601096 /0.202814930**. The predeclared primary family has two LL contrasts, with 5,000 paired monthly-block draws, seed20261023, nominal two-sided **97.5%** intervals. Negative differences favor the blend.

| Alpha0.5 minus comparator | Delta LL | Six-month-block 97.5% interval | Source-name/year proxy 97.5% upper bound |
|---|---:|---:|---:|
| Market-only LOO consensus | -0.004136 | [-0.016444, +0.003553] | +0.007393 |
| Sports-only | -0.008863 | [-0.016493, +0.004302] | +0.002041 |

Both LL upper bounds must be negative, with Brier point scores nonincreased against both endpoints. The Brier point guards pass, but **neither LL requirement passes**. Equal-month LL favors the blend against market, but is approximately zero against sports; its sports interval crosses zero. The 50 source-name/year LOO proxies are not authenticated independent tournament editions, and their LL intervals also cross zero. Full-focal 482-fixture results are descriptive only; their 67-proxy LL intervals cross zero too.

This does not establish equivalence, inferiority or impossibility. It does establish that this exposed sample cannot support the requested superiority claim under the frozen criterion. The intervals are not adjustment for the complete history of model/alpha development.

## Can earlier results choose a better alpha?

The diagnostic uses May–June as warmup. For each later target month, the parameter origin is its earliest evaluated quote clock, which can fall in the preceding calendar month. Training requires an earlier target month, every included quote strictly before origin, and the fixture's maximum recorded result-known timestamp strictly before origin. Evaluation support is unchanged.

| Evaluation month | Eligible prior training fixtures | Alpha selected from released prior results | Evaluation fixtures |
|---|---:|---:|---:|
| July | 29 | 1.00 | 54 |
| August | 56 | 1.00 | 212 |
| September | 255 | 0.25 | 77 |
| October | 357 | 0.25 | 17 |

Across the same **360 evaluation fixtures**, fixed market/fixed50:50/fixed sports/prior-selected LL is **0.588990082 /0.586001291 /0.596576233 /0.598122439**. Equal-month LL is **0.555025091 /0.548407603 /0.554238459 /0.554420776**. Retuning from prior results is worse pointwise than fixed50:50 here. This is retrospective temporal evaluation, not genuinely unseen confirmation of an alpha already studied on these years.

The first played-month-only implementation failed to enforce result release. It is retired and preserved under `probability/retired_played_month_only/`. The corrected selection excludes 11/38/51/26 unreleased training fixtures at the four origins. No evaluation fixtures, fixed-alpha scores or primary intervals changed. `causal_integrity_receipt.json` and the independent verification retain exact memberships and clocks.

## Financial execution and coverage

Primary policy: first eligible **recorded focal quote** in **[0,48) hours** before the known start bound. Qualify both sides and all recorded candidates at an identical observation clock before deterministic selection; one ticket per fixture. No future maximum EV/price selection. This is not a reconstruction of every board state or a UTC2h scanning policy.

Unchanged settings: 12% tax on the full winning payout, minimum net EV5% plus the existing low-odds hurdle, DRO epsilon0.025, EV ceiling25%, odds ceiling4, sports/market logit bound0.85, Tier1 and BO1 gap guards, quarter Kelly, initial capital10,000 PLN, stake2–150 PLN. The unshrunk sports probability drives the logit gate for alpha>0. Alpha0 is genuinely market-only with q as its base, not secretly sports-filtered.

The DRO gate checks positive return at `max(0,p-0.025)`; the existing wallet sizes from the qualified mean probability. This is **not a claim of robust-Kelly sizing or guaranteed downside protection**.

Rating reference and contemporaneous market-drift evidence are unavailable and explicitly **NOT_ASSESSED**. Missing data are not agreement or zero disagreement; no future closing price is used. Thus the study is not certification that every evidence-dependent production guard was assessed. Existing null/disabled policy options remain unchanged, not newly disabled for this study.

Source coverage, not just gate rejection, matters:

- Original paired-source eligible: **964 fixtures /178,094 quotes**.
- Matched current-model support: **482 fixtures /100,949 quotes**.
- Recorded source quotes in the primary48h interval: **452 fixtures /43,421 quotes**.
- Fresh LOO support in that interval: **370 fixtures /22,536 quotes**; another20,885 focal cases lack qualifying LOO support.
- All482 matched model fixtures remain in the financial denominator, including no-action cases. Quote/side counts are not independent bets.

Capital is reserved until the stored `outcome_known_at`, a recorded-result/finished-snapshot upper bound. Median lag from start is **17.49h**, p90 **367.68h**, maximum **1,322.50h**; 138/482 fixtures lag more than a week. These are not authenticated actual bookmaker settlement times. No earlier release time is invented. All accepted tickets settle by the retained export; pending count is zero.

## Primary financial results

Each alpha has an independent wallet and chooses its own tickets. ROI is realized profit divided by settled stake turnover, not return on initial capital.

| Sports share | Bets funded | Net P&L PLN | Stake ROI | Max drawdown | Flat100 P&L PLN |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.00 | undefined | 0.00% | 0.00 |
| 0.10 | 0 | 0.00 | undefined | 0.00% | 0.00 |
| 0.25 | 2 | -201.71 | -100.00% | 2.02% | -200.00 |
| 0.50 | 16 | -397.69 | -18.46% | 8.78% | -179.68 |
| 0.75 | 27 | +47.42 | +1.33% | 10.90% | +120.40 |
| 0.90 | 33 | -96.60 | -2.13% | 9.81% | +99.44 |
| 1.00 | 36 | -579.47 | -11.35% | 16.02% | -364.24 |

For50:50, turnover is2,154.85 PLN and final equity9,602.31 PLN. Fewer bets and lower absolute loss do not establish improved ROI: it performs worse than model-only on stake ROI and on fixed100 unit ROI. Flat100 remains negative, so delayed settlement/cash lockup alone cannot explain away the loss.

On the **same16 tickets**, gross-before-tax fixed100 ROI is +0.875%, versus -11.23% after the frozen tax. This is a fixed-selection arithmetic attribution, not a tax-free policy rerun or proof of a pre-tax edge. The selection originally required positive predicted after-tax EV.

Alpha0.75's +1.33% point ROI has a conditional monthly95% interval of approximately **[-53.59%, +101.07%]**. It is not a credible winning alpha. Market-only abstention preserves capital in this sample but is not evidence of profitable market-only betting; zero-bet ROI is undefined.

## Selected-EV reliability and fixed-ticket attribution

For50:50, equal-ticket mean predicted net return is **+10.80%** versus realized **-11.23%**: overstatement **22.03 pp**. On its eight original(5%,10%] EV tickets, overstatement is **8.06 pp**. Stake-weighted errors are different estimands: **29.49 pp overall /15.57 pp near-band**.

Holding the same16 sides/prices and the same original eight-ticket near-band fixed:

| Probability used to score fixed50:50 tickets | Mean predicted net EV | Expected-minus-realized error | Fixed near-band error |
|---|---:|---:|---:|
| Market-only | -7.80% | +3.43 pp | -9.13 pp |
| 50:50 | +10.80% | +22.03 pp | +8.06 pp |
| Sports-only | +30.57% | +41.80 pp | +26.24 pp |

All three rows have exactly the same -179.68 PLN flat100 P&L. Shrinkage reduces the sports-only EV overstatement on these fixed choices but does not make them reliable or change their realized payout. The market-only score already expected a negative return on these choices. This is a selected historical diagnostic, not a global calibration certificate.

Conditional50:50 ROI/error intervals are withheld: **8/5,000** resamples contain no retained tickets and **77/5,000** contain no original near-band tickets. Undefined denominators are counted, not replaced by zero or discarded. Consequently the95% error interval containment requirement in[-2,+2]pp is **not established**, even before considering the adverse point errors.

Paired cash contrasts resample full preselection months with zero-action fixture contributions retained. They condition on saved tickets/stakes; they do not rerun sequential wallet paths.

|50:50 minus control|Observed P&L difference PLN|Monthly95% interval PLN|
|---|---:|---:|
|Market-only| -397.69 | [-1,892.73, +633.18] |
|Sports-only| +181.78 | [-2,623.64, +2,401.98] |

Neither interval establishes a financial advantage. Observed differences are not bootstrap means; both are retained separately in the machine output.

## Timing sensitivity

Each interval is an independent policy/wallet, selecting its own first eligible observation. The same fixture may appear in several rows. Do not add bets/profits or choose the best interval after seeing these outcomes.

|50:50 allowed lead interval|Bets|P&L PLN|Stake ROI|
|---|---:|---:|---:|
|[0,6)h|6|+357.88|+47.64%|
|[6,12)h|5|+73.11|+11.04%|
|[12,24)h|8|-1,055.75|-100.00%|
|[24,48)h|6|-523.74|-61.44%|

Sparse, shared historical selections do not establish that betting inside six hours is reliably superior. The primary48h policy must act at its first eligible opportunity rather than hindsight-switching to a later favorable quote.

## What gates actually removed

Within the primary window at50:50 there are **45,072 side observations**. The following are ordered first failures, not independent causal contributions or distinct lost bets:

|Gate stage|First rejected side observations|Remaining|
|---|---:|---:|
|Net EV plus dynamic hurdle|44,065|1,007|
|DRO|359|648|
|Maximum EV|281|367|
|Odds ceiling|183|184|
|Tier1 market gap|33|151|
|Unshrunk sports/market logit gap|65|86|

Other stages add no first failures in this cell; unavailable rating/drift checks are not evidence of passing those references. The86 surviving observations collapse to16 first-eligible fixture tickets. Most quote observations do not exceed the after-tax EV hurdle; removing gates would be a different untested policy, not recovered demonstrated edge.

## Exposure semantics finding

The existing SSOT's `max_portfolio_exposure=0.25` parameter caps **each new stake relative to free cash**, not aggregate reserved exposure. An isolated actual-wallet probe with20 synthetic simultaneous tickets reserves3,000/10,000 PLN despite the0.25 parameter. No source or policy code was changed.

Actual primary peak reserved fractions in this sample remain below25% (50:50:2.91%; maximum across primary alphas11.68%). The semantic risk therefore does not explain these observed losses. Do not describe the existing parameter as enforcement of a hard aggregate25% exposure ceiling.

## Verification, recoveries and decision

Executed both archived analysis programs and independent verification programs in the available project Python environment. No application tests/build/lint, model training or new inference was run for this investigation.

- Independent probability reconstruction: **5,600 per-fixture score values**, maximum error **6.66e-16**; both primary adjusted intervals reproduced; all four release-aware temporal choices and exact memberships independently reconstructed.
- Independent finance reconstruction: **35 alpha/timing cases**, exact first-eligible keys using a separate raw-price/gate oracle; **316 funded ledger entries** checked against raw canonical prices/outcomes, Decimal payouts, chronological cash/reservations, sizing, equity and drawdown. Maximum money/summary error **0**. These repeated entries are not316 independent bets.
- Independent primary5,000-draw finance statistics: maximum cash/flat bootstrap draw difference **9.10e-13**; observed paired contrasts, undefined counts, fixed near-band membership and final reporting verified.
- All15 frozen input/policy bindings match; finance script hash matches provenance. Root/v7 canonical-to-GOLGG mappings and tournament labels agree across all482 finance fixtures. Scalar/vector parity samples agree; representative parity is not an independent rebuild of all source eligibility.
- Parent source-profile verifier initially confused the stored empty eligible-reason encoding with a display label; corrected to original source `exclusion_reason == ''`, retaining the964/178,094 invariant. No rows, probability, threshold or tolerance changed.
- The retired finance reporting attempt is retained under the run root's `retired_reporting_attempt/`. Repairs distinguish observed contrasts from bootstrap means, freeze near-band membership, scope gate counts to their timing windows, and label stake-weighted bootstrap versus equal-ticket error. Exact alpha1 endpoint handling is restored. These are analysis-integrity corrections, not a new profit-selected recipe.

**Decision:** no established probability superiority, no established positive policy value, and no established selected-EV reliability. Keep the negative results. Do not switch to alpha0.75 because its observed P&L is positive, loosen DRO/gates to regain volume, claim global impossibility, or deploy based on these results. Genuine confirmation still requires a frozen future-valid recipe and authenticated available inputs/execution; more reuse of these exposed outcomes cannot supply it. Canonical models/manifest, prior failed replacement guards, G0, historical evidence, old10/10 counter and forward dates remain unchanged.
