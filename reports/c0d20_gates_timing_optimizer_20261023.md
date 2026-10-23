# C0-D-20: fixed shrinkage, gates, betting times and optimizer's curse

Date: 2026-10-23. Run: `c0_gate_optimizer_20261023_01`.

**COMPLETED / HISTORICAL_CONDITIONAL_DIAGNOSTIC / POLICY_SUPERIORITY_NOT_ESTABLISHED. No adoption or production change.**

## Decision

There is a positive historical zero-tax signal, not evidence that the model has become uniformly terrible. Some safeguards are logically redundant under the requested fixed 50/50 blend. Other gates materially change entry times, prices, exposure and returns. Removing everything does not improve the primary wallet.

The strongest searched recipes do not establish positive profit or improvement over the baseline after accounting for this declared search family. A release-aware historical selector underperforms the fixed policy at zero tax; at 12% tax it loses less, but still loses. Neither result proves profitability impossible. All history is development-exposed, and source/issuance limits remain.

## 1. Inputs and model identity

Only genuine **`c0-native-fresh20-2026-10-21-v1-regional-v1`**, called C0-D-20/fresh20+regional here, is used. The sports probabilities are the previously verified native counterfactual replay. The regional input transformation and one series projection remain unchanged. There are no EXP081 or tabular prediction comparisons, new fits, new inference, new acquisition or server writes.

Prices come from the retained read-only production export from `192.168.1.17`, not an OddsPortal opening-price substitute. The frozen matched source contains **100,949 scorable quotes / 482 fixtures / eight books**. Fresh leave-one-book-out consensus retains **51,208 quotes / 400 fixtures**. The focal bookmaker never contributes to its own reference: at least two other distinct books must have valid latest states no more than 300 seconds old. Latest invalid states block older quotes.

Retained LOO entry clocks run from **2026-05-29 10:09:28 UTC through 2026-10-22 11:45:38 UTC**. These are six partial/unequal source months, not six years. The source-fixture start-month counts are **14/46/78/232/94/18** for May–October. Current-model forecasts were reconstructed, not certified as forecasts actually issued by production at those historical clocks. Quote visibility, announced rosters, fills, limits and actual settlement remain conditional evidence.

For the primary [0,48)h window, 452 of the 482 matched fixtures have source prices; 370 have fresh LOO support. The fixed logit policy has 1,566 passing oriented side observations and 82 selected/funded fixtures at tax0, versus 86 passing observations and 16 selected/funded fixtures at tax12%. These are distinct counts; quote rows are not bets or independent tournaments. Full coverage remains the 482-fixture universe, including abstentions.

## 2. Frozen experiment

Exactly **88 configurations × 15 windows × two shrinkage formulas × two tax rates = 5,280 cases**. The wallet cache represents 2,435 distinct chronological wallets and 2,845 reused cases; it does not reduce the declared family. No alpha was optimized.

- All 64 on/off combinations of six guard groups: DRO; global plus longshot EV caps; maximum odds; Tier-1 positive model–market gap; BO1 absolute model–market gap; original-sports logit disagreement.
- Twenty additional parameter settings, two individual-cap redundancy controls and two minimal-value policies: 24 nonfactorial configurations in total.
- Eight disjoint intervals: [0,2), [2,6), [6,12), [12,24), [24,48), [48,96), [96,168), [168,infinity) hours before the conservative start.
- Seven cumulative windows: [0,6), [0,12), [0,24), [0,48), [0,96), [0,168), and all retained prestart observations. The last interval's actual support ends around 336 hours, not arbitrary distant forecasts.
- Tax0 primary; tax12% sensitivity, applied consistently to qualification, DRO, ceilings, Kelly sizing and payouts.

Input validity, identity alignment, nonlive status, chronology, usable model inputs and fresh consensus are never disabled. Rating disagreement and an authentic contemporaneous drift reference are unavailable: **NOT_ASSESSED**, not successful safeguards or evidence that those gates are unnecessary. Future closing prices are not inserted into qualification.

The baseline is minEV5%, dynamic short-odds hurdle scale0.08/cutoff2.2, DRO epsilon0.025, global and longshot maxEV25%, longshot odds3.5, max odds4, Tier-1 positive gap8pp, BO1 absolute gap12pp, and original-sports logit gap0.85. Tier1-only, BO1 maximum odds and coinflip gap are disabled in the baseline. Other unavailable-reference settings are retained without pretending they are assessed.

## 3. Probability, EV and execution contracts

For sports series probability `p`, same-clock other-book consensus `q`, decimal odds `o` and tax `t`:

- Probability blend: `r = (p + q) / 2`.
- Logit blend: `r = sigmoid((logit(p) + logit(q)) / 2)`.
- Net unit EV: `r * o * (1 - t) - 1`.
- Dynamic minimum: `minEV + max(0, (2.2 - o) / 2.2) * hurdle_scale`. This raises the hurdle for **short odds**, not long odds.
- DRO admission requires `max(0, r - epsilon) * o * (1 - t) - 1 > 0`. Epsilon is a probability haircut, not a guaranteed confidence level. DRO affects admission; the existing wallet sizes using mean probability.

The logit disagreement gate uses **unshrunk sports `p` versus `q`**, not the artificially closer blended probability. Probabilities are not clipped and the series projection is not applied twice.

Execution is the **first eligible recorded observation inside each interval**, not an exact T−6h/T−12h timestamp or the future maximum. Simultaneous observations use descending EV/odds then deterministic book/side ties. Each alternative has its own wallet and at most one bet per fixture; wallets and overlapping intervals must not be added together. This is quote-triggered execution, not a newly run every-two-hour board simulation.

Wallet: initial10,000 PLN, quarter Kelly, min stake2, max stake150; explicit tax; stakes reserved until recorded `outcome_known_at`. Entry outcomes are side-oriented; the wallet receives canonical outcome and flips side B exactly once. The observed model and price are unchanged by settlement or future results.

## 4. Fixed 50/50 method comparison

Proper scores average quote losses within bookmaker, then books within fixture, then fixtures. Repeated snapshots do not create independent samples.

| Forecast, same 400 LOO fixtures | Log loss | Brier |
|---|---:|---:|
| Sports only | 0.585025169 | 0.202814930 |
| Other-book market | 0.580297896 | 0.199347083 |
| 50/50 probability blend | 0.576631193 | 0.198585347 |
| 50/50 logit blend | 0.576161822 | 0.198601096 |

Probability-minus-logit LL is **+0.000469370**, 97.5% paired monthly interval **[−0.001008466,+0.001628610]**. The 50 source-name/year proxy interval also crosses zero. Probability averaging has a tiny favorable Brier point difference, also with intervals crossing zero. On the 370 primary-window fixtures, LL is 0.572061944/0.571678648 and Brier0.196581278/0.196629819, probability/logit respectively; neither method contrast is established. These comparisons do not newly establish bookmaker superiority.

| Baseline [0,48)h method | Tax | Bets | Kelly P&L, PLN | ROI | Max drawdown | Flat100 P&L, PLN |
|---|---:|---:|---:|---:|---:|---:|
| Probability | 0% | 79 | +1,364.43 | +12.64% | 7.31% | +761.00 |
| Logit | 0% | 82 | +1,728.80 | +15.40% | 8.08% | +964.00 |
| Probability | 12% | 16 | −357.26 | −16.39% | 6.95% | −118.08 |
| Logit | 12% | 16 | −397.69 | −18.46% | 8.78% | −179.68 |

Different methods select different tickets and stakes; this is not an isolated fixed-ticket probability effect. Fixed-ticket alternate-method scores, original near-band membership and exact stake weights are separately retained. Mean absolute probability/logit blend difference is 0.1951pp across raw LOO quotes; maximum3.9072pp. Neither formula is a validated winner.

## 5. Betting-time intervals

Baseline logit50 throughout. Fresh-LOO fixture support varies by interval; comparisons do not silently replace missing prices. P&L is from separate quarter-Kelly wallets.

| Disjoint lead interval | LOO fixtures | Tax0 bets | Tax0 P&L, PLN | Tax0 ROI | Tax12 bets | Tax12 P&L, PLN | Tax12 ROI |
|---|---:|---:|---:|---:|---:|---:|---:|
| [0,2)h | 291 | 31 | +83.50 | +2.08% | 3 | +50.20 | +13.92% |
| [2,6)h | 325 | 39 | +1,289.76 | +24.67% | 5 | +74.10 | +12.24% |
| [6,12)h | 328 | 49 | +962.59 | +14.37% | 5 | +73.11 | +11.04% |
| [12,24)h | 304 | 44 | −515.87 | −8.42% | 8 | −1,055.75 | −100.00% |
| [24,48)h | 249 | 42 | +0.87 | +0.02% | 6 | −523.74 | −61.44% |
| [48,96)h | 230 | 42 | −342.80 | −6.25% | 4 | −492.34 | −100.00% |
| [96,168)h | 168 | 25 | +0.47 | +0.01% | 1 | −150.00 | −100.00% |
| [168,infinity)h | 49 | 6 | +310.42 | +39.76% | 1 | +338.40 | +225.60% |

| Cumulative allowed lead | LOO fixtures | Tax0 bets | Tax0 P&L, PLN | Tax0 ROI | Tax12 bets | Tax12 P&L, PLN | Tax12 ROI |
|---|---:|---:|---:|---:|---:|---:|---:|
| [0,6)h | 329 | 43 | +1,136.54 | +19.87% | 6 | +357.88 | +47.64% |
| [0,12)h | 346 | 63 | +1,360.76 | +16.00% | 8 | +418.25 | +39.15% |
| [0,24)h | 356 | 75 | +1,401.34 | +13.48% | 13 | −271.89 | −15.90% |
| [0,48)h | 370 | 82 | +1,728.80 | +15.40% | 16 | −397.69 | −18.46% |
| [0,96)h | 393 | 93 | +2,036.40 | +16.02% | 17 | −464.92 | −20.93% |
| [0,168)h | 400 | 98 | +1,859.89 | +13.98% | 17 | −467.69 | −21.07% |
| All retained | 400 | 100 | +2,286.72 | +16.89% | 18 | −114.00 | −4.79% |

Among disjoint intervals with at least20 bets, the zero-tax [2,6)h slice looks strongest, but its nominal95% ROI interval is **[−23.74%,+38.54%]**. The primary [0,48)h interval is **[−4.22%,+48.59%]**. The [0,96)h and all-retained nominal intervals are positive, **[+2.14%,+48.71%]** and **[+1.81%,+50.93%]**, but these are unadjusted within-case retained-stake intervals after examining many alternatives—not validated timing choices.

All 60 method/tax/window baseline cases retain 5,000-draw uncertainty using the complete six-month source universe. Conditional ROI intervals are withheld if any resample has zero turnover: for zero-tax logit disjoint [12,24), [48,96), [96,168), [168,infinity), undefined counts are1/8/92/458 of5,000. Cash contribution zero for no action is valid; zero-bet ROI is undefined. These intervals resample retained ticket/stake contributions, not newly simulated wallet paths or policy refits.

## 6. Single-guard removals

Only the named guard group changes; fixed logit50, [0,48)h. Both EV caps are removed together in that row.

| Configuration | Tax0 bets | Tax0 P&L, PLN | Tax0 ROI | Tax12 bets | Tax12 P&L, PLN | Tax12 ROI |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 82 | +1,728.80 | +15.40% | 16 | −397.69 | −18.46% |
| DRO off | 89 | +1,557.98 | +13.78% | 20 | +66.06 | +2.89% |
| EV caps off | 83 | +2,177.67 | +18.89% | 16 | −397.69 | −18.46% |
| Maximum odds off | 100 | +1,706.84 | +13.25% | 33 | −137.46 | −3.56% |
| Tier-1 gap off | 85 | +2,005.62 | +17.06% | 18 | +6.63 | +0.27% |
| BO1 gap off | 82 | +1,728.80 | +15.40% | 16 | −397.69 | −18.46% |
| Sports-logit gap off | 89 | +1,315.11 | +10.81% | 22 | −1,294.76 | −43.12% |

Effects include earlier/different entries on already selected fixtures. At tax0, DRO removal adds seven fixtures and changes 11 existing entries; EV-cap removal adds one and changes four; odds-cap removal adds18 and changes two; Tier-1 removal adds three and changes five; logit-cap removal adds seven and changes four. Relaxation is not equivalent to appending some new bets at unchanged old prices.

DRO illustrates the metric distinction: removal raises flat100 P&L from964 to1,190 PLN but lowers the Kelly wallet by170.82 PLN. It is not established as universally helpful or harmful. Removing the sports-logit guard hurts the primary baseline at both tax rates.

## 7. All nonfactorial parameter settings

The table lists the primary zero-tax logit wallet; all methods, taxes and windows are retained in the machine outputs. Changes are relative to the baseline, not successively accumulated. Minimal-value policies deliberately change several guards, so they are not isolated parameter effects.

| Setting | Bets | Kelly P&L, PLN | ROI |
|---|---:|---:|---:|
| minEV0 | 102 | +786.55 | +5.71% |
| minEV2.5% | 96 | +844.13 | +6.51% |
| minEV7.5% | 66 | +1,819.04 | +19.59% |
| minEV10% | 54 | +1,290.00 | +16.85% |
| Short-odds hurdle scale0.04 | 86 | +1,581.86 | +13.43% |
| DRO epsilon0.01 | 89 | +1,557.98 | +13.78% |
| DRO epsilon0.05 | 41 | +1,068.00 | +17.37% |
| Maximum odds3 | 60 | −337.01 | −4.20% |
| Maximum odds6 | 95 | +2,079.11 | +16.46% |
| Sports-logit gap0.5 | 61 | +1,472.30 | +18.26% |
| Sports-logit gap1.25 | 87 | +1,389.02 | +11.69% |
| Tier-1 gap4pp | 66 | +506.25 | +5.85% |
| Tier-1 gap12pp | 85 | +2,005.62 | +17.06% |
| BO1 gap8pp | 81 | +1,596.19 | +14.37% |
| BO1 gap20pp | 82 | +1,728.80 | +15.40% |
| Longshot threshold odds2.5 | 82 | +1,728.80 | +15.40% |
| Longshot threshold odds5 | 82 | +1,728.80 | +15.40% |
| Dynamic hurdle off | 93 | +1,014.80 | +8.04% |
| Both EV ceilings15% | 80 | +629.73 | +6.02% |
| Both EV ceilings50% | 83 | +2,177.67 | +18.89% |
| Only global EV cap off | 83 | +2,177.67 | +18.89% |
| Only longshot EV cap off | 82 | +1,728.80 | +15.40% |
| EV-only5%, six guards/hurdle off | 180 | +1,178.13 | +5.84% |
| EV-only nonnegative, six guards/hurdle off | 271 | +1,360.51 | +6.28% |

Lowering minEV or removing all safeguards adds volume without improving this primary wallet. Maximum odds3 substantially worsens it. These are historical sensitivities, not selected future settings.

## 8. Full-factorial attribution and actual redundancy

Each of six guards is toggled across 32 matched settings of the other five guards, for each method/tax/window: **11,520 comparisons**. A separate **5,220-row** table compares every non-baseline configuration with baseline; its **1,440 nonfactorial rows** are the 24 parameter/redundancy/minimal-policy variants, not87 OAT settings. Fixture-level attribution distinguishes newly admitted fixtures from changed sides, books, prices and decision clocks; flat cash decomposes into new, removed and changed-common-fixture contributions.

For logit50 [0,48)h, the counts below describe the sign of **guard-off minus guard-on Kelly P&L** across those 32 contexts. They are overlapping descriptive comparisons, not32 independent replications.

| Guard removed | Tax0 better / worse / same | Tax12 better / worse / same |
|---|---:|---:|
| DRO | 0 / 32 / 0 | 16 / 16 / 0 |
| EV caps | 12 / 20 / 0 | 4 / 20 / 8 |
| Odds cap | 14 / 18 / 0 | 16 / 16 / 0 |
| Tier-1 gap | 32 / 0 / 0 | 32 / 0 / 0 |
| BO1 gap | 0 / 0 / 32 | 0 / 0 / 32 |
| Sports-logit gap | 16 / 16 / 0 | 0 / 32 / 0 |

The Tier-1 8pp limit is the clearest gate to challenge in a **separately confirmed** recipe: removal helps every primary matched context at both taxes. The result does not demonstrate a population effect or authorize removing it from production.

Two stronger redundancy statements are mathematical, not profit-picked:

1. **Longshot EV cap:** when the global ceiling25% is enforced, a second25% ceiling conditional on longshot odds cannot reject another row. Removing it or moving its odds threshold to2.5/5 leaves exact actions unchanged in all60 baseline contexts.
2. **BO1 gap12pp:** while the original-sports logit gap is at most0.85 and blend weights are50/50, the maximum arithmetic-blend gap from market is `0.5*tanh(0.85/4) = 0.104679085`; for logit blending it is `tanh(0.85/8) = 0.105851976`. Both are below0.12. Thus that BO1 guard cannot activate in this contract. All60 baseline contexts and960 factorial pairs keeping the logit guard active have identical actions.

The bound follows from the maximum logistic probability difference at logit distance `d`, `tanh(d/4)`, attained at symmetric logits. It was also checked numerically and on actual inputs. It does **not** justify globally deleting the BO1 gate for other alphas, a disabled/looser logit guard or different probability inputs. Tier-1 8pp can still activate.

## 9. Ticket-level selection optimism and EV reliability

The decomposition is exact on each side-oriented ticket:

`predicted_return - realized_return = (r - q) * odds * (1-tax) + (q - y_side) * odds * (1-tax)`.

For zero-tax logit82, equal-ticket expected return is **10.54%**, realized11.76%, error−1.21pp, monthly95% interval **[−30.61,+18.50]pp**. Stake-weighted error is−4.72pp. The original **42 tickets in (5%,10%] expected EV still overstate return by11.59pp**, stake-weighted7.08pp; their conditional interval is withheld because8/5,000 monthly draws have no near-band bets. The ±2pp reliability objective is not met. Probability averaging's79-ticket unit/stake errors are+1.04/−1.83pp; its36-ticket near-band error is+4.55pp, also with eight undefined draws.

At zero-tax logit primary, a same-clock lexicographic-book control changes zero tickets: it supplies no identifying evidence about book-shopping optimism. A deliberately **NONCAUSAL future-max-predicted-EV** diagnostic changes51/82 tickets, raises mean stated EV by2.91pp and realized return by1.66pp. The additional error is1.26pp,95% interval[−0.51,+1.64]pp. This diagnostic is not an executable strategy or proof of a general optimizer-bias parameter. The actual strategy never selects the future maximum. All60 baseline cases, fixed-ticket alternate scoring and120 selection-rule comparisons are archived with exact stakes and original near-band definitions.

## 10. Searching configurations: what survives correction?

The formal search diagnostic ranks flat100 total cash with at least20 selected tickets. Each tax has **2,640 candidate cases**. All482 source fixtures remain in the denominator, with zero cash for abstentions. Paired5,000-draw monthly bootstrap uses the same six sampled blocks across the entire family. Centered maximum-absolute-deviation bands are computed for both absolute cash and cash relative to the fixed logit/baseline/[0,48)h case. A67 source-name/year proxy sensitivity is separate; those proxies are not authenticated independent tournament editions.

| Hindsight flat-cash winner | Bets | Flat100 P&L, PLN | Kelly P&L, PLN | Kelly ROI |
|---|---:|---:|---:|---:|
| Tax0: logit, EV-only nonnegative, [0,168)h | 309 | +3,309.00 | +3,133.87 | +13.01% |
| Tax12: probability, g46, [0,12)h | 38 | +1,459.76 | +1,090.77 | +21.27% |

g46 removes EV ceilings, maximum odds, Tier-1 gap and sports-logit gap, while retaining DRO, BO1 gap and the baseline hurdle. These are hindsight-selected configurations, not forward recommendations.

- Tax0 winner excess mean flat cash is **+4.865 PLN/source fixture**, simultaneous95% monthly band **[−10.964,+20.694]**. Its absolute band is **[−8.979,+22.710]** around6.865 PLN/fixture.
- Tax12 winner excess is **+3.401**, band **[−7.532,+14.335]**. Its absolute band is **[−6.029,+12.086]** around3.029 PLN/fixture.
- **No case** has a positive absolute lower band or an excess-over-baseline band excluding zero at either tax, under either monthly or proxy blocking. Unadjusted relative pointwise bands exclude zero for79/155 cases, tax0/tax12; the family bands do not.

These are nominal bootstrap bands within this declared experiment, not correction for every earlier adaptive study, proof of independence, certification of Kelly ROI, or proof that an edge is impossible.

Choosing a different objective creates different apparent winners. The maximum observed Kelly-cash case with at least20 bets is probability/g45/all-retained at tax0:171 bets,+3,541.54 PLN,+19.73% ROI; at tax12 it is logit/g45/[0,6)h:34 bets,+1,541.51 PLN,+48.64% ROI. Maximum-ROI cases are different again and sparse. All are labeled noncausal recipe selection; none replaces the declared flat-cash comparison or establishes future success.

## 11. Release-aware historical policy selection

May and June decision months are warmup; evaluation is **July1 through the last retained October22 observation**, with an exclusive November1 boundary. There are360 shared observed evaluation fixtures, not a completed future October/November collection period.

At each UTC month origin, candidate ranking uses only fixtures whose recorded outcome-known clock is strictly earlier than origin, and only entries earlier than origin. Support requires at least20 past bets and two bet-decision months. Ranking is past flat100 cash per closed source fixture, with deterministic conservative ties. No support means explicit fixed-baseline fallback. Training closed-fixture counts at July/August/September/October origins are44/130/312/432.

The chosen policy selects first eligibility **after that month's boundary**, excluding already bet fixtures. One continuous wallet spans all evaluation months; there are no monthly bankroll resets. Baseline uses the same evaluation boundary and its own one-bet-per-fixture history. Evaluation hindsight winners are separately recomputed after the evaluation start, not filtered from an earlier global first-entry list.

| Tax | Policy | Bets | Flat100 P&L, PLN | Kelly P&L, PLN | Kelly ROI | Max drawdown |
|---|---|---:|---:|---:|---:|---:|
| 0% | Fixed baseline | 74 | +986.00 | +1,578.63 | +15.55% | 8.26% |
| 0% | Prior-released-label selector | 94 | +557.00 | −38.79 | −0.35% | 8.97% |
| 12% | Fixed baseline | 14 | −287.68 | −521.66 | −28.03% | 8.85% |
| 12% | Prior-released-label selector | 17 | −204.88 | −167.28 | −9.62% | 6.85% |

At tax0, selected recipes are baseline/48h in July, g41/168h in August, g45/[48,96)h in September and EV-only-nonnegative/168h in October. At tax12, July/August fall back to baseline; September/October choose g13/12h. Exact TRAIN IDs, origins, support counts, clocks, tickets and all eight future-label-flip proofs are retained.

Selector-minus-baseline flat cash is−429.00 PLN at tax0 and+82.80 at tax12. Per shared evaluation fixture, four-month95% intervals are **[−15.319,+3.894]** around−1.192 and **[−1.408,+1.808]** around+0.230 PLN. Equal-month and36-proxy intervals also cross zero. The zero-tax observed loss of performance is consistent with search optimism, not a precise causal estimate proving overfitting. At12%, loss reduction is not profitability. The historical family/model design remains exposed to later outcomes despite release-aware selection inside the replay.

## 12. Capital risk is not a 25% aggregate limit

The existing `max_portfolio_exposure=0.25` parameter limits **each new stake relative to free cash**, not total reserved capital. This behavior was preserved rather than silently changing bankroll rules during a gate experiment.

Primary logit0/tax12 peak reserved fractions are15.49%/2.91%. Across the full family, **148 cases exceed25% aggregate reservation**, maximum **41.07%**. The zero-tax maximum-cash hindsight case reserves28.23%; the flat-cash winner reserves33.61%. Higher volume can therefore violate an assumed aggregate risk cap even when an existing parameter is named25%. These cases cannot be described as satisfying a true25% aggregate-cap contract.

No capital engine fix was deployed. A real aggregate exposure cap and authentic decision/fill/settlement evidence are separate operational prerequisites, not evidence that modifying the parameter would preserve the reported returns.

## 13. Verification and retained corrections

Executed archived scripts performed the actual sweep, selection, scoring, resampling and independent monetary replay. No new application tests were run because application code did not change.

- 21,120 scalar/vector gate decisions across all88 configs, two methods and two taxes on60 representative rows; zero mismatches. This is not exhaustive scalar evaluation of every source row.
- All5,280 selected-action cases independently reconstructed against source/gates/windows; all246,521 selected contribution cells match exact oriented flat100 arithmetic, maximum error0.
- Every one of246,521 repeated funded entries is checked against frozen side, odds, outcome, clock, probability and payout. All case counts, flat P&L, funded profit and turnover match.
- Independent heap-based Decimal replay covers all2,435 non-reused chronological wallets and134,943 funded entries; stakes, settlements, final cash/reservation, profit, turnover and maximum drawdown match. Maximum monetary difference4.55e−13 PLN. Counts include repeated hypothetical actions, not unique real bets.
- Independent proper-score reconstruction checks39,504 scalar score values with maximum error0; method contrasts include weighted/equal-month and source-proxy results.
- Parent independently recomputes both2,640-case monthly search bands, the67-proxy sensitivities, all eight source-bound training choices and perturbed future-cash choices, all evaluation first-entry sets, and all four continuous forward/baseline wallets. Maximum recomputation error0. Absolute-profit family bands are independently computed as well.
- All31,680 case-month rows reconcile to case money, counts and exact joined stakes. All11,520 factorial toggle decompositions reconcile, every guard has32 other-guard contexts, and all1,440 nonfactorial comparisons are retained separately from5,220 total baseline comparisons.

Reporting-stage errors were caught and retained: canonical-versus-side returns in an early monthly table, decision-month versus fixture-start-month attribution, total-window rows mislabeled eligible, approximate stake alignment, factorial baseline/on-off labeling, event-ID overlap hiding changed fixture entries, and repeated risk-merge suffixes. The corrected reporting path does not change the frozen5,280 configurations, raw selections or wallet profits. Retired shrinkage and gate-reporting attempts remain in the run directory.

One independent bootstrap attempt failed because NumPy's default int64 random draws differ from the producer's int16 draws even with the same seed. The verifier now uses the retained actual draw matrix and independently confirms its int16 regeneration. No model, policy, confidence level or numerical tolerance changed. A report-table index-column error was corrected before publication. These are verifier/reporting failures, not extra model recipes.

## 14. Evidence and conclusion

Frozen inputs: `data/05_model_input/c0_gate_optimizer/c0_gate_optimizer_20261023_01/` (`protocol.json`, `configurations.json`, `events.parquet`, `fixtures.parquet`, `source_loo_accounting.parquet`, `binding.json`).

Executed outputs: `data/08_reporting/c0_gate_optimizer/c0_gate_optimizer_20261023_01/`:

- `sweep/summary.parquet`, `selected.parquet`, `funded_ledger.parquet`, `monthly_aggregates.parquet`, `baseline_tables.csv`, `gate_attribution.parquet`, `baseline_variant_attribution.parquet`, `parameter_attribution.parquet`, `gate_directional_summary.parquet` and receipt.
- `shrinkage/` corrected scoring, fixed-ticket and selection-rule outputs; `verification/shrinkage_statistics.json` supplies complete stake-weighted and interval results.
- `optimizer/` numeric pickle-free contribution/count/event-index matrices, retained paired month draws, full case tables, training provenance, evaluation tickets/wallets, result and receipt.
- `verification/{core_crosscheck,sweep_crosscheck,optimizer_crosscheck,absolute_search_uncertainty,policy_diagnostics,report_tables,bootstrap_rng_diagnostic}.json`, independent wallet-risk parquet and executed verifier sources.
- `retired_shrinkage_attempt/`, `retired_gate_reporting_attempts/` and final closure `final_verification.json`.

The command sequence actually exercised the sweep and scoring analyzers, `verification/verify_core.py`, `verification/verify_sweep.py`, final `sweep/repair_and_attribute.py`, `verification/complete_shrinkage_statistics.py`, `optimizer/analyze_optimizer.py`, `verification/verify_optimizer.py`, `verification/absolute_search_uncertainty.py`, `verification/complete_policy_diagnostics.py` and `verification/build_report_tables.py`. These paths record this run, not authorization to reuse an immutable run name for another experiment. A new experiment requires a new namespace and a new binding before execution.

**Practical result:** keep50/50 as requested; neither probability nor logit blending has established superiority. The longshot ceiling and BO1 gap are redundant only under the stated contracts. The Tier-1 gap is a plausible challenge for separately frozen confirmation, not a deployed improvement. There is no evidence here for dropping every guard or selecting the highest historical ROI. A profitable zero-tax point result exists; reliable future financial edge and near-band ±2pp reliability do not. No impossibility claim follows.

Canonical models/benchmark, prior failed replacement guards, serving configuration, source activation, G0, the exhausted historical10/10 journal and frozen forward dates are not changed. No H fit, new future forecast, live acquisition, deployment, financial certification or automatic follow-up sweep occurred.
