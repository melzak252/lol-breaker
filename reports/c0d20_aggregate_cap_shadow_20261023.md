# Aggregate reserved-capital correction: historical shadow replay

**Date:** 2026-10-23  
**Run:** `c0_aggregate_cap_20261023_01`  
**Status:** COMPLETED / HISTORICAL_CONDITIONAL_DIAGNOSTIC / NO_POLICY_ADOPTION

## Decision

The corrected Decimal wallet replay executed four fixed wallets on the frozen primary `[0,48)`-hour historical cohort. All four stayed within the 25% aggregate reserved-capital limit; none of the selected entries was rejected for cap/headroom. As a result, the selected/admitted counts and profit match the corresponding frozen uncapped historical comparisons. This is evidence that this particular replay did not exercise the cap, **not** evidence that the former per-ticket rule was an aggregate cap, that the cap is unnecessary, or that these historical returns will persist.

No model, production gate, serving default, source, or deployment changed. The baseline is unchanged. The research challenger differs from the frozen baseline only by setting `max_tier1_market_gap=null` (disabling the Tier-1 market-gap limit); every other policy setting is held fixed. The original uncapped selection/profit records remain the sealed comparator and were not rescored or overwritten.

## Frozen evidence and scope

The replay used protocol `data/05_model_input/c0_aggregate_cap/c0_aggregate_cap_20261023_01/protocol.json`, frozen before corrected-source scoring, and `data/08_reporting/c0_aggregate_cap/c0_aggregate_cap_20261023_01/freeze.json`. The run receipt confirms the frozen inputs and policies were validated. The frozen source contains 102,416 recorded event rows and a 482-fixture denominator; each wallet reports all 482 fixtures, including abstentions and no-action fixtures.

The model identity remains `c0-native-fresh20-2026-10-21-v1-regional-v1`; sports/market share is 50/50 with logit shrinkage. The underlying retained source is historical and conditional: the prior matched export comprises 100,949 scorable quotes across 482 fixtures and eight books, while fresh leave-one-book-out consensus has 51,208 quotes across 400 fixtures. Focal-book exclusion, retained quote freshness/state, historical source visibility, and source binding constrain the replay. These records do not authenticate that a prediction was issued, a displayed price was executable, a wager was placed, or a settlement clock was known at the historical decision time. Model forecasts are reconstructed counterfactuals, not certified historical emissions; roster, fill, limit and settlement evidence remains conditional.

Selection is deterministic and first eligible within `[0,48)` hours, never future-max. It reuses the frozen source selections and does not refit, infer, acquire data, access a database, or use the network. Both tax worlds (0% and 12%) are kept separate. The frozen source and study comparator are described in [the prior gate/timing audit](c0d20_gates_timing_optimizer_20261023.md), whose result files remain unchanged.

## Aggregate-cap wallet contract

The old `max_portfolio_exposure=0.25` behavior capped a new ticket as a fraction of free cash; it did not cap aggregate reserved capital. The corrected replay treats the 25% value as an aggregate limit. For cash `C` and already reserved stakes `R`, equity is `C + R` and remaining headroom is:

`max(0, 0.25 × (C + R) − R)`.

Stake sizing uses 25% Kelly on **free cash** (`C = equity − R`), with the existing 2 PLN minimum and 150 PLN maximum, and limits a new stake to the minimum of free cash, aggregate headroom, the maximum stake and the Kelly proposal. The Decimal stake itself must be no greater than Decimal headroom; existing Kelly precision is preserved and a proposed stake is never rounded upward to fit. A minimum stake is not forced when available headroom is smaller than the minimum.

If an externally supplied/recovered state has reservation above 25% of current equity, headroom is zero and new admissions remain rejected until settlement restores nonnegative headroom. The wallet does not force-settle or liquidate; ordinary settlement changes cash/equity and headroom normally. The implementation cutover is pinned in the receipt, and the pre-correction `staking.py` and engine preimages remain archived under the frozen study root.

## Four measured wallets

All wallets start with 10,000.00 PLN. P&L and returns below are exact Decimal-ledger results, displayed to two decimal PLN and two decimal percent; the full precision is preserved in `run_receipt.json`. All selected entries settled; no pending positions remained at the end. “No action” is the remainder of the fixed 482-fixture universe, not an additional set of cap rejections.

| Policy | Tax | Sealed uncapped bets / P&L (PLN) | Cap-replay selected / admitted | Cap rejections | No-action fixtures (of 482) | P&L change vs sealed (PLN) | Final equity (PLN) | Turnover / settled turnover (PLN) | ROI | Max drawdown | Peak reserved (PLN) | Peak reserved / equity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0% | 82 / +1,728.80 | 82 / 82 | 0 | 400 | 0.00 | 11,728.80 | 11,227.02 / 11,227.02 | +15.40% | 8.08% | 1,684.57 | 15.49% |
| Baseline | 12% | 16 / −397.69 | 16 / 16 | 0 | 466 | 0.00 | 9,602.31 | 2,154.85 / 2,154.85 | −18.46% | 8.78% | 300.00 | 2.91% |
| Challenger (Tier-1 gap off) | 0% | 85 / +2,005.62 | 85 / 85 | 0 | 397 | 0.00 | 12,005.62 | 11,754.68 / 11,754.68 | +17.06% | 7.72% | 1,697.63 | 15.21% |
| Challenger (Tier-1 gap off) | 12% | 18 / +6.63 | 18 / 18 | 0 | 464 | 0.00 | 10,006.63 | 2,424.54 / 2,424.54 | +0.27% | 6.86% | 422.77 | 4.10% |

The first count/profit column is the prior sealed uncapped comparison; the next is the corrected cap replay. Thus all four before/after wallet comparisons are explicit. Each row accounts for all 482 fixtures: cap-replay admitted plus no-action equals 482. There were **zero cap/headroom rejections in each corrected wallet**, so no source selection or profit changed; the original uncapped records remain intact rather than being replaced by the replay. Prior comparison figures are from the existing baseline/Tier-1-removal rows in the sealed audit, not a rescoring of those records.

These are historical point results with a limited, conditional source and no new independent future sample. They do not establish profitability, policy superiority, confidence bounds, reliable execution, or a profitable effect from the cap correction. The positive challenger zero-tax point result and near-flat 12%-tax point result are not adoption evidence.

## Aggregate admission stress

A separate controlled software stress exercised the contract in an intentionally saturated state: 12 simultaneous unknown-outcome opportunities, 1,000.00 PLN initial cash, and 25% aggregate exposure. The corrected wallet admitted stakes of 150.00 PLN and 100.00 PLN, rejected the other 10 admissions, and ended at 750.00 PLN free cash plus 250.00 PLN reserved; peak reservation was exactly 25% of 1,000.00 PLN equity. This demonstrates the admission boundary and retained unknown positions, not a real-world wager or a study outcome. No loss was forced to settle.

The parent-reported focused test run was **93 passed, with one existing Starlette warning**. The independent verifier now reports **PASS** (`independent_wallet_verification.json`): it checks the four frozen wallets and all **201 admitted entries**, independently replays the preserved FP64 Kelly proposal and then strictly applies the Decimal cash/headroom cap, with no Kelly tolerance. It also checks frozen source selections/input pins, the pre-change code archive, the sealed prior comparator hashes, and recomputed cash, reservation, profit and turnover. The earlier strict Decimal-proposal comparison failed because it compared the unchanged FP64 Kelly proposal as if the proposal itself were Decimal; the corrected verifier preserves that source calculation and strictly verifies the actual Decimal cap. Technical attempts are retained, including the runner's undefined-summary `NameError` and nested-summary Decimal-serialization failure; they were corrected before the immutable run receipt, without changing the frozen recipe or fits.

The parent separately reported configured Ruff checks passing for all six changed Python files; no application-wide or financial-certification claim follows from that check.

## Prospective shadow remains blocked

The prospective protocol is not a continuation of this retrospective `[0,48)` replay. A future shadow must keep the registered **T−1h primary admission cutoff**; these historical selections cannot substitute for it. The supplemental [shadow analysis contract](../data/08_reporting/c0_aggregate_cap/c0_aggregate_cap_20261023_01/shadow_analysis_contract.json) remains `BLOCKED_NOT_RUN` and authorizes no acquisition, scoring, deployment, or production change.

The original phases and maturity locks remain unchanged: TRAIN `[2026-11-01, 2027-11-01)` with labels locked 2027-11-15; CAL `[2027-12-01, 2028-06-01)` with labels locked 2028-06-15; DEVELOPMENT `[2028-07-01, 2029-07-01)` with labels locked 2029-07-15. The readiness deadline remains **2026-10-31T00:00:00Z** and the existing start remains **2026-11-01**. Confirmation dates remain null; no phase is rolled or extended.

Activation remains blocked on verified source entitlement/acquisition authority, authenticated release-aware native history and roster evidence, authentic actual starts and authoritative results/corrections/settlement, and independent fresh20 artifact eligibility/validity at the future decision date, alongside the other locked source and coverage requirements. No fixed sample count guarantees edge or precision. No activation, network/database request, collection, new forecast, fit, wagering, or deployment is claimed or authorized here. Native artifacts and all existing policy defaults remain unchanged.

## Evidence

- Frozen protocol: `data/05_model_input/c0_aggregate_cap/c0_aggregate_cap_20261023_01/protocol.json`
- Freeze manifest and pre-correction implementation preimages: `data/08_reporting/c0_aggregate_cap/c0_aggregate_cap_20261023_01/freeze.json` and `data/05_model_input/c0_aggregate_cap/c0_aggregate_cap_20261023_01/source_before_fix/`
- Actual four-wallet run: `data/08_reporting/c0_aggregate_cap/c0_aggregate_cap_20261023_01/run_receipt.json`
- Independent four-wallet replay PASS: `data/08_reporting/c0_aggregate_cap/c0_aggregate_cap_20261023_01/independent_wallet_verification.json`
- Prospective design only: `data/08_reporting/c0_aggregate_cap/c0_aggregate_cap_20261023_01/shadow_analysis_contract.json`
- Preserved prior comparison: `data/08_reporting/c0_gate_optimizer/c0_gate_optimizer_20261023_01/final_verification.json` and `optimizer/optimizer_results.json`
