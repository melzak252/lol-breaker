# C0-D-20: zero tax and closing-line value — 2026-10-23

**COMPLETED / HISTORICAL_CONDITIONAL_DIAGNOSTIC.** Requalifying and restaking at0% tax changes the primary50:50 policy from16 bets/-397.69 PLN to **82 bets/+1,728.80 PLN, ROI+15.40%**. Its monthly95% ROI interval **[-4.22%,+48.59%]** crosses zero. Fresh closing-line comparisons are positive on the covered selected bets, but cover only a minority of them. Neither profitable future execution nor complete-portfolio CLV is established.

Run `production_zero_tax_clv_20261023_01`; frozen protocol `data/05_model_input/production_zero_tax_clv/production_zero_tax_clv_20261023_01/protocol.json`. Outputs are under `data/08_reporting/production_zero_tax_clv/production_zero_tax_clv_20261023_01/`, including `finance/`, `clv/` and independent `verification/`.

## Scope and controlled change

Only genuine **C0-D-20 / fresh20+regional**, identity `c0-native-fresh20-2026-10-21-v1-regional-v1`. Reuses the actual retained production export from192.168.1.17, saved counterfactual probabilities and unchanged source/model exclusions. Matched support is482 fixtures/100,949 quotes from May–October2026; entry-time LOO consensus has51,208 quotes/400 fixtures. This is not the older2021–2026 odds archive or genuine issued forecasts.

Zero tax is applied consistently to qualification EV, the DRO return calculation, EV ceilings/hurdles, mean returns, Kelly sizing and payouts. Every other policy setting remains fixed: minimum EV5% plus the existing low-odds hurdle, DRO0.025, EV ceiling25%, odds ceiling4, unshrunk sports/market logit gap0.85, existing Tier1/BO1 guards, quarter Kelly, capital10,000 PLN, stakes2–150 PLN and recorded result-known settlement clocks. Missing rating/current-drift references remain NOT_ASSESSED. The0.25 exposure parameter retains its existing per-ticket free-cash semantics, not an aggregate exposure guarantee.

The primary policy takes the first eligible recorded observation in[0,48)h before the known start bound, one ticket per fixture. All482 matched fixtures remain in accounting, including abstentions. No future-best quote, closing price or result enters selection. Tax changes can add or remove bets: the EV ceiling also changes eligibility. Alpha is the sports share in a logit blend, with exact market/model endpoints.

No source refresh, model fit, neural inference, live tax/configuration change, production write, deployment or real betting occurred.

## Zero-tax own-policy results

Each row independently reselects and funds its own tickets. ROI is profit divided by settled stake turnover.

|Sports share alpha|Bets|P&L PLN|Stake ROI|Max drawdown|
|---:|---:|---:|---:|---:|
|0 — market only|14|-144.08|-8.34%|3.62%|
|0.10|21|+421.14|+15.94%|4.49%|
|0.25|38|+781.30|+16.44%|5.91%|
|0.50|82|+1,728.80|+15.40%|8.08%|
|0.75|104|-424.16|-2.96%|18.88%|
|0.90|113|-405.60|-2.57%|21.44%|
|1 — sports only|112|-1,231.80|-7.84%|19.74%|

All primary selected bets were funded and settled, with zero pending. For50:50, stake turnover is11,227.02 PLN, final equity11,728.80 PLN and flat100 P&L+964.00 PLN (unit ROI+11.76%). Peak reserved fraction is15.49%; no aggregate25% cap is claimed.

The50:50 equal-ticket expected/realized return is **+10.54%/+11.76%**, error **-1.21pp**. Its42 bets originally in the(5%,10%] zero-tax EV band still have error **+11.59pp**. Stake-weighted overall error is-4.72pp, with95% interval[-36.86,+15.39]pp; the stake-weighted near-band error is+7.08pp and its interval is withheld for8/5,000 no-band resamples. Thus the favorable overall point error does not establish the required±2pp reliability.

Monthly95% ROI intervals are not alpha-search-adjusted or wallet-path resimulations. For alpha0.25 the interval is[-0.59%,+146.52%]; for0.50 it is[-4.22%,+48.59%]. Alpha0/0.10 intervals are withheld because92/5,000 draws have no bets. No weight is selected for live use from this grid.

Conditional retained-ticket/stake P&L difference50:50-minus-market is+1,872.88 PLN,95% interval[-119.82,+4,462.39]; versus sports-only it is+2,960.60,[+6.58,+6,022.02]. The latter nominal cash contrast is favorable, but the fixed100 contrast crosses zero, the grid/history were exposed, and positive standalone policy value remains unestablished. This is not proof of probability superiority or a deployment decision.

## Removing tax from the original bets is different

|Scenario for primary50:50|Tickets|P&L PLN|ROI on stakes|
|---|---:|---:|---:|
|Original12% policy and original stakes|16|-397.69|-18.46%|
|Same16 sides/prices/stakes, payouts at0%|16|-158.08|-7.34%|
|Full0% policy: requalify and resize|82|+1,728.80|+15.40%|

Tax removal alone adds239.61 PLN on the original funded stakes but does not make that wallet profitable. On the original16 tickets at flat100, zero-tax P&L is+14.00 PLN, ROI+0.875%; this is a fixed-selection diagnostic, not the82-bet policy. New selections and sizing explain why the full replay differs from merely subtracting tax.

## Timing sensitivity at0% tax

Each interval is an independent wallet; fixtures overlap and the rows must not be added or selected by hindsight.

|50:50 allowed lead interval|Bets|P&L PLN|Stake ROI|
|---|---:|---:|---:|
|[0,6)h|43|+1,136.54|+19.87%|
|[6,12)h|49|+962.59|+14.37%|
|[12,24)h|44|-515.87|-8.42%|
|[24,48)h|42|+0.87|+0.015%|

These are recorded observation-triggered scenarios, not complete every-two-hour board scans. They do not establish a universally superior betting time.

## What CLV means here

Two distinct references are reported; neither is the model/current-consensus gap.

1. **Same-book price CLV:** `entry_odds / later_same_book_prestart_odds - 1`. Positive means the entry price was higher than that book's later price.
2. **Fair-price CLV against other-book closing consensus:** `entry_odds * q_close_same_side - 1`. The proportional no-vig closing probability is a logit consensus of at least two OTHER books; the focal entry book is excluded. At0% tax this is the return implied by that closing benchmark, not a prediction guarantee.

References are the last retained state strictly before the conservative minimum known fixture start. A latest invalid/conflicting state is not replaced with an older valid price. Same-book clocks must be strictly after entry; a consensus needs at least one contributing observation after entry. No live/poststart observations are used. Primary freshness is≤300s before the start bound;≤3,600s is a sensitivity, and unrestricted-age references are explicitly stale diagnostics.

The timestamps/bindings do not authenticate official market CLOSE, actual start or executable fills. All statistics are conditional on covered references; missing-close values are not zeros. The full-population CLV of uncovered entries/bets remains unidentified.

## CLV across all matched opportunities

There is no unique signed CLV without specifying a side. The primary overall side is the sports model's favorite at entry; exact-neutral forecasts are explicit absences. Quotes are averaged within fixture/book, books equally within fixture, then fixtures equally. Thus prolific snapshots or fixtures with more books do not dominate.

|Closing reference age|Favorite fair-price CLV|Covered quotes / fixtures|Favorite same-book price CLV|Covered quotes / fixtures|
|---|---:|---:|---:|---:|
|≤5min|-6.38%|35,528 /146|+1.99%|31,578 /172|
|≤1h|-6.42%|62,516 /229|+1.90%|56,849 /273|
|Any age, diagnostic|-6.71%|97,270 /402|+1.70%|96,839 /475|

There are100,949 quotes/482 fixtures in the full matched universe. The≤5min overall intervals are withheld because1/5,000 monthly draws has no covered fixture. At≤1h, fair-price CLV95% interval is[-6.72%,-5.72%]; same-book price CLV is[+1.22%,+3.27%]. These metrics use different references and coverage; favorable own-book price movement does not imply positive margin-free market value.

A separate pre-gate side rule selects the highest entry-time50:50 expected return, without financial gating or closing-price knowledge. Its fair-price CLV is **-4.05%/-4.03%/-4.07%** for5min/1h/any-age references, on19,663/34,237/50,811 quotes. This differs from both the favorite-side summary and actual selected bets; it is not a profitable ungated policy.

Any-age consensus contributor ages have median887s,75th percentile7,469s,95th percentile241,498s and maximum1,215,983s (about14.1days). The increased coverage is not certified closing freshness.

## CLV on selected50:50 bets

The tax labels below identify different selected portfolios; tax itself does not change the price-CLV formula. Means are equal-ticket and conditional on coverage.

|Selected portfolio|Reference age|Fair-price CLV|Covered bets|Stake-weighted fair-price CLV|
|---|---|---:|---:|---:|
|Original12% policy|≤5min|+12.36%|4/16|+12.17%|
|Original12% policy|≤1h|+10.29%|8/16|+11.31%|
|Original12% policy|Any age|+10.69%|15/16|+11.79%|
|New0% policy|≤5min|+1.82%|19/82|+1.37%|
|New0% policy|≤1h|+0.60%|47/82|+0.62%|
|New0% policy|Any age|+2.30%|80/82|+2.53%|

For the0% policy,11/19 covered fresh-reference bets have positive fair-price CLV. The fresh conditional interval is withheld because8/5,000 draws contain no covered bets. At≤1h, the equal-ticket interval is[-0.99%,+9.04%], and any-age[-0.30%,+7.13%]; neither establishes positive fair-price CLV. Original12% intervals are withheld, with458/458/8 undefined draws respectively. The favorable original12% conditional means cannot be generalized to the12/8/1 uncovered tickets.

Conventional same-book price CLV on the0% portfolio is **+3.77% on17/82** fresh-covered bets, **+7.16% on41/82** at≤1h and **+8.77% on74/82** at any age. The respective equal-ticket intervals are withheld,[-0.63%,+12.19%] and[+4.89%,+13.49%]. The last result uses potentially stale references, not an authenticated market close. Original12% same-book figures are+5.30% on2/16 fresh,+3.78% on5/16 at≤1h,+11.39% on10/16 at any age.

Positive gross fair-price CLV does not automatically overcome12% payout tax: break-even requires gross CLV above13.636%. On the original covered tickets, the closing-consensus-implied after-tax returns are approximately-1.12%/-2.95%/-2.59% for the three references. These are benchmark arithmetic on fixed covered choices, not realized ROI or a new policy.

Machine-readable bet joins cover all seven alphas and both tax scenarios under `verification/both_tax_primary_bet_clv.parquet`; full metric/coverage/interval objects are in `verification/clv_crosscheck.json`.

## Verification and retained corrections

- Actual12% replay reproduces the sealed baseline across all35 timing/alpha cases; zero-tax money/selection is separately reconstructed from raw source prices and a separate gate oracle.
- Independent zero-tax verification covers **35 cases /1,557 repeated funded entries**, canonical outcomes/side alignment, Decimal payouts, chronological cash/reservations, mean Kelly sizing, equity and drawdown, with maximum money/summary error **0**. Entries across wallets are not independent bets.
- Independent5,000-draw zero-tax statistics reproduce cash draws within **2.73e-12**, including ROI/error intervals and undefined-draw counts. These condition on retained tickets/stakes, not full path resimulation.
- Independent CLV reconstruction verifies all **5,762 latest strictly-prestart fixture/book states**, raw-state eligibility/conflicts, canonical raw prices and **908,541 entry metric values**, maximum error **2.45e-15**. Both tax portfolios are joined using their actual selected sides and prices; no favorite-side substitution on bets.
- Superseded CLV artifacts are retained under `retired_clv_attempt/`. Review caught a microsecond/nanosecond search mismatch before publication, same-book age masks, an optional side rule using market-only rather than the required blend, and conditional-bootstrap/overall-weighting mismatches. The corrected computation enforces strict prestart clocks and the frozen estimands. These corrections alter analysis, not native probabilities or financial policy.
- The parent verifier's initial bet join collided with entry `q`/eligibility field names; namespaced closing-reference columns resolve it. Inputs, prices, alpha grid and tolerances remain unchanged.
- The finance runner uses a documented deterministic in-memory tax-literal adapter over pinned parent analysis source and existing SSOT functions. It copies policy state per run and does not edit shared files. Copied baseline summaries are not mislabeled replay verification; genuine replay parity and scenario comparison are separate artifacts.

No new application tests, full application regression, genuine future forecasts or source authentication are claimed. The canonical model/benchmark manifest, prior sealed research, operational defaults, financial gates, G0, old10/10 and forward dates are unchanged.

**Decision:** zero tax materially improves this historical50:50 policy's point returns, while sports-only still loses. Selected covered bets show more favorable CLV than indiscriminate opportunities, but coverage and uncertainty do not establish a reliable positive edge. Do not choose a new alpha or betting time by these outcomes, relabel stale references as official CLOSE, or deploy from these historical diagnostics.
