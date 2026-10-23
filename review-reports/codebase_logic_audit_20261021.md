# Codebase logic audit and comparison with best-known practice — 2026-10-21

Scope: read-only audit of `main` at `9fe6bb2` (clean tree). No code, models, manifest, database or server was changed. Model artifacts, DVC data and the research archive are **not present on this machine**, so findings come from code, executed tests and reports, not from re-scoring data.

## 0. Verdict

**Not production ready.** The application plumbing is solid: wallet money is Decimal, settlement is idempotent and atomic, quote timing is checked properly, and migrations apply cleanly. The problem is the documentation. The newest README, `docs/MODEL_CATALOG.md` and the 2026-10-17/20 sections of `docs/RESEARCH.md` describe a decision stack, a model family and financial results that the running code does not implement, and in places that no code in the repo produces. Several headline numbers are in-sample, look ahead, are hard-coded, or are true by construction.

The literature puts pre-match LoL models at roughly market parity. The project is close to that: on the matched 1,467-series Polish cohort its model LL is 0.5659 against a market LL of 0.5596. The claimed +24–28% yield and "99% guaranteed profit" are not credible under accepted standards for proving a betting edge.

## 1. What actually ran today

| Check | Result |
|---|---|
| Full `betting_app/tests` on a throwaway Timescale container (loopback port, destroyed afterwards) | **4,484 passed, 67 failed, 16 errors, 53 skipped** (17.5 min) |
| `alembic upgrade head` on an empty database | All 16 revisions applied without error |
| `client`: `npm ci`, `npm run build`, `npm run lint` | All passed |
| `run_model_benchmark.py --suite --doctor` | Could not run: the research root is absent |
| `ruff --select F` on `src`, `betting_app`, `scripts` | 716 findings, mostly unused imports; 30 undefined names, 13 duplicate dict keys, 1 syntax error |

Environment note: there is no `.venv`. Tests ran in a temporary Python 3.12 venv with CPU `torch==2.5.1`, not the CUDA pin.

### 1.1 Test failures that are real code bugs, not missing data

- **`current_roster_service.py:161,181`** (24 failures): `CASE WHEN :expected_team IS NOT NULL …` raises `AmbiguousParameter` on PostgreSQL/psycopg3. Roster resolution by player name fails, which breaks the native A1/C0 persistence and lineup-refresh paths. Fix: `CAST(:expected_team AS TEXT)`.
- **`oddspapi_service.py:643`** (2 failures): it compares the `scraped_at` timestamp with `.isoformat()`, a string. PostgreSQL rejects this with `timestamptz >= varchar`, so the Pinnacle horizon fetch crashes. Fix: compare against the `datetime` itself.
- **`api/routers/tournaments.py:116`** (2 failures): `/tournaments/worlds/simulate` now defaults to `engine="advanced"`. That path:
  - skips participant validation;
  - ignores `play_in_teams`;
  - returns 500 instead of 422 on bad input;
  - runs on hand-typed Elo ratings instead of C0 (see 4.1).
- **`forecast_evidence.py`**: two failures report a missing table, `relation "forecast_evidence_snapshots" does not exist`. Tests that create their schema from the ORM do not get the table that the migration chain creates. This is schema drift between the ORM and the migrations.
- **`test_tournaments.py::test_audit_api_rejects_impossible_manual_winner`**: fails with `RuntimeError: Unresolved or colliding participants (G2, G2)` before it reaches the expected rejection.

The remaining ~50 failures need artifacts that are not on this machine: `canonical_benchmark`, `research_root.txt`, `series_results.csv`, `48h_betting_benchmark_results.json`, `source_contracts.json` and `c0_distilled/manifest.json`. Many tests therefore depend on untracked local files, so this suite cannot pass on a fresh clone.

### 1.2 Static defects on reachable paths

- `forecast_evidence.py:797` uses an undefined `UTC`, so every forward-admitted C0 study opportunity raises `NameError`.
- `mapping_service.py:206` uses an undefined `rows`. `GET /matches/golgg-teams` returns 500 when `golgg_teams` is empty and the JSON fallback file exists.
- `core/db.py:150` references `pd` without importing it; only the annotation string is affected.
- `refresh_golgg_direct.py:97` contains an f-string that is invalid on Python 3.11, while `pyproject` declares `>=3.11`.
- `mapping_service.py` and `canonical_match_service.py` contain duplicate alias keys, so the later value silently wins.

## 2. Betting decision path: what the app does versus what the docs say

This was traced end to end: scheduler → `run_upcoming_prediction_pipeline --operational-hybrid` → `generate_model_ev_signals` → `scan_and_dispatch_ev_alerts`.

| Item | Docs (README §2.2/§5.4.2, MODEL_CATALOG) | Running code |
|---|---|---|
| Shrinkage | 0.35·C0 + 0.65·LOO opening consensus | **0.50/0.50** (`registry.py:449`), applied once |
| Market consensus | Leave-one-out, Shin, at scrape time | **Mean of multiplicative de-vig over all fresh books, focal book included, no minimum book count** (`prematch_quote_service.py:134`) |
| Divergence gate | ∣logit p_C0 − logit q∣ ≤ 0.85 | Applied to the **hybrid** p (`bet_qualification_service.py:459`), so the effective C0 bound is ~1.70 logits |
| Max EV 25%, Min EV 5%, EXP-039 veto | "Permanently retired" | **All active** (`upcoming_inference_service.py:2933`) |
| EXP-039 veto | Model consensus check | Reads `base_prob_a`, a key the hybrid never writes. It falls back to the hybrid p and uses stale thesis rows, so it is effectively inactive |
| Staking | 1/4 Kelly, 150 PLN cap | Bankroll defaults to a **fictitious 100 PLN** (`run_upcoming_prediction_pipeline.py:85`, never overridden by the scheduler) minus *real* open stakes. Suggestions are 0 once open stakes exceed 100. Alerts print the PLN amount as "% bankrolla" (`alert_service.py:179`) |
| Dashboard/alert re-qualification | Same gates | `qualify_prediction_sides` has no DRO gate, no odds ceiling and no max EV, so the UI can recommend bets that the signal pipeline quarantines. Tax is hard-coded at 0.12/0.88 in `alert_service.py:354,412` |

**Consequence:** with one admitted book, the hybrid shrinks C0 toward the very price being bet. EV > 5% then means "C0 strongly disagrees with this book". That is exactly the selection pattern earlier audits found overstated by 30–45 pp.

**Verified correct:**
- EV is `p·odds·(1−tax)−1` everywhere, matching the Polish stake tax.
- Display and gate use the same probability under C0.
- Side alignment and p_a + p_b ≈ 1 are enforced.
- Quotes are timezone-aware, ≤300 s old and before the start.
- Wallet settlement is Decimal, conditional-update idempotent and atomic.

## 3. Models: what can actually be served

- **The app serves only the legacy 17-token C0** (`c0-native-2026-w32-e12-v1`, history16). Nothing in `betting_app` builds 18- or 20-token tensors.
- `C0_NATIVE.metadata` carries the 18-token metrics, so the legacy artifact displays numbers it did not produce.
- `C0_DISTILLED` points at `data/06_models/c0_distilled/manifest.json`. That file is not tracked by DVC (`dvc.yaml`, `dvc.lock`, `.dvc`) and not committed. `C0_DISTILLED_20T` points at a reporting JSON, not a model.
- **"C0-D" in `predict_matchup.py` is legacy C0 with a hard-coded logit slope** of 1.004/1.008/1.012 by BO (`predict_matchup.py:35`, `registry.py`). Setting `ACTIVE_PREDICTION_MODEL=C0-D` passes the operational check and labels C0 output as C0-D.

### 3.1 Evaluation validity of the new model family

1. **The cross-region normalizer looks ahead (critical).** `simulation_pipeline.py:529` applies one *current* `regional_offsets_state.json` (latest `rating_runs` row, `live_native_feature_builder.py:460`) to every row from 2024 to 2026. October 2026 offsets therefore score the international events they were fitted on. The scaling constants are hand-tuned, and a test asserts agreement with Worlds 2026 market favourites. The "0.533600 with Cross-Region Norm" result and the international-slice gain (0.573→0.508) are not causal.
2. The C0-D "Norm" figure transplants a logit shift measured on legacy C0 onto C0-D-20T probabilities. Rows are aligned by position, not by ID (`simulation_pipeline.py:576–593`). The README gives two numbers for the same model, 0.533779 and 0.533600.
3. **No code computes the "6 replacement guards".** The evaluation scripts emit only point LL/Brier per slice: no slice intervals, no 0.005 threshold, no A0 comparison. "PASS (All 6)" is not backed by executed code.
4. **Multiple comparisons.** More than 20 variants were scored on the same 3,103 development-exposed rows without the 98.33% Bonferroni correction used by earlier protocols. The 18-token C0 interval [−0.001195, +0.000039] crosses zero, yet that model is "ADOPTED_AS_STANDARD".
5. The C0-D interval in RESEARCH.md, [−0.002232, **−**0.000396], appears to be the Track 1 interval [−0.002232, **+**0.000396] with its sign flipped. This needs checking against the source JSON.
6. Provenance gaps:
   - The 365-day anchor bank builder is not in the repo, so its causality cannot be verified.
   - The 90/180-day builder is causal only if `metadata.parquet` is chronologically ordered, which is not asserted.
   - The 20T script reads `year_anchor_distilled.parquet`, which no script writes; the 18T script writes `…_predictions.parquet`.

**Correct here:**
- The distillation teacher uses TRAIN-only checkpoints, and the student never sees CAL or 2026 teacher logits.
- Calibration is fitted on CAL.
- ONS recalibration is causal.
- The bootstrap is paired, 9-month-blocked, 5,000 draws, with consistent clipping.
- Cold-start and side-swap checks exist.

## 4. Tournament simulators and the CLI

1. **The default Worlds and bracket API paths use no model at all.** They run on `DEFAULT_WORLDS_RATINGS`, hand-typed Elo, anchored to Glicko built from the same numbers.
   - Missing teams (Invictus Gaming, Movistar KOI) silently get 1500.
   - Missing pairs silently fall back to Elo with no coverage report.
2. **The Fearless engine breaks series probabilities and side symmetry.** `fearless_series_engine.py:392` inverts series→map, then adds a ±0.1242-logit blue-side bonus. Team A is always blue in map 1, and the loser takes blue afterwards. With depletion at zero, a 0.50 input becomes 0.531 (BO1), 0.516 (BO3) and 0.512 (BO5), so p(A,B)+p(B,A)=1.062 in BO1. It is enabled by default, so every Swiss and knockout result carries a "listed first" bias. It also invents side and draft, which are unknown pre-match.
3. The "C0-D (Norm)" Worlds table actually loads the legacy control ensemble. Its cache is never invalidated, and `decision_iso` is hard-coded to 2026-10-21. If the round-1 snapshots were built after round 1 was played, they include round-1 results. This could not be checked here because the artifacts are absent.
4. `--live-db` is not live. With no cutoff it writes `extended_history_latest.json`, and returns that cache forever on later runs. On failure it silently falls back to the baseline.
5. **The live feature builder turns missing stats into 0.0** (`live_native_feature_builder.py:73`, asserted by a test). The frozen bank keeps these values masked, and CSD/XPD@15 are only about 65% present.
   - Games are sorted by `game_id` as strings.
   - The cutoff uses the played date, not the release day.
   - The CLI help says "inclusive" while the code is exclusive.
6. Shared mutable module state (`_PLAYER_POOL_REGISTRY` and others) makes the same seed give different results after any `/crystal-ball` call.
7. The momentum update adds `0.03·map_margin`, which has positive drift for favourites. "Hybrid exact" freezes momentum after Swiss while Monte Carlo keeps updating it, so the two are different models.
8. Mass checks renormalize after simulation (`:964`, `:1323`), so they cannot fail. The underlying knockout DP does conserve mass on its own.
9. `predict_matchup` prints "ADOPTED BET RECOMMENDATION" using a partial gate set, additive de-vig and no thesis, BO1 or longshot rules. It is not the app's policy.
10. Other issues:
    - A 2-tuple key in `get_matchup_probability` is treated as a map probability, which projects BO twice for series tables keyed `(a,b)`.
    - Already-played series are recorded as sweeps.
    - The fallback Swiss draw allows rematches.
    - The Swiss lottery is not uniform over valid matchings: 0.022–0.038 against 0.032 in an 8-team, round-3 pool.

**Correct here:**
- The knockout DP sums to 1.
- QF pairing puts 3-0 teams against 3-2 teams in opposite halves.
- Swiss always yields 2/3/3 qualifiers.
- The normalizer sign flips correctly under side swap.
- Calibration slopes are applied on the logit.

## 5. Financial claims (README §5.3–5.4, RESEARCH 2026-10-17/20)

All of the following were checked in code.

1. **Gate parameters were picked in-sample.** `optimize_clv_vs_optimizers_curse.py:734–805` runs a 90-cell grid over α, ε, Guard B and max odds on the production cohort.
   - It keeps cells with `B_hyb < Net CLV` and `B_DRO ≤ 0`, then selects by maximum CLV profit.
   - The "verification" cohort B is a superset of cohort A (`cohort_b = all_matches_df.copy()`).
   - So "B = +3.76 pp < Net CLV 4.83%" and "B_DRO ≤ 0" are selection criteria, not findings.
2. **The 43-bet C0-D ledger re-filters the 50 bets already chosen by C0 + EXP-039.** It uses a probability shift, stale taken odds and known outcomes. "No EXP-039" is therefore false in substance, and +27.82% is survivorship within a hand-picked set.
3. **Ledger bets are joined to matches by nearest probability within ±3 days, with no tolerance.** The side is taken from whichever complement happened to be closer, so wrong matches or flipped sides are possible.
4. **The "α = 0.65 zero-lookahead LOO" evaluation of the 50 bets has no generating code.**
   - No script writes `optimal_strategy_simulation_ledger.csv`.
   - The Oct-12 report and `DATA_CATALOG` say it was built with α = 0.50.
   - Scripts invert it inconsistently, some with 0.65/0.35 and others with 0.50/0.50.
5. **The 2-hour-bucket q_open_loo is the latest price, not the opening one.**
   - It uses quotes up to 2 h after the nominal decision time.
   - A market-only prefilter (`0 ≤ q_loo·odds − 1 ≤ 0.38`) keeps only prices already better than other books' fair odds, before the model is consulted. That is stale-line shopping, and it manufactures raw CLV on its own.
6. **A missing closing line defaults to p_close = 0.5** (`optimize_clv…:256`, `run_walk_forward_prod_db…:302,378`). At odds 3.0 that adds +32% "CLV" from nothing.
7. **The Pinnacle and multi-year cohorts bet at de-vigged fair odds** (`o = 1/q`), with the same q used as the CLV reference. Net CLV is therefore 0% or −12% by construction, and the README's Pinnacle Net CLV rows cannot come from repo code.
8. The Polish "T−1h" bets in the 1,467-series cohort are placed at the best *closing* odds and scored against the mean of those same prices, which is positive by construction.
9. **The multi-year "3,002 bets, +7.60%, Net CLV +6.35%" row is a literal** (`simulation_pipeline.py:1456–1519`), as are the 454- and 147-bet rows. The only multi-year code uses EXP-039 with missing values set to 0.5 and closing averages as both price and reference. C0 TRAIN covers 2020 to 2025-07, so any C0 scoring of 2021–2025 would be in-sample anyway.
10. **"99% Guaranteed Profit at 2,792 bets"** comes from Φ(μ/(σ/√N)), with μ = 4.83% treated as known and σ hard-coded at 1.0979.
    - At the CI lower bound of 1.68%, about 11,600 bets would be needed.
    - It assumes CLV converts 1:1 into post-tax yield at stale prices that stay available.
    - It is a conditional sampling probability, not a guarantee.
11. Inference is per-bet IID: a t-test and a bet-level bootstrap on 50 bets bunched in July–August. That contradicts the monthly-block rule in AGENTS §4.
12. "Sharpe 1.56" is mean/sd·√N of per-bet returns, a t-statistic (p ≈ 0.13 at N = 43), not an annualized Sharpe ratio. Max drawdown is bounded by construction, since stakes are capped at 150 on a 10,000 bankroll.
13. "Zero-Disappointment Guarantee" is arithmetic. B_DRO = B_hyb − ε·c·Ō ≈ B_hyb − 5.7 pp, so any B_hyb < 5.7 pp passes. It is a stake-weighted mean, not a per-bet bound.
14. The trajectory "CLV" in `historical_production_simulator.py:327` is measured against the focal quote's own Shin de-vig, not the close. Research Shin (closed form, `betting_guards_a_to_d.py:84`) also differs from app Shin (bisection, `shin_devig.py`), and the live app uses neither.
15. `run_c0_distilled_financial_simulations.py:807` and `generate_marginal_blocked_bets_audit.py:458` infer side orientation from the outcome, which bypasses the documented identity joins.

None of this meets the project's own ±2 pp predicted-minus-realized reliability gate. None of it is reconciled with the Sept-30 audits, which found 30–45 pp EV overstatement and negative ROI. The items above are enough to explain the difference.

## 6. Comparison with best-known solutions (web research)

| Topic | Best-known practice | This project | Gap |
|---|---|---|---|
| Pre-match LoL prediction | Guan & Tung 2026 [1]: team-level paired comparison plus first-pick draft; per-map LL 0.637, Brier 0.223; **statistically indistinguishable from Polymarket** on per-game contracts, with the market ahead at Worlds and deciders. Tennis [2]: market LL 0.55 against 0.59 for the best of 11 models. No public source shows a pre-match LoL model beating the market. | Series LL ~0.534 (not comparable to per-map numbers). It trails the Polish consensus by ~0.006 LL, and the interval crosses 0. Inputs (player histories plus anchors) are richer than [1]. | **At par.** Also report per-map LL, Brier and calibration slope for a direct comparison with [1]. Do not expect LL gains to beat the market. |
| Proving an edge | CLV against margin-free Pinnacle close: per-bet SD ≈ 0.08–0.11, so a 2% edge separates from zero in ~50 bets. Yield SD ≈ 1 unit per bet, so 3% ROI needs ~4,300 bets and 5% needs ~1,500 [10]. Deflated Sharpe and PBO correct best-of-N selection [11][12]. Soft books limit winners within months [13]. | Two simulated ledgers of ~50 bets, gates chosen in-sample, IID tests, CLV reference inconsistent or by construction, "99% guaranteed" extrapolation. | **Far below standard.** Pre-register the rules, run a locked prospective paper ledger, and measure CLV against Pinnacle's fair close. Apply DSR/PBO to the grid search. Retire the guarantee table. |
| Staking | Baker & McHale shrunk Kelly under parameter uncertainty [17]. Half Kelly cuts the 50%-drawdown risk from 1/2 to 1/8 [18]. Real DRO-Kelly optimizes over an ambiguity set [20][21]. | The 1/4 Kelly plus cap is conservative in form. The fixed 2.5 pp haircut is not Wasserstein DRO in the literature's sense. The live app sizes from a 100 PLN placeholder. | Fit the shrinkage from out-of-sample calibration error, wire in the real wallet balance, and rename the gate. |
| Polish tax | Stake tax of 12% (Act on gambling, Art. 73–74 [22]). Break-even p = 1/(0.88·odds), +6.8 pp at odds 2.0. | Applied correctly in the EV formula. | No gap in the formula. It is the main reason a market-parity model cannot profit at soft books. |
| De-vig / blending | The power method is best or tied with Shin; methods converge at low margins [23][24]. Profit comes from **decorrelating** from the bookmaker, not averaging toward it [27][28]. | App: multiplicative, focal book included. Research: two different Shin implementations. Shrink-then-bet selects residual disagreements. | Use one de-vig method everywhere, with real LOO and ≥2 books. Fit α out of time. |
| Production practice | Shadow/champion–challenger rollout with preset gates [29][30], point-in-time feature joins [31], ML Test Score [33], versioned artifacts. | DVC and hash pinning are good. Pinned C0-D artifacts are missing. The live builder breaks the mask contract. There is no locked prospective ledger. Tests depend on untracked local files. | Freeze an artifact hash before a real event and run it in shadow. Make the test suite pass on a fresh clone. |

Evidence caveats: GRID and PandaScore publish no accuracy figures. Bayes Esports figures are marketing. The bet-count thresholds are arithmetic from [10]'s SD figures, not a published table.

## 7. What is left to fix, in priority order

1. **Correct the documentation first.** Withdraw or label as unverified: the 50/43-bet yields, Net CLV, "Zero-Disappointment Guarantee", "99% Guaranteed Profit", Sharpe, the multi-year 3,002-bet row, the Pinnacle Net CLV rows, the "6 guards PASS" claims and the 0.533600 normalized LL. Restore the AGENTS rule that a model or strategy is adopted only with a held-out or prospective result. Then update AGENTS.md: its handoff is dated 2026-10-07 and does not mention the C0-D work.
2. **Fix the live betting path** (section 2):
   - real LOO consensus with ≥2 books and one de-vig method;
   - apply the divergence gate to p_C0;
   - remove or document the "retired" gates;
   - fix the EXP-039 key;
   - pass the real wallet balance to Kelly;
   - make the UI and alerts use the same gate set and `cfg.tax_rate`.
3. **Fix the five code bugs the tests found:** roster SQL cast, OddsPapi timestamp comparison, Worlds route validation and default engine, the ORM/migration drift for `forecast_evidence_snapshots`, and the G2–G2 collision. Also fix the `UTC` and `rows` undefined names.
4. **Simulator:**
   - default to the plain series path, or make Fearless preserve the input series probability and randomize map-1 side;
   - drop hand-typed Elo as the API default, or fail loudly on missing coverage;
   - remove the post-hoc renormalization;
   - make momentum zero-mean;
   - isolate module state.
5. **Model work:**
   - rebuild the normalizer as-of each cutoff and refit its constants on CAL;
   - implement the 6 guards in code, with intervals and multiplicity correction;
   - commit the 365-day anchor builder and assert row chronology;
   - join predictions by ID;
   - if C0-D is wanted in the app, build a real 18/20-token serving path and DVC-track the manifest, and remove the slope surrogate.
6. **Live feature builder:** keep nulls masked, sort `game_id` numerically, apply the release-day cutoff, and invalidate the cache.
7. **Evidence:** pre-register one rule set and run a locked forward paper ledger with forecast, quote and clock logged per bet. Score CLV against Pinnacle's fair close, with monthly-block intervals. Treat yield as uninformative until roughly 1,500–4,000 bets.
8. **Hygiene:** make the suite pass on a fresh clone (fixtures, or skip when an artifact is absent), and add `ruff` and CI.

## Sources

1. Guan & Tung, "Pre-game paired-comparison modeling of professional League of Legends map outcomes", arXiv 2609.08060, 2026-09 — https://arxiv.org/abs/2609.08060
2. Kovalchik, "Searching for the GOAT of tennis win prediction", JQAS 2016 — https://vuir.vu.edu.au/34652/1/jqas-2015-0059.pdf
3. De Bois et al., "PandaSkill", arXiv 2501.10049, 2025 (vendor researchers) — https://arxiv.org/abs/2501.10049
4. Joshi, "OpenSkill", arXiv 2401.05451, 2024 — https://arxiv.org/abs/2401.05451
5. "DraftRec", arXiv 2204.12750, 2022 — https://arxiv.org/abs/2204.12750
6. Bayes Esports via iGaming Business (vendor) — https://igamingbusiness.com/?p=88035
7. Bayes Esports press release, 2023 (vendor) — https://bayesesports.com/press-releases/bayes-esports-offering-probabilities-as-a-service-improving-odds-margins-and-esports-betting-offerings-of-sportsbooks
8. Jalovaara, Aalto MSc presentation, 2024 — https://sal.aalto.fi/files/seminar/2024-08-26-perttu-jalovaara.pdf
9. Choi et al., "Rethinking Evaluation Metric for Probability Estimation Models Using Esports Data", arXiv 2309.06248 — https://arxiv.org/abs/2309.06248
10. Buchdahl, "Using the closing line to test your skill in betting", Pinnacle Betting Resources (copy), 2019 — https://www.sportstradingnetwork.com/article/using-the-closing-line-to-test-your-skill-in-betting/
11. Bailey & López de Prado, "The Deflated Sharpe Ratio", 2014 — https://papers.ssrn.com/abstract=2460551
12. Bailey, Borwein, López de Prado & Zhu, "The Probability of Backtest Overfitting", 2017 — https://papers.ssrn.com/abstract=2326253
13. Kaunitz, Zhong & Kreiner, "Beating the bookies with their own numbers", arXiv 1710.02824, 2017 — https://arxiv.org/abs/1710.02824
17. Baker & McHale, "Optimal Betting Under Parameter Uncertainty", Decision Analysis 2013 — https://www.semanticscholar.org/paper/d03cda6e9aec9a6674047b1c093780fdc2bf2d56
18. MacLean, Thorp & Ziemba, "Good and bad properties of the Kelly criterion" — https://www.stat.berkeley.edu/~aldous/157/Papers/Good_Bad_Kelly.pdf
20. Sun & Boyd, "Distributional Robust Kelly Gambling", arXiv 1812.10371 — https://arxiv.org/abs/1812.10371
21. Li, "Wasserstein-Kelly Portfolios", arXiv 2302.13979 — https://arxiv.org/abs/2302.13979
22. Ustawa o grach hazardowych, Art. 73–74 (Dz.U. 2020 poz. 2094) — https://www.infor.pl/akt-prawny/DZU.2020.246.0002094,ustawa-o-grach-hazardowych.html
23. Clarke, Kovalchik & Ingram, "Adjusting Bookmaker's Odds to Allow for Overround", 2017 — https://outlier.bet/wp-content/uploads/2023/08/2017-clarke-adjusting_bookmakers_odds.pdf
24. Štrumbelj, "On determining probability forecasts from betting odds", IJF 2014 — https://www.researchgate.net/publication/264349990_On_determining_probability_forecasts_from_betting_odds
27. Hubáček, Šourek & Železný, "Exploiting sports-betting market using machine learning", IJF 2019 — https://ida.fel.cvut.cz/papers/hubacek2019exploiting.html
28. Hubáček & Šír, "Beating the market with a bad predictive model", arXiv 2010.12508 — https://arxiv.org/abs/2010.12508
29. Atlan, "Shadow Deployment for ML Models" (vendor blog) — https://atlan.com/know/shadow-deployment-for-ml-models/
30. DataExpert, "Champion/challenger guide" (blog) — https://dataexpert.io/blog/champion-challenger-guide-lakehouse-ml
31. Microsoft Learn, "Point-in-time join" — https://learn.microsoft.com/en-us/azure/machine-learning/offline-retrieval-point-in-time-join-concepts?view=azureml-api-2
33. Breck et al., "The ML Test Score", IEEE Big Data 2017 — https://research.google/pubs/pub46555/

Source numbering follows the research digest; unused numbers are omitted.
