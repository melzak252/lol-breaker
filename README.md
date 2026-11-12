# LoL-Breaker

League of Legends match prediction, tournament research and betting analytics in one repository. LoL-Breaker combines player-history models, historical market analysis and a local application for exploring odds and recording manual decisions. **It does not place bets automatically.**

## What it does

- Predicts pre-match BO1/BO3/BO5 series winners from player histories and regional context.
- Compares probability forecasts against historical models and bookmaker markets.
- Provides research tools for tournament scenarios, map and series props, and betting simulations.
- Offers a FastAPI and React application for odds, model signals, manual bet records and wallets.

## Main model: C20R

**C20R** is the 20-token player-history ensemble with regional adjustment, configured locally as the match-winner model. It outputs whole-series BO1/BO3/BO5 win probabilities without bookmaker odds. See the [model catalog](docs/MODEL_CATALOG.md) and [serving configuration](conf/base/c0_serving.json) for its runtime identity.

The separate Hybrid-50 blends the sports prediction with market consensus. For a focal-book assessment, the app requires at least two other distinct fresh bookmakers.

## Key results

The [four-cell factorial audit](reports/c0_regional_factorial_audit_20261022.md) compares both models on the same **3,083 supported 2026 series**. Of 3,103 requested rows, 20 had no supported regional prediction. Lower log loss and Brier are better.

| Model | Log loss | Brier | Accuracy | AUC |
|---|---:|---:|---:|---:|
| Historical C0 | 0.53695 | 0.18158 | 71.94% | 0.79804 |
| C20R | **0.53302** | **0.18003** | **72.40%** | **0.80165** |

These are development-exposed historical results, not an independent holdout. The combined comparison favors C20R on the reported paired intervals, but BO5 and LPL replacement guards failed; the study decision was **not adopted**. Local configuration is separate from that research decision and does not imply deployment or future performance.

### Historical financial simulations

The 2021–2026 OddsPortal study used **annual fresh20+regional historical forecasts** and a 50/50 blend with the focal book's own no-vig price. These are flat-stake, first-observed-price scenarios—not C20R deployment results, the app's two-other-book hybrid or its full gate stack.

| Scenario | Selected tickets | Known outcomes | Return on known stakes |
|---|---:|---:|---:|
| Zero tax | 1,968 | 1,967 | +13.69% |
| 12% tax, same original tickets | 1,968 | 1,967 | +0.05% |
| 12% tax, requalified selection | 315 | 315 | +22.81% |

The 12% tax applies to the full winning payout. The requalified row changes which tickets are selected, not just their payouts; one unknown outcome leaves whole-portfolio ROI undefined for the first two rows. First observed is not certified opening, historical book activity and fills are unverified, and these development results establish neither true CLV nor an attainable betting edge. See the [project report](docs/PROJECT_REPORT.md) and [aggregate snapshot](docs/results/project_summary.json).

## Reproduce and run

Start with the [reproduction guide](docs/REPRODUCING.md) for dependencies, data access and the supported evidence workflow. Python **3.11 or newer** is required by [project metadata](pyproject.toml). Source data, checkpoints and generated receipts under `data/` are generally not in Git and may require separate access; existing frozen run directories must not be overwritten.

For the local application, see the [application guide](betting_app/README.md). Review its database, secrets and scheduler precautions before starting anything: the default Compose configuration is not a disposable offline setup.

## Project map and documentation

| Area | Start here |
|---|---|
| Overview and detailed results | [Project report](docs/PROJECT_REPORT.md) |
| Reproduction and artifact prerequisites | [Reproduction guide](docs/REPRODUCING.md) |
| Models and local serving identity | [Model catalog](docs/MODEL_CATALOG.md) |
| Architecture and repository entry points | [Architecture](docs/PROJECT_ARCHITECTURE.md) |
| Data, research history and scripts | [Documentation index](docs/README.md) |
| Application and manual workflows | [Application guide](betting_app/README.md) |

## License and disclaimer

Released under the [MIT License](LICENSE). This project is for research and education, not financial advice. Historical simulations do not guarantee future returns.

LoL-Breaker is not endorsed by Riot Games. League of Legends and Riot Games are trademarks or registered trademarks of Riot Games, Inc.
