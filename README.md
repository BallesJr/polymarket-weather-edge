# POLYMARKET WEATHER EDGE BOT

This project builds an automated paper trading bot that exploits a systematic mispricing in Polymarket's daily temperature markets. I noticed that the market consistently overprices extreme temperature outcomes (YES side), so I built a pipeline that combines live METAR observations from each market's resolution station and Open-Meteo forecasts with a Gaussian probability model to trade the NO side. The live entry rule is an explicit NO-price band validated on historical trade buckets; every trade records the full set of weather features so a Random Forest calibrator can be retrained on clean data and reintroduced as a gate.

---

## WHAT I WORKED ON

- **Weather market pipeline**: Connected to Polymarket's Gamma API to fetch active daily temperature contracts and parsed each market to extract city, date, and temperature range from the question text, handling both °C and °F market formats with unit-aware bin corrections.
- **Resolution-source alignment**: Audited every market's resolution station against its live description (catching that Paris resolves at Le Bourget, not CDG) and validated data sources against the actual resolution of 500+ historical trades: daily highs computed from station METARs agree with market resolutions 93% of the time vs 66% for the ERA5 reanalysis grid. Same-day observations now read the exact resolution station's METARs (aviationweather.gov), with Open-Meteo as fallback, so the bot trades on the same thermometer the market settles on.
- **Probabilistic model**: Built a Gaussian engine that converts temperature forecasts into outcome probabilities, with tighter sigma on same-day observations and wider sigma on multi-day forecasts.
- **RF calibration (and a lesson in inference consistency)**: Trained a Random Forest on historical resolved trades to refine the raw Gaussian probabilities. A later audit caught that live inference fed the model mirrored features (YES price where training saw the NO entry price), saturating its output near 0.5: the deployed system had silently degenerated into a pure price-band rule. I made that band rule explicit, pulled the RF out of the live path, and rebuilt training to use only clean-regime trades with a persisted feature mapping, gated on a minimum sample count before any model is deployed again.
- **Signal filtering**: Applied a data-driven filter stack based on historical win rates: BUY_YES (11%) and T+1 (18%), T+2 (4%) were all disabled. Only BUY_NO T+0 trades are opened, inside a NO-token price band [0.15, 0.40]: bucket analysis showed that betting against near-certain same-day markets (NO price < 0.10) is pure adverse selection with a ~0% win rate, while the 0.20-0.35 zone won 62%. Entry prices are re-checked against the live CLOB midpoint before opening. A 60-day rolling city win-rate filter blocks underperforming cities automatically.
- **Paper portfolio**: Built a persistent portfolio that opens positions, resolves them via the CLOB API, tracks P&L with fee deduction, and survives bot restarts by persisting state to disk.
- **Telegram notifications**: Real-time alerts for cycle summaries, high-edge openings (edge ≥ 0.40), and position resolutions.
- **Automation**: The bot runs every 30 minutes via GitHub Actions; the RF model retrains every Sunday at 04:00 UTC on fresh trade data.

---

## PROJECT STRUCTURE

- `market_filter.py`: Fetches and parses active weather markets from the Polymarket Gamma API.
- `weather_api.py`: Fetches station METAR observations (aviationweather.gov), Open-Meteo forecasts, and IEM daily summaries for resolution auditing; computes Gaussian outcome probabilities per contract.
- `calibration_weather.py`: Trains and evaluates the Random Forest calibration model on historical resolved trades.
- `signal_engine.py`: Generates ranked trading signals; applies edge, direction, horizon, and city-performance filters.
- `paper_trader.py`: Paper portfolio manager; opens positions, resolves via the CLOB API, persists state to `data/paper_portfolio_weather.json`.
- `notifier.py`: Sends Telegram notifications for cycle events and high-edge trades.
- `backtester.py`: Portfolio analytics (P&L breakdown, win rate by segment, and open position summaries).
- `.github/workflows/bot_cycle.yml`: Runs `paper_trader.py` every 30 minutes on GitHub Actions.
- `.github/workflows/retrain.yml`: Retrains the RF model every Sunday at 04:00 UTC and commits the updated model.

---

## CONFIGURATION

| Parameter | Value |
|---|---|
| Min edge (T+0) | 0.25 net of fees |
| Directions enabled | BUY_NO only |
| Horizons enabled | T+0 only |
| NO-token price band | [0.15, 0.40] |
| City tail-guard | binomial test vs per-city break-even, p < 0.05 over last 60 days (clean-data regime only) |
| Kelly fraction | Half-Kelly, capped at $5 |

---

## PAPER TRADING RESULTS

The bot is currently live in paper trading mode. Live portfolio state (bankroll, P&L, win rate, and open positions) is persisted to `data/paper_portfolio_weather.json` and updated after every cycle.

The current filter stack (edge ≥ 0.25, BUY_NO only, T+0 only, NO price band) was applied in June 2026. Trades opened before 2026-06-10 ran under since-fixed data bugs (wrong Paris station, unparsed Fahrenheit markets, model-interpolated observations), so out-of-sample validation only counts trades from that date onward. Forward performance under the clean regime is still accumulating.

---

## REQUIREMENTS

`pip install requests pandas numpy scipy scikit-learn`

---

## EXECUTION

```bash
python paper_trader.py                # run one full cycle
python paper_trader.py --status       # print current portfolio
python paper_trader.py --resolve-only # check resolutions only
python calibration_weather.py         # retrain RF model
python backtester.py                  # portfolio analytics
```

---

## LIMITATIONS

**Gaussian model**: Assumes daily maximum temperatures are normally distributed. Heavy-tail events like heat waves are underestimated, so the model likely underprices YES outcomes in extreme conditions.

**Forecast quality**: Open-Meteo is the only forecast source (observations now come from station METARs). Forecast accuracy degrades significantly beyond T+0, which is the main reason T+1 and T+2 were disabled. A multi-source ensemble would help.

**Station coverage**: New Zealand stations are not in the IEM archive, so Wellington's resolution audit falls back to the less reliable ERA5 grid.

**City tail-guard latency**: A city is only blocked once its record is statistically incompatible with break-even (binomial test, p < 0.05), which takes ~9 straight losses at current entry prices — a genuinely bad city can lose ~$45 before exclusion. This is deliberate: with the strategy's natural win rate (~32%) barely above break-even (~31%), any harder cutoff blocks cities on noise (the earlier ≥35% win-rate rule had excluded 30 of 42 cities within a month, 11 of them profitable).

**No live execution**: Still paper trading. Real execution requires Polymarket CLOB v2 API access and would introduce slippage and partial fills not modelled here.

**Fee model**: The fee is calculated as `feeRate × size × (1 − entry_prob)`, which is a simplification. Real fees may differ for illiquid markets.

---

## BACKGROUND

I built this as a live extension of the [Polymarket Edge Model](../polymarket_edge_model/), which first identified the longshot bias in Polymarket's resolved market data. Weather temperature markets are a good fit because outcome probabilities can be modelled independently using real meteorological data, without relying on external prediction platforms.
