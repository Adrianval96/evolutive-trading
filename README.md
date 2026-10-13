# kronos-paper-trader

Daily paper-trading loop: [Kronos](https://github.com/shiyu-coder/Kronos) forecasts tomorrow's close for a 20-stock watchlist, the top 5 positive forecasts are held in equal weight, and orders go to an Alpaca **paper** account. Every forecast is appended to `predictions.csv` so accuracy can be studied later.

## Setup

1. Create a free Alpaca account, switch to **Paper Trading**, and generate API keys.
2. Install:

   ```bash
   git clone --recurse-submodules <this repo>
   cd kronos-paper-trader
   python3.12 -m venv .venv.nosync
   .venv.nosync/bin/pip install -r requirements.txt
   cp .env.example .env
   ```

3. Paste the paper keys into `.env`.

`.venv.nosync` keeps the ~1 GB of torch out of iCloud. Python 3.12 because Kronos pins `pandas==2.2.2`, which has no 3.13 wheels.

## Run

Outside US market hours (the script refuses while the market is open):

```bash
set -a; . ./.env; set +a
.venv.nosync/bin/python trader.py
```

Compare against SPY:

```bash
.venv.nosync/bin/python trader.py report
```

Test the rebalance logic:

```bash
.venv.nosync/bin/python test_trader.py
```

## Schedule

`crontab -e`, with a local time that lands after 16:30 New York time on weekdays:

```
30 23 * * 1-5 cd ~/Documents/projects/kronos-paper-trader && set -a && . ./.env && .venv.nosync/bin/python trader.py >> trader.log 2>&1
```

macOS blocks cron from reading `~/Documents` unless `/usr/sbin/cron` has Full Disk Access. The Mac must also be awake at that time.

## Known simplifications

- Re-running is safe: every run cancels open orders before rebalancing.
- The paper account is assumed to belong to the bot. Any position not in today's picks is closed, including ones you opened by hand.
- Forecast is close→close, but orders fill at the next open.
- Kept positions are not resized; only new buys get `equity / 5`.
- No transaction-cost or slippage modelling; paper fills are optimistic.

## Later

- Directional hit rate per ticker vs. a "tomorrow = today" baseline, from `predictions.csv`.
- [FinRL](https://github.com/AI4Finance-Foundation/FinRL) for learning a trading *policy* instead of forecasting prices.
