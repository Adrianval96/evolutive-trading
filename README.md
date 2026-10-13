# evolutive-trading

Trading agents in a simulated market that moves forward one day at a time. Simulation only: no real money.

Each day, every agent sees the price history up to that day and decides how much of its money to hold. Orders fill at the next day's open, with trading costs. Each agent has its own account and is scored against buy-and-hold on the same days. Agents are price-takers: their trades never move the market they trade in.

## Plan

1. **Backtest harness.** Agents trade a daily replay of one instrument (SPY) forward in time: buy-and-hold and moving-average crossover agents, ending in a leaderboard of return, drawdown and trades.
2. **Walk-forward agents.** Agents that re-tune their own settings on the recent past as time moves, then trade the period that follows. Does adapting beat a fixed rule?
3. **A Kronos agent.** [Kronos](https://github.com/shiyu-coder/Kronos), a foundation model for financial candlesticks, as an agent's brain, judged only on data from after its training cutoff.

## Honest expectations

Simple technical rules have a long record of not beating buy-and-hold once costs are counted, and searching over many settings manufactures good-looking backtests on its own. The harness exists to measure this honestly: no look-ahead, costs on every trade, buy-and-hold on the same bars.

## Status

The harness is not built yet. What exists is the forward-test end: `trader.py`, a daily loop where Kronos forecasts tomorrow's close for a 20-stock watchlist, the top 5 positive forecasts are held in equal weight, and orders go to an Alpaca **paper** account. Every forecast is appended to `predictions.csv`.

## Setup

1. Create a free Alpaca account, switch to **Paper Trading**, and generate API keys.
2. Install:

   ```bash
   git clone --recurse-submodules https://github.com/Adrianval96/evolutive-trading.git
   cd evolutive-trading
   python3.12 -m venv .venv.nosync
   .venv.nosync/bin/pip install -r requirements.txt
   cp .env.example .env
   ```

3. Paste the paper keys into `.env`.

`.venv.nosync` keeps the ~1 GB of torch out of iCloud. Python 3.12 because Kronos pins `pandas==2.2.2`, which has no 3.13 wheels.

## Run the forward test

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
30 23 * * 1-5 cd ~/Documents/projects/evolutive-trading && set -a && . ./.env && .venv.nosync/bin/python trader.py >> trader.log 2>&1
```

macOS blocks cron from reading `~/Documents` unless `/usr/sbin/cron` has Full Disk Access. The Mac must also be awake at that time.

## Known simplifications in `trader.py`

- Re-running is safe: every run cancels open orders before rebalancing.
- The paper account is assumed to belong to the bot. Any position not in today's picks is closed, including ones you opened by hand.
- Forecast is close→close, but orders fill at the next open.
- Kept positions are not resized; only new buys get `equity / 5`.
- No transaction-cost or slippage modelling; paper fills are optimistic.
