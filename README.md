# evolutive-trading

Evolve rule-based trading strategies against many possible futures instead of one backtest. Simulation only: no real money.

[Kronos](https://github.com/shiyu-coder/Kronos) is a foundation model for financial candlesticks. Sampled repeatedly from the same starting point, it produces different, plausible price paths. evolutive-trading uses those paths as the environment: a population of rule-based strategies trades a fresh batch of futures every generation, the strongest survive and are mutated, and the rest are dropped.

## How it will work

1. **Strategies** are small rule sets described by a few numbers, such as a momentum window, how strong a Kronos forecast must be to act on it, a stop-loss or a position size. Evolution changes the numbers, not the code.
2. **Futures** are sampled from Kronos once, on a local machine, starting from real market history, and stored as files. Generating them needs torch; everything else does not.
3. **Evolution** puts every strategy in a generation through the same batch of futures, with trading costs, then keeps the fittest and fills the population with mutated copies of them.
4. **Reality check**: the champion is compared with buy-and-hold on real price history it never saw during evolution, then forward-tested on an Alpaca paper account with `trader.py`.

It starts with a single instrument (SPY). Kronos forecasts each series on its own, so synthetic futures for several stocks lose the way they move together, and a multi-stock strategy would look safer than it is.

## Honest expectations

Evolution optimises whatever it is given, including the quirks of Kronos's generated prices. The likely result is a champion that looks strong on synthetic futures and falls back towards buy-and-hold on real data. The point of the tool is to make that gap visible. A strategy that trades on Kronos forecasts has an unfair edge on Kronos-generated futures, so it is only ever judged on real data.

## Status

The evolution side is not built yet. What exists is the forward-test end: `trader.py`, a daily loop where Kronos forecasts tomorrow's close for a 20-stock watchlist, the top 5 positive forecasts are held in equal weight, and orders go to an Alpaca **paper** account. Every forecast is appended to `predictions.csv`.

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
