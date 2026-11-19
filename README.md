[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/naylafeb/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/naylafeb/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast — make.fun entry

Forecast, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn
rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles (5/25/50/75/
95%).

## Entry

`main.py` (Python 3.13, standard library only). It reads one case
`/input/<id>.in` and writes `/output/<id>.out` with five quantiles of
each trophy's earn rate at the case's target.

## Model

A rate is earners divided by players tracked, and the two move for
different reasons, so they are forecast separately and divided at the
end:

* **Earners** grow as players play and reach each trophy. In log time
  (hours since release, 2026-11-19T00:00:00Z), log earners is roughly
  linear; each trophy gets its own least-squares line, shrunk toward a
  prior slope that rises with how widely it is already earned.
* **Players tracked** grows in steps at Exophase refreshes; between
  refreshes the page is frozen. Log players is likewise roughly linear
  in log time (638 -> 3,868 in the first 4.5 hours after release).

The median rate is 100 * earners(target) / players(target). Forecasting
the two separately is what makes the entry dilution-aware: a refresh
that adds thousands of newly tracked players who have not earned much
yet pushes every rate down, even though nobody un-earned a trophy. A
persistence forecast (last rate, narrow band) misses that step by ~1.8
logit units on the practice.

Quantiles are log-normal around the median in logit space. The spread
combines the fit residuals with the size of the move the forecast
itself makes and a term that grows with the horizon, floored so the
band never collapses. A flat history (no refresh seen yet) gets a wide
band, not a confident one.

## Practice

Practice cases p00/p01 (checkpoints 03:00/03:30 UTC 19 Nov, targets
04:30/05:00 UTC) straddle Exophase's first refresh of the game's
counts (638 -> 3,868 players tracked). Scored locally against the
refresh reconstructed from the known player counts and the live page
(the practice answers themselves are server-side; the house baseline
is stated at about 1.51 in the rules, and this reconstruction puts it
at 1.52):

| case | this entry | house baseline |
|------|-----------|----------------|
| p00  | 0.084     | 1.523          |
| p01  | 0.099     | 1.523          |

Mean pinball loss on the logit scale; lower is better. The entry's
typical miss is ~0.01 points with ~0.98 90% coverage on the
reconstruction. Runs in ~0.06 s per case, deterministic.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
