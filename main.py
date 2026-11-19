"""Forecast GTA VI's PS5 trophy earn rates on Exophase at the case target.

For each case, read the trophy page's read history and forecast every
trophy's earn rate at the target as five quantiles (5/25/50/75/95%).

Model: earners over players, in log time
----------------------------------------
A rate is earners / players-tracked, and the two move for different
reasons, so they are forecast separately and divided at the end:

* Earners E_i(t) for trophy i grow as players play and reach it.  In
  log time, ln E_i is roughly linear in ln(1 + h), h the hours since
  release (2026-11-19T00:00:00Z): early trophies climb fast, late ones
  slow.  Each trophy gets its own least-squares line, shrunk toward a
  prior slope that rises with how widely it is already earned (a trophy
  at 100% is earned by every new player, a rare one by almost none),
  when its history is flat or short.
* Players tracked P(t) grows in steps at Exophase refreshes; between
  refreshes the page is frozen.  ln P is likewise roughly linear in
  ln(1 + h) over the first days (it rose 638 -> 3,868 in the first 4.5
  hours after release, ~6x, then kept climbing).  One line, shrunk
  toward a prior slope of about 1.05 when the history shows no refresh.

The median rate is 100 * E_i(target) / P(target).  Working with earners
and players separately is what makes the forecast dilution-aware: a
refresh that adds thousands of newly tracked players who have not earned
much yet pushes every rate down, even though nobody un-earned a trophy,
and a persistence forecast (last rate, narrow band) misses that step by
~1.8 logit units on the practice.

Spread: quantiles are log-normal around the median in logit space.  The
logit sigma combines the fit residuals with the size of the move the
forecast itself makes (a big predicted dilution is an uncertain one)
and a term that grows with the horizon, floored so the band never
collapses.  A refresh the history has not seen is a step risk, so a flat
history gets a wide band, not a confident one.

Run: python3 main.py (reads $MAKEFUN_INPUT_DIR or /input, writes
$MAKEFUN_OUTPUT_DIR or /output).  Standard library only, deterministic,
no network, no model calls.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release instant: GTA VI released 19 November 2026 (RULES.txt).
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Prior slopes in log time (ln value vs ln(1 + hours since release)) used
# when a history is flat or too short to fit: players tracked climbs
# steeply for a new game (638 -> 3,868 in the first 4.5h is a slope near
# 1.05), earners climb much more slowly in the first hours (the practice
# answers sit near a 0.1 slope: earners up ~20% while players rose 6x).
PRIOR_PLAYERS_SLOPE = 1.05
# How much a fitted slope is trusted over the prior, per distinct value
# seen in the history (a flat history carries no slope information).
SHRINK_PER_POINT = 0.35
# A trophy's earners grow with players to a power that rises with how
# widely it is already earned: a trophy at 100% is earned by every new
# player the moment they are tracked (exponent 1, its rate stays 100),
# while a rare trophy is earned by almost none of them (exponent near
# 0, its rate is diluted almost one-for-one with players).  The square
# root is the shape that fits the launch-to-noon calibration (early
# trophies near 0.7, late ones near 0.3, the 100% one at 1).
def earner_exponent(rate_pct):
    return max(0.05, math.sqrt(max(0.0, rate_pct) / 100.0))
# Spread (logit sigma) floors and weights.
SPREAD_FLOOR = 0.30
SPREAD_MOVE_WEIGHT = 0.25
SPREAD_HORIZON_WEIGHT = 0.5
SPREAD_CAP = 1.4


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(when):
    return (when - RELEASE).total_seconds() / 3600.0


def log_time(h):
    """ln(1 + h), h hours since release, kept above ln(0.1)."""
    return math.log(max(0.1, 1.0 + h))


def ols(xs, ys):
    """Least squares y = a + b x.  Returns (a, b, residual std)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(r * r for r in resid) / max(1, n - 2)
    return a, b, math.sqrt(var)


def shrink(value, prior, n_distinct):
    """Pull a fitted slope toward the prior; flat data trusts the prior.

    A series with a single distinct value carries no slope information
    at all (weight 0 to the fit); each further distinct value earns
    weight, up to full trust at four or more.
    """
    w = min(1.0, SHRINK_PER_POINT * max(0, n_distinct - 1))
    return w * value + (1.0 - w) * prior


def distinct_count(values, tol=1e-9):
    seen = []
    for v in values:
        if all(abs(v - s) > tol for s in seen):
            seen.append(v)
    return len(seen)


def round4(x):
    """Round to 4 decimals, half up, as JavaScript's Math.round does."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]

    # Read times and per-read players / earners.
    times = [parse_time(r["at"]) for r in history]
    xs = [log_time(hours_since_release(t)) for t in times]
    players = [float(r["players_tracked"]) for r in history]
    log_players = [math.log(max(1.0, p)) for p in players]

    # Players-tracked line.  A flat history means the page has not
    # refreshed since before release: the displayed count is the
    # release-time count, stale, so extrapolate from h = 0 (release)
    # with the prior slope instead of fitting a zero slope through
    # stale values.  A history with real variation gets the OLS line.
    h_target = hours_since_release(parse_time(case["target"]))
    x_target = log_time(h_target)
    if distinct_count(players) <= 1:
        players_target = players[-1] * (1.0 + max(0.0, h_target)) ** PRIOR_PLAYERS_SLOPE
        b_players_used = PRIOR_PLAYERS_SLOPE
        resid_p = 0.0
    else:
        _, b_players, resid_p = ols(xs, log_players)
        b_players = shrink(b_players, PRIOR_PLAYERS_SLOPE,
                           distinct_count(players))
        b_players_used = b_players
        a_players = sum(log_players) / len(log_players) - b_players * (sum(xs) / len(xs))
        players_target = math.exp(a_players + b_players * x_target)
    players_target = max(players_target, players[-1])

    # Per-trophy earner fits.  A trophy whose earners never move carries
    # no slope of its own; its prior slope is its earner exponent (from
    # its last rate) times the players slope, so a 100% trophy's earners
    # grow with players and its rate stays 100, while a rare trophy's
    # earners barely grow and its rate is diluted.
    per_trophy = []
    for i in range(len(trophies)):
        earners = []
        ok = True
        for r in history:
            rate = r["rates"][i]
            if rate is None:
                ok = False
                break
            earners.append(max(0.5, rate * r["players_tracked"] / 100.0))
        if not ok or len(earners) < 2:
            per_trophy.append(None)
            continue
        if distinct_count(earners) > 1:
            log_e = [math.log(e) for e in earners]
            a, b, resid = ols(xs, log_e)
        else:
            a, b, resid = 0.0, 0.0, 0.0
        per_trophy.append((a, b, resid, earners[-1], distinct_count(earners)))

    horizon_days = case["horizon_minutes"] / 1440.0
    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = history[-1]["rates"][i]
        if last_rate is None:
            last_rate = 0.0
        last_logit = logit(last_rate)
        fit = per_trophy[i]
        if fit is None:
            # No usable earner history: hold earners, dilute by players.
            earners_target = max(0.5, last_rate * players[-1] / 100.0)
            resid = 0.0
        else:
            a, b, resid, last_earners, n_distinct = fit
            prior_slope = earner_exponent(last_rate) * b_players_used
            if n_distinct <= 1:
                # Flat earners: the page has not refreshed, so the shown
                # earners are the release-time count; grow them from
                # h = 0 with the trophy's prior slope.
                earners_target = last_earners * (1.0 + max(0.0, h_target)) ** prior_slope
                earners_target = max(0.5, earners_target)
                resid = 0.0
            else:
                # Shrink the trophy's slope toward its prior slope.
                b = shrink(b, prior_slope, n_distinct)
                a = (sum(math.log(max(0.5, r["rates"][i] * r["players_tracked"] / 100.0))
                         for r in history if r["rates"][i] is not None)
                     / max(1, sum(1 for r in history if r["rates"][i] is not None))
                     - b * (sum(xs) / len(xs)))
                earners_target = math.exp(a + b * x_target)
                earners_target = max(0.5, earners_target)
        median = clamp_pct(100.0 * earners_target / players_target)
        med_logit = logit(median)
        move = abs(med_logit - last_logit)
        sigma = max(SPREAD_FLOOR, resid,
                    SPREAD_MOVE_WEIGHT * move,
                    SPREAD_HORIZON_WEIGHT * math.sqrt(horizon_days))
        sigma = min(sigma, SPREAD_CAP)
        qs = []
        prev = None
        for z in Z:
            q = round4(clamp_pct(inv_logit(med_logit + z * sigma)))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
