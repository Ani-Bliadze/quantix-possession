"""Per-frame evidence from the tracking data.

Everything here produces *log-odds* style evidence per video frame:

  team_ball[t]  > 0 favours home, < 0 favours away   (from who is near a controlled ball)
  dead[t]       > 0 favours dead ball, < 0 favours ball alive
  shape_X[t]    team-shape features used by the self-supervised model (model.py)
"""

import warnings

import numpy as np

from . import config as C


# ------------------------------------------------------------------------------------- helpers
def interp_short_gaps(values, max_gap):
    """Linearly fill NaN runs of length <= max_gap that have data on both sides."""
    out = values.copy()
    isnan = np.isnan(out)
    if not isnan.any() or isnan.all():
        return out
    edges = np.diff(np.concatenate(([0], isnan.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    for a, b in zip(starts, stops):
        if a == 0 or b >= len(out) or b - a > max_gap:
            continue
        out[a:b] = np.linspace(out[a - 1], out[b], b - a + 2)[1:-1]
    return out


def rolling_nanmean(x, window):
    """Centred rolling mean ignoring NaN; NaN where the window has no data."""
    ok = ~np.isnan(x)
    s = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    n = np.concatenate(([0], np.cumsum(ok)))
    half = window // 2
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - half)
    hi = np.minimum(len(x), idx + half + 1)
    cnt = n[hi] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (s[hi] - s[lo]) / np.maximum(cnt, 1), np.nan)


def trailing_all(mask, window):
    """True where mask has been True for each of the last `window` frames (inclusive)."""
    c = np.concatenate(([0], np.cumsum(mask.astype(np.int32))))
    idx = np.arange(len(mask))
    lo = np.maximum(0, idx - window + 1)
    return (c[idx + 1] - c[lo]) == (idx + 1 - lo)


def run_lengths_of_true(mask):
    """For each frame, the length of the True run it belongs to (0 where False)."""
    out = np.zeros(len(mask), dtype=np.int32)
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        out[a:b] = b - a
    return out


def median_filter(x, k):
    if k <= 1:
        return x
    pad = k // 2
    xp = np.concatenate((np.full(pad, np.nan), x, np.full(pad, np.nan)))
    win = np.lib.stride_tricks.sliding_window_view(xp, k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(win, axis=1)


def graded_mask(n=C.N_FRAMES):
    m = np.zeros(n, dtype=bool)
    for a, b in C.GRADED_HALVES:
        m[a:b + 1] = True
    return m


def half_index(n=C.N_FRAMES):
    """0 for first half block, 1 for second, -1 elsewhere."""
    h = np.full(n, -1, dtype=np.int8)
    for i, (a, b) in enumerate(C.GRADED_HALVES):
        h[a:b + 1] = i
    return h


def gmm_low_logodds(values, mask, iters=100):
    """Fit a 2-component 1-D Gaussian mixture on values[mask]; return per-frame log-odds of the
    lower-mean component (NaN-safe: NaN values get 0)."""
    v = values[mask & ~np.isnan(values)]
    out = np.zeros(len(values))
    if v.size < 100:
        return out, None
    mu = np.percentile(v, [20, 80]).astype(float)
    sd = np.array([v.std(), v.std()]) / 2 + 1e-3
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        ll = (-0.5 * ((v[:, None] - mu) / sd) ** 2 - np.log(sd) + np.log(w))
        ll -= ll.max(axis=1, keepdims=True)
        r = np.exp(ll)
        r /= r.sum(axis=1, keepdims=True)
        nk = r.sum(axis=0) + 1e-9
        mu = (r * v[:, None]).sum(axis=0) / nk
        sd = np.sqrt((r * (v[:, None] - mu) ** 2).sum(axis=0) / nk) + 1e-3
        w = nk / nk.sum()
    lo, hi = (0, 1) if mu[0] < mu[1] else (1, 0)
    x = np.where(np.isnan(values), mu[hi], values)
    ll_lo = -0.5 * ((x - mu[lo]) / sd[lo]) ** 2 - np.log(sd[lo]) + np.log(w[lo])
    ll_hi = -0.5 * ((x - mu[hi]) / sd[hi]) ** 2 - np.log(sd[hi]) + np.log(w[hi])
    out = ll_lo - ll_hi
    out[np.isnan(values)] = 0.0
    return out, {"mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist(), "low": int(lo)}


# --------------------------------------------------------------------------------- main entry
def compute(atd, team_of_slot):
    length, width = C.PITCH
    px, py, ps = atd["px"], atd["py"], atd["ps"]
    n = len(atd["bx"])
    home = team_of_slot == 0
    away = team_of_slot == 1
    info = {}

    # ---- ball
    bx_raw, by_raw = atd["bx"].astype(np.float64), atd["by"].astype(np.float64)
    seen_raw = ~np.isnan(bx_raw)
    bx = interp_short_gaps(bx_raw, C.BALL_INTERP_MAX_GAP)
    by = interp_short_gaps(by_raw, C.BALL_INTERP_MAX_GAP)
    seen = ~np.isnan(bx)
    vx = np.gradient(bx) * C.FPS
    vy = np.gradient(by) * C.FPS
    bspeed = median_filter(np.hypot(vx, vy), C.BALL_SPEED_SMOOTH)
    info["ball_seen_raw"] = float(seen_raw.mean())
    info["ball_seen_interp"] = float(seen.mean())

    # ---- distances ball -> players (metres)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = np.hypot(px - bx[:, None].astype(np.float32), py - by[:, None].astype(np.float32))
        d_home = np.nanmin(np.where(home[None, :], d, np.nan), axis=1)
        d_away = np.nanmin(np.where(away[None, :], d, np.nan), axis=1)
    del d
    d_home = np.where(np.isnan(d_home), 60.0, d_home)
    d_away = np.where(np.isnan(d_away), 60.0, d_away)
    dmin = np.minimum(d_home, d_away)

    inb = (seen_raw
           & (bx_raw > -C.OUT_OF_BOUNDS_MARGIN) & (bx_raw < length + C.OUT_OF_BOUNDS_MARGIN)
           & (by_raw > -C.OUT_OF_BOUNDS_MARGIN) & (by_raw < width + C.OUT_OF_BOUNDS_MARGIN))
    out = seen_raw & ~inb

    # ---- team evidence from the ball
    control = np.exp(-np.maximum(dmin - C.CONTROL_RADIUS, 0.0) / C.CONTROL_DECAY)
    flight = 1.0 / (1.0 + (np.nan_to_num(bspeed, nan=0.0) / C.BALL_FLIGHT_SPEED) ** 2)
    team_ball = C.TEAM_GAIN * np.tanh((d_away - d_home) / C.TEAM_MARGIN_SCALE) * control * flight
    team_ball[~seen | out] = 0.0

    # ---- activity (median player speed), self-calibrated with a 2-component mixture
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sp = np.where(ps > 12.0, np.nan, ps)
        med_speed = np.nanmedian(sp, axis=1)
    act = rolling_nanmean(med_speed, C.ACTIVITY_WINDOW)
    graded = graded_mask(n)
    log_act = np.log(np.maximum(act, 0.05))
    low_lo, gmm = gmm_low_logodds(log_act, graded)
    info["activity_gmm"] = gmm
    missing_run = run_lengths_of_true(~seen)
    boost = np.where(missing_run >= C.MISSING_LONG, C.MISSING_LONG_BOOST, 1.0)
    dead_activity = C.ACTIVITY_GAIN * np.clip(low_lo, -C.ACTIVITY_CLIP, C.ACTIVITY_CLIP) * boost

    # ---- static ball, out of bounds, moving ball near a player
    static = trailing_all(seen_raw & (np.nan_to_num(bspeed, nan=99.0) < C.STATIC_BALL_SPEED),
                          C.STATIC_BALL_WINDOW)
    moving = inb & (np.nan_to_num(bspeed, nan=0.0) > 1.5) & (dmin < 3.0)
    over = np.maximum(missing_run - C.MISSING_LONG, 0) / float(C.FPS)
    dead_missing = np.where(missing_run >= C.MISSING_LONG,
                            np.minimum(C.MISSING_RUN_GAIN + C.MISSING_RUN_RAMP * over,
                                       C.MISSING_RUN_CAP), 0.0)
    dead = (dead_activity
            + dead_missing
            + C.OUT_GAIN * out
            + C.STATIC_GAIN * static
            - C.ALIVE_MOVING_GAIN * moving)

    # ---- attack direction per half (from the deepest player of each team: the goalkeeper)
    halves = half_index(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        home_min_x = np.nanmin(np.where(home[None, :], px, np.nan), axis=1)
        away_min_x = np.nanmin(np.where(away[None, :], px, np.nan), axis=1)
    home_dir = np.zeros(n)          # +1: home attacks towards +x
    dirs = []
    for h in (0, 1):
        m = halves == h
        hm, am = np.nanmedian(home_min_x[m]), np.nanmedian(away_min_x[m])
        sign = 1.0 if (np.isnan(am) or hm <= am) else -1.0
        home_dir[m] = sign
        dirs.append(sign)
    # Ungraded frames: use the nearest half's direction (they are never scored anyway).
    home_dir[home_dir == 0] = dirs[0]
    info["home_attacks_plus_x"] = dirs

    shape = shape_features(px, py, home, away, home_dir)
    motion = motion_features(px, py, sp)

    return {
        "n": n, "seen": seen, "seen_raw": seen_raw, "bx": bx, "by": by, "bspeed": bspeed,
        "d_home": d_home, "d_away": d_away, "control": control,
        "team_ball": team_ball, "dead": dead, "out": out, "static": static,
        "activity": act, "shape": shape, "motion": motion, "dmin": dmin, "graded": graded, "halves": halves,
        "home_dir": home_dir, "info": info,
    }


def shape_features(px, py, home, away, home_dir):
    """Home-minus-away team-shape descriptors, smoothed over 1 s, x measured towards each team's
    own attacking direction. Shape (who is stretched, who is pushed up, who is moving forward)
    still carries possession information when the ball is not tracked."""
    length, _ = C.PITCH
    vx = np.gradient(px, axis=0) * C.FPS
    vx = np.clip(np.nan_to_num(vx, nan=0.0), -10, 10)
    vx[np.isnan(px)] = np.nan

    feats = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_team = []
        for mask, sign in ((home, home_dir), (away, -home_dir)):
            x = np.where(mask[None, :], px, np.nan)
            y = np.where(mask[None, :], py, np.nan)
            v = np.where(mask[None, :], vx, np.nan)
            fwd_x = sign * (np.nanmean(x, axis=1) - length / 2)
            sx = np.nanstd(x, axis=1)
            sy = np.nanstd(y, axis=1)
            fv = sign * np.nanmean(v, axis=1)
            # highest outfield line: 2nd most advanced player along the attacking direction
            adv = np.sort(sign[:, None] * (x - length / 2), axis=1)
            cnt = np.sum(~np.isnan(x), axis=1)
            idx = np.clip(cnt - 2, 0, x.shape[1] - 1)
            top = np.take_along_axis(np.where(np.isnan(adv), -np.inf, adv), idx[:, None],
                                     axis=1)[:, 0]
            top[cnt < 2] = np.nan
            per_team.append((fwd_x, sx, sy, fv, top))
    for a, b in zip(per_team[0], per_team[1]):
        feats.append(rolling_nanmean(a - b, C.FPS))
    X = np.stack(feats, axis=1)
    return X


def motion_features(px, py, sp):
    """Collective player-motion descriptors, smoothed over 2 s. Used by the self-supervised
    dead-ball model. Deliberately no ball features: the alive pseudo-labels are taken where the
    ball is seen, so ball visibility would be a trivially leaky feature."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(sp, axis=1)
        p80 = np.nanpercentile(sp, 80, axis=1)
        cnt = np.sum(~np.isnan(sp), axis=1).astype(float)
        cnt[cnt == 0] = np.nan
        walk = np.sum(sp < 1.0, axis=1) / cnt
        run_ = np.sum(sp > 4.0, axis=1) / cnt
        sx = np.nanstd(px, axis=1)
        sy = np.nanstd(py, axis=1)
    feats = [np.log(np.maximum(med, 0.05)), np.log(np.maximum(p80, 0.05)), walk, run_, sx, sy]
    return np.stack([rolling_nanmean(v, C.ACTIVITY_WINDOW) for v in feats], axis=1)
