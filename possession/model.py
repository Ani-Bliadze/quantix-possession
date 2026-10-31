"""Segmentation: a 3-state (H, A, D) Viterbi decoder over per-frame evidence, run twice.

Pass 1 uses ball + dead-ball + tagging evidence. Its long, confident possessions become
pseudo-labels for a logistic regression on team-shape features (self-supervised: no human label
is involved). Pass 2 adds that shape evidence, which matters most where the ball is not tracked.
"""

import numpy as np

from . import config as C

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])


def viterbi(e_h, e_a, e_d, switch, dead_pen):
    """Max-score state path. Scores are additive log-odds; transitions cost a penalty."""
    n = len(e_h)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    back = np.zeros((n, 3), dtype=np.int8)
    sh, sa, sd = float(e_h[0]), float(e_a[0]), float(e_d[0])
    eh, ea, ed = e_h.tolist(), e_a.tolist(), e_d.tolist()
    for t in range(1, n):
        # into H
        ch = (sh, sa - switch, sd - dead_pen)
        bh = 0 if ch[0] >= ch[1] and ch[0] >= ch[2] else (1 if ch[1] >= ch[2] else 2)
        # into A
        ca = (sh - switch, sa, sd - dead_pen)
        ba = 1 if ca[1] >= ca[0] and ca[1] >= ca[2] else (0 if ca[0] >= ca[2] else 2)
        # into D
        cd = (sh - dead_pen, sa - dead_pen, sd)
        bd = 2 if cd[2] >= cd[0] and cd[2] >= cd[1] else (0 if cd[0] >= cd[1] else 1)
        sh, sa, sd = ch[bh] + eh[t], ca[ba] + ea[t], cd[bd] + ed[t]
        back[t, 0], back[t, 1], back[t, 2] = bh, ba, bd
        m = max(sh, sa, sd)            # keep numbers small
        sh, sa, sd = sh - m, sa - m, sd - m
    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(np.argmax((sh, sa, sd)))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(team, dead, f):
    """States for the whole video; -1 outside the graded halves."""
    states = np.full(f["n"], -1, dtype=np.int8)
    for a, b in C.GRADED_HALVES:
        sl = slice(a, b + 1)
        t, d = team[sl], dead[sl]
        states[sl] = viterbi(t / 2.0, -t / 2.0, d, C.SWITCH_PENALTY, C.DEAD_PENALTY)
    return states


def spells(states, a, b):
    """[(start, stop, state)] runs inside [a, b]."""
    s = states[a:b + 1]
    edges = np.flatnonzero(np.diff(s)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(s)]))
    return [(a + int(i), a + int(j), int(s[i])) for i, j in zip(starts, stops)]


# ------------------------------------------------------------------ self-supervised shape model
def fit_logistic(X, y, l2=1.0, iters=25):
    """L2-regularised logistic regression by Newton/IRLS. Returns (w, b, mean, std)."""
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-6
    Z = (X - mu) / sd
    Z1 = np.hstack([Z, np.ones((len(Z), 1))])
    w = np.zeros(Z1.shape[1])
    reg = l2 * np.eye(Z1.shape[1])
    reg[-1, -1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z1 @ w, -30, 30)))
        g = Z1.T @ (p - y) + reg @ w
        Hm = (Z1 * (p * (1 - p))[:, None]).T @ Z1 + reg
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w[:-1], w[-1], mu, sd


def shape_logodds(f, states):
    """Train on pass-1 pseudo-labels, return per-frame home-vs-away log-odds (0 where unknown)."""
    X = f["shape"]
    ok = ~np.isnan(X).any(axis=1)
    idx, lab = [], []
    for a, b in C.GRADED_HALVES:
        for s0, s1, st in spells(states, a, b):
            if st == D or s1 - s0 < C.PSEUDO_MIN_SPELL:
                continue
            r = np.arange(s0 + C.PSEUDO_TRIM, s1 - C.PSEUDO_TRIM)
            idx.append(r)
            lab.append(np.full(len(r), 1.0 if st == H else 0.0))
    info = {"n_pseudo": 0}
    out = np.zeros(f["n"])
    if not idx:
        return out, info
    idx, lab = np.concatenate(idx), np.concatenate(lab)
    keep = ok[idx]
    idx, lab = idx[keep], lab[keep]
    if len(idx) < 500 or lab.min() == lab.max():
        return out, info
    # balance classes and cap size, deterministically
    rng = np.random.RandomState(0)
    pos, neg = np.flatnonzero(lab == 1), np.flatnonzero(lab == 0)
    k = min(len(pos), len(neg), C.PSEUDO_MAX_SAMPLES // 2)
    sel = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)])
    w, b, mu, sd = fit_logistic(X[idx[sel]], lab[sel])
    z = np.zeros(f["n"])
    z[ok] = ((X[ok] - mu) / sd) @ w + b
    out = np.clip(z, -C.SHAPE_CLIP, C.SHAPE_CLIP)
    pred = (z[idx[sel]] > 0).astype(float)
    info = {"n_pseudo": int(len(sel)), "train_agreement": round(float((pred == lab[sel]).mean()), 3),
            "weights": [round(float(v), 3) for v in w]}
    return out, info


def dead_logodds(f):
    """Self-supervised dead-ball model on collective player motion.

    Positives (dead): frames between a tagged set piece's start and its detected restart, the
    pre-kickoff stretch of each half, and frames with the ball tracked out of the pitch.
    Negatives (alive): ball seen, in bounds, moving and within 1.5 m of a player, away from any
    set piece. All pseudo-labels come from algorithms; no human label is involved."""
    X = f["motion"]
    ok = ~np.isnan(X).any(axis=1) & f["graded"]
    pos = f.get("pseudo_dead", np.zeros(f["n"], bool)) | f["out"]
    alive = (f["seen_raw"] & ~f["out"] & (np.nan_to_num(f["bspeed"], nan=0.0) > 2.0)
             & (f["dmin"] < 1.5) & ~f.get("pseudo_skip", np.zeros(f["n"], bool)))
    pi, ni = np.flatnonzero(pos & ok), np.flatnonzero(alive & ok & ~pos)
    info = {"n_dead_pseudo": int(len(pi)), "n_alive_pseudo": int(len(ni))}
    out = np.zeros(f["n"])
    if len(pi) < 200 or len(ni) < 200:
        return out, info
    rng = np.random.RandomState(1)
    k = min(len(pi), len(ni), C.PSEUDO_MAX_SAMPLES // 2)
    sel = np.concatenate([rng.choice(pi, k, replace=False), rng.choice(ni, k, replace=False)])
    y = np.concatenate([np.ones(k), np.zeros(k)])
    w, b, mu, sd = fit_logistic(X[sel], y)
    good = ~np.isnan(X).any(axis=1)
    z = np.zeros(f["n"])
    z[good] = ((X[good] - mu) / sd) @ w + b
    info["train_agreement"] = round(float(((z[sel] > 0) == (y == 1)).mean()), 3)
    info["dead_share_if_thresholded"] = round(float((z[f["graded"]] > 0).mean()), 3)
    return np.clip(z, -C.DEAD_MODEL_CLIP, C.DEAD_MODEL_CLIP), info


def run(f, tag_dead, tag_team):
    team1 = f["team_ball"] + tag_team
    dz, dinfo = dead_logodds(f)
    dead = f["dead"] + tag_dead + C.DEAD_MODEL_GAIN * dz
    s1 = decode(team1, dead, f)
    shape, sinfo = shape_logodds(f, s1)
    gain = np.where(f["seen"], C.SHAPE_GAIN_VISIBLE, C.SHAPE_GAIN_MISSING)
    team2 = team1 + gain * shape
    s2 = decode(team2, dead, f)
    return s2, {"pass1": s1, "shape": sinfo, "dead_model": dinfo}


def to_labels(states):
    labels = np.full(len(states), "X", dtype="<U1")
    m = states >= 0
    labels[m] = LETTERS[states[m]]
    return labels
