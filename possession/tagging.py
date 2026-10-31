"""Smart Tagging as evidence (never as labels).

* SET PIECES  -> dead-ball evidence from the tag start until the detected restart (the moment
                 the ball starts moving inside the tag window), then team evidence for the
                 restarting team when the tag names one. Kick Off (Start) additionally marks
                 everything in that half before the restart as dead (teams lining up).
* phase codes -> weak team evidence. Whether a code's Team tag names the attacking or the
                 defending side is *estimated* per code, by its agreement with the ball-based
                 evidence -- no assumption, no labels.
"""

import difflib

import numpy as np

from . import config as C


def team_side(tag_team, names):
    """0 home, 1 away, None for N/A / unknown. Fuzzy, because the tag spells 'Atletic Club'."""
    if not tag_team or tag_team.strip().upper() in ("N/A", "NA", "NONE", ""):
        return None
    t = tag_team.lower()
    scores = [difflib.SequenceMatcher(None, t, (nm or "").lower()).ratio() for nm in names]
    if max(scores) < 0.4:
        return None
    return int(np.argmax(scores))


def detect_restart(f, start, end):
    """First frame inside [start, end) where the ball accelerates from rest; middle if none."""
    end = min(end, f["n"])
    if end <= start:
        return start
    sp = np.nan_to_num(f["bspeed"][start:end], nan=0.0)
    seen = f["seen_raw"][start:end]
    still = seen & (sp < C.STATIC_BALL_SPEED * 2)
    moving = seen & (sp > 4.0)
    # first moving frame preceded (within 1 s) by a still frame
    for i in np.flatnonzero(moving):
        if still[max(0, i - C.FPS):i].any():
            return start + int(i)
    return (start + end) // 2


def evidence(events, f, names):
    n = f["n"]
    dead = np.zeros(n)
    team = np.zeros(n)
    halves = f["halves"]
    info = {"n_events": len(events), "set_pieces": 0, "phase_polarity": {}}
    pseudo_dead = np.zeros(n, dtype=bool)      # algorithmic D pseudo-labels for model.py
    pseudo_skip = np.zeros(n, dtype=bool)      # frames near set pieces: not alive examples
    f["pseudo_dead"], f["pseudo_skip"] = pseudo_dead, pseudo_skip
    if not events:
        return dead, team, info

    team_ball = f["team_ball"]

    for ev in events:
        a, b = max(0, ev["start"]), min(n, ev["end"])
        if b <= a:
            continue
        if ev["code"] == "SET PIECES":
            info["set_pieces"] += 1
            r = detect_restart(f, a, b)
            kind = ev["tags"].get("type", "").lower()
            if kind.startswith("kick off (start)") and halves[r] >= 0:
                h = halves[r]
                half_start = C.GRADED_HALVES[h][0]
                dead[half_start:r] += 5.0
                pseudo_dead[half_start:r] = True
            dead[a:r] += C.SETPIECE_DEAD_GAIN
            pseudo_dead[a:r] = True
            pseudo_skip[max(0, a - C.SETPIECE_DEAD_LOOKBACK):min(n, b + C.FPS)] = True
            dead[max(0, a - C.SETPIECE_DEAD_LOOKBACK):a] += C.SETPIECE_LOOKBACK_GAIN
            side = team_side(ev["tags"].get("team"), names)
            if side is not None:
                s = 1.0 if side == 0 else -1.0
                team[r:min(n, r + C.SETPIECE_TEAM_FRAMES)] += s * C.SETPIECE_TEAM_GAIN

    # phase codes: estimate polarity from agreement with ball evidence
    by_code = {}
    for ev in events:
        if ev["code"] in ("SET PIECES", "GOALS"):
            continue
        side = team_side(ev["tags"].get("team"), names)
        if side is None:
            continue
        by_code.setdefault(ev["code"], []).append((ev, side))
    for code, items in by_code.items():
        num, den = 0.0, 0.0
        for ev, side in items:
            a, b = max(0, ev["start"]), min(n, ev["end"])
            s = 1.0 if side == 0 else -1.0
            seg = team_ball[a:b]
            num += s * seg.sum()
            den += np.abs(seg).sum()
        agreement = num / den if den > 0 else 0.0
        info["phase_polarity"][code] = round(float(agreement), 3)
        if abs(agreement) < C.PHASE_MIN_AGREEMENT:
            continue
        polarity = 1.0 if agreement > 0 else -1.0
        for ev, side in items:
            a, b = max(0, ev["start"]), min(n, ev["end"])
            s = 1.0 if side == 0 else -1.0
            team[a:b] += polarity * s * C.PHASE_TEAM_GAIN
    return dead, team, info
