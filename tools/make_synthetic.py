"""Generate a synthetic match in the exact bundle format, plus its ground truth.

Used only to test the pipeline end to end (parsing, alignment, file format, scorer) without the
real data, which may not leave the participant's machine. Nothing here is fitted to, or derived
from, the real match.

    python tools/make_synthetic.py --out /tmp/synth        # writes /tmp/synth/tactical/* + gt.csv
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from possession import config as C  # noqa: E402

L, W = C.PITCH
N = C.N_FRAMES
FPS = C.FPS
SLOTS = 46

BASE = np.array([[5, 34], [20, 10], [18, 26], [18, 42], [20, 58], [35, 15], [33, 34],
                 [35, 53], [48, 12], [50, 34], [48, 56]], dtype=float)   # 4-3-3, attacking +x


def team_targets(sign, poss, ball):
    """Formation targets for a team attacking towards `sign`; pushes up and spreads in
    possession, compacts and drops off without it."""
    base = BASE.copy()
    shift = (ball[0] - L / 2) * 0.5 if sign > 0 else (L / 2 - ball[0]) * 0.5
    if poss:
        base[1:, 0] = base[1:, 0] * 1.15 + 8
        base[1:, 1] = W / 2 + (base[1:, 1] - W / 2) * 1.15
    else:
        base[1:, 0] = base[1:, 0] * 0.8 + 2
        base[1:, 1] = W / 2 + (base[1:, 1] - W / 2) * 0.7 + (ball[1] - W / 2) * 0.25
    base[1:, 0] += shift
    x = base[:, 0] if sign > 0 else L - base[:, 0]
    y = base[:, 1]
    return np.stack([np.clip(x, 1, L - 1), np.clip(y, 1, W - 1)], axis=1)


def simulate(rng):
    pos = np.full((N, 22, 2), np.nan)
    spd = np.full((N, 22), np.nan)
    ball = np.full((N, 2), np.nan)
    gt = np.full(N, "X", dtype="<U1")
    events = []   # (code, start, end, team, type)

    for h, (a, b) in enumerate(C.GRADED_HALVES):
        home_sign = 1.0 if h == 0 else -1.0
        p = np.vstack([team_targets(home_sign, False, (L / 2, W / 2)),
                       team_targets(-home_sign, False, (L / 2, W / 2))])
        p[:, 0] = np.where(np.arange(22) < 11,
                           np.minimum(p[:, 0], L / 2 - 1) if home_sign > 0 else np.maximum(p[:, 0], L / 2 + 1),
                           np.maximum(p[:, 0], L / 2 + 1) if home_sign > 0 else np.minimum(p[:, 0], L / 2 - 1))
        bpos = np.array([L / 2, W / 2])
        kickoff = a + (0 if h == 0 else 2934)
        poss = 0 if h == 0 else 1
        mode = "dead"          # dead | hold | flight
        dead_until = kickoff
        carrier = 9 if poss == 0 else 20
        hold_until = 0
        flight = None
        events.append(("SET PIECES", kickoff - 60, kickoff + 60, poss, "Kick Off (Start)"))
        poss_start = kickoff
        restart_team = poss

        for t in range(a, b + 1):
            if mode == "dead" and t >= dead_until:
                # restart
                poss = restart_team
                carrier = rng.randint(1, 11) + 11 * poss
                bpos = p[carrier].copy()
                mode, hold_until = "hold", t + rng.randint(5, 30)
                poss_start = t
            if mode == "dead":
                lab = "D"
                speed_target = 0.8
            else:
                lab = "H" if poss == 0 else "A"
                speed_target = 3.5

            # ball
            if mode == "hold":
                bpos = p[carrier] + rng.normal(0, 0.3, 2)
                if t >= hold_until:
                    mates = [k for k in range(11 * poss, 11 * poss + 11) if k != carrier]
                    tgt = rng.choice(mates)
                    r = rng.rand()
                    if r < 0.12:      # intercepted
                        opp = range(11 * (1 - poss), 11 * (1 - poss) + 11)
                        tgt = min(opp, key=lambda k: np.hypot(*(p[k] - p[carrier])))
                    elif r < 0.17:    # out of play
                        tgt = -1
                    dest = (p[tgt] if tgt >= 0 else
                            np.array([bpos[0] + rng.normal(0, 15), -3.0 if rng.rand() < .5 else W + 3]))
                    dur = max(5, int(np.hypot(*(dest - bpos)) / 15.0 * FPS))
                    flight = (bpos.copy(), dest, t, t + dur, tgt)
                    mode = "flight"
            elif mode == "flight":
                s, e, t0, t1, tgt = flight
                u = min(1.0, (t - t0) / max(1, t1 - t0))
                bpos = s + (e - s) * u
                if t >= t1:
                    if tgt < 0:
                        mode = "dead"
                        dead_until = t + rng.randint(8 * FPS, 35 * FPS)
                        restart_team = 1 - poss if rng.rand() < 0.6 else poss
                        kind = rng.choice(["Throw-in", "Goal Kick", "Free Kick", "Corner Kick"])
                        events.append(("SET PIECES", dead_until - 60, dead_until + 60,
                                       None if kind == "Throw-in" else restart_team, kind))
                        lab = "D"
                    else:
                        new_poss = 0 if tgt < 11 else 1
                        if new_poss != poss:
                            if t - poss_start > 125:
                                events.append(("BUILD UP", poss_start, t, poss, None))
                                events.append(("DEFENDING IN MIDDLE THIRD", poss_start, t,
                                               1 - poss, None))
                            poss_start = t
                            lab = "H" if new_poss == 0 else "A"
                        poss, carrier = new_poss, tgt
                        mode, hold_until = "hold", t + rng.randint(10, 75)
            if mode == "dead" and t < dead_until and lab == "D":
                bpos = bpos + rng.normal(0, 0.05, 2)

            # players
            if mode != "dead" or True:
                tg = np.vstack([team_targets(home_sign, poss == 0, bpos),
                                team_targets(-home_sign, poss == 1, bpos)])
                if mode == "hold":
                    tg[carrier] = p[carrier] + np.array([home_sign if poss == 0 else -home_sign, 0]) * 0.5
                if mode == "flight" and flight[4] >= 0:
                    tg[flight[4]] = flight[1]
                if t < kickoff:
                    tg = p
                step = tg - p
                dist = np.hypot(step[:, 0], step[:, 1]) + 1e-9
                v = np.minimum(dist * FPS, speed_target * (1 + rng.rand(22))) / FPS
                p = p + step / dist[:, None] * v[:, None] + rng.normal(0, 0.02, (22, 2))
                spd[t] = v * FPS
            pos[t] = p
            ball[t] = bpos
            gt[t] = lab

    return pos, spd, ball, gt, events


def write_bundle(out, pos, spd, ball, events, rng):
    folder = os.path.join(out, "tactical")
    os.makedirs(folder, exist_ok=True)
    stem = os.path.join(folder, C.TACTICAL_STEM)
    with open(stem + "_atd.xml", "w") as fh:
        fh.write('<?xml version="1.0"?>\n<main><Metadata><Teams>'
                 '<Team id="T1"><Name>Barcelona</Name></Team>'
                 '<Team id="T2"><Name>Athletic Club</Name></Team></Teams><Players>\n')
        for s in range(SLOTS):
            fh.write('<Player id="P{0}" teamId="{1}"><Name>Track {0}</Name>'
                     '<ShirtNumber>{0}</ShirtNumber></Player>\n'.format(s + 1, "T1" if s < 23 else "T2"))
        fh.write('</Players></Metadata></main>\n')

    # ball visibility: two-state Markov chain, ~40% missing during play, more when dead
    vis = np.zeros(N, dtype=bool)
    state = True
    for t in range(N):
        if state and rng.rand() < 1 / 40.0:
            state = False
        elif not state and rng.rand() < 1 / 28.0:
            state = True
        vis[t] = state
    # identity switches: player -> slot mapping changes every ~60 s
    slot_of = np.concatenate([np.arange(11), 23 + np.arange(11)])
    nanG = "NaN,NaN,NaN"
    with open(stem + "_atd.txt", "w") as fh:
        for t in range(N):
            if t % 1500 == 0 and t > 0:
                k = rng.randint(22)
                lo, hi = (0, 23) if k < 11 else (23, 46)
                free = sorted(set(range(lo, hi)) - set(slot_of.tolist()))
                slot_of[k] = rng.choice(free)
            groups = [nanG] * SLOTS
            if not np.isnan(pos[t, 0, 0]):
                for k in range(22):
                    if rng.rand() < 0.04:
                        continue
                    groups[slot_of[k]] = "{:.5f},{:.5f},{:.2f}".format(
                        pos[t, k, 0] / L, pos[t, k, 1] / W, spd[t, k])
            if vis[t] and not np.isnan(ball[t, 0]):
                bl = "{:.5f},{:.5f},0".format(ball[t, 0] / L, ball[t, 1] / W)
            else:
                bl = "NaN,NaN,NaN"
            fh.write("{}:{}:{}\n".format(t + 1, ";".join(groups), bl))

    names = {0: "Barcelona", 1: "Atletic Club", None: "N/A"}
    with open(stem + "_smart_tagging.xml", "w") as fh:
        fh.write('<?xml version="1.0"?>\n<file><ALL_INSTANCES>\n')
        for i, (code, s, e, team, kind) in enumerate(events):
            if rng.rand() < 0.15 and team is not None:
                team = 1 - team       # tagging noise
            fh.write("<instance><ID>{}</ID><start>{:.2f}</start><end>{:.2f}</end><code>{}</code>"
                     "<label><group>Team</group><text>{}</text></label>".format(
                         i, s / FPS, e / FPS, code, names[team]))
            if kind:
                fh.write("<label><group>Type</group><text>{}</text></label>".format(kind))
            fh.write("</instance>\n")
        fh.write("</ALL_INSTANCES></file>\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)
    pos, spd, ball, gt, events = simulate(rng)
    write_bundle(args.out, pos, spd, ball, events, rng)
    with open(os.path.join(args.out, "gt.csv"), "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, g) for i, g in enumerate(gt))
    print("synthetic bundle in", args.out)


if __name__ == "__main__":
    main()
