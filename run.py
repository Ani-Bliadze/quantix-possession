"""Regenerate the submission file from the provided match data.

    python run.py --data-dir /path/to/data --out submissions/TEAMNAME.csv \
                  [--challenge-repo /path/to/more-than-a-hack-2027]

--data-dir is the unpacked bundle (the folder that contains tactical/). With --challenge-repo the
file is also validated with the official validator and scored on the public 5 minutes.

Deterministic: same data in, byte-identical file out.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from possession import config as C            # noqa: E402
from possession import data_io, features, model, tagging   # noqa: E402


def summarise(labels):
    out = {}
    for h, (a, b) in enumerate(C.GRADED_HALVES):
        s = labels[a:b + 1]
        runs = []
        start = 0
        for i in range(1, len(s) + 1):
            if i == len(s) or s[i] != s[start]:
                runs.append((s[start], i - start))
                start = i
        poss = [n for k, n in runs if k in ("H", "A")]
        out["half{}".format(h + 1)] = {
            "H": int((s == "H").sum()), "A": int((s == "A").sum()), "D": int((s == "D").sum()),
            "possessions": len(poss),
            "mean_possession_s": round(float(np.mean(poss)) / C.FPS, 1) if poss else 0.0,
            "dead_stretches": sum(1 for k, _ in runs if k == "D"),
            "dead_share": round(float((s == "D").mean()), 3),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR",
                                                         os.path.join(HERE, "data")))
    ap.add_argument("--out", default=os.path.join(HERE, "submissions", "submission.csv"))
    ap.add_argument("--challenge-repo", default=None,
                    help="path to a clone of metrica-sports/more-than-a-hack-2027")
    ap.add_argument("--no-tagging", action="store_true", help="ignore the Smart Tagging")
    args = ap.parse_args()

    t0 = time.time()
    paths = data_io.asset_paths(args.data_dir)
    for key in ("atd_meta", "atd_raw"):
        if not os.path.isfile(paths[key]):
            sys.exit("missing {}: {}".format(key, paths[key]))

    team_of_slot, names = data_io.read_meta(paths["atd_meta"])
    print("teams: H = {}, A = {}   ({} track slots)".format(names[0], names[1], len(team_of_slot)))
    print("reading tracking ...")
    atd = data_io.read_atd(paths["atd_raw"], len(team_of_slot))
    print("  {} records, {} extra ball channels, {} malformed player lines  [{:.0f}s]".format(
        atd["n_records"], atd["ball_extra"], atd["n_bad_player_lines"], time.time() - t0))

    f = features.compute(atd, team_of_slot)
    del atd
    print("features  [{:.0f}s]".format(time.time() - t0))

    events = [] if args.no_tagging else data_io.read_smart_tagging(paths["smart_tagging"])
    tag_dead, tag_team, tinfo = tagging.evidence(events, f, names)

    states, minfo = model.run(f, tag_dead, tag_team)
    labels = model.to_labels(states)
    data_io.write_submission(args.out, labels)
    print("decoded   [{:.0f}s]".format(time.time() - t0))

    report = {"features": f["info"], "tagging": tinfo, "shape_model": minfo["shape"], "dead_model": minfo["dead_model"],
              "summary": summarise(labels)}
    print(json.dumps(report, indent=1))
    print("wrote", args.out)

    if args.challenge_repo:
        repo = os.path.abspath(args.challenge_repo)
        print("\n--- official validator")
        subprocess.call([sys.executable, os.path.join(repo, "evaluation", "validate_submission.py"),
                         os.path.abspath(args.out)])
        print("\n--- public 5-minute window (pipeline check only, NOT a tuning signal)")
        res_dir = os.path.dirname(os.path.abspath(args.out))
        subprocess.call([sys.executable, os.path.join(repo, "evaluation", "run_evaluation.py"),
                         "--results", res_dir, "--no-detail"], cwd=repo)


if __name__ == "__main__":
    main()
