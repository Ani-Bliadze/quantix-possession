"""Readers for the provided assets: ATD (FIFA/EPTS) and Smart Tagging (SportsCode XML).

Standard library + numpy only. Formats as documented in the challenge repository (DATA.md and
starter/baseline.py):

  raw ATD line   "<frame_id>:<slot1>;<slot2>;...;<slotN>:<ball>"
                 each group "x,y,speed" (normalised 0-1, speed m/s), "NaN" when unobserved
  frame_id       counts from 1;  video_frame = frame_id - 1
  meta XML       <Team id=..> (first = home) and <Player teamId=..> in slot order
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

from . import config as C


def asset_paths(data_dir):
    folder = os.path.join(data_dir, "tactical")
    stem = os.path.join(folder, C.TACTICAL_STEM)
    return {
        "atd_meta": stem + "_atd.xml",
        "atd_raw": stem + "_atd.txt",
        "smart_tagging": stem + "_smart_tagging.xml",
    }


def read_meta(meta_path):
    """(team_of_slot int8 array: 0 home / 1 away, (home_name, away_name))."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(teams)))
    home_id = teams[0].get("id")
    slots = [p.get("teamId") for p in root.iter("Player")]
    if not slots:
        raise ValueError("no Player entries in " + meta_path)
    team_of_slot = np.array([0 if t == home_id else 1 for t in slots], dtype=np.int8)
    names = tuple(t.findtext("Name") for t in teams)
    return team_of_slot, names


def read_atd(raw_path, n_slots, n_frames=C.N_FRAMES):
    """Dense arrays over video frames.

    Returns dict with
      px, py, ps : float32 (n_frames, n_slots)  player x, y (metres) and speed (m/s), NaN if unseen
      bx, by     : float32 (n_frames,)          ball x, y (metres), NaN if unseen
      ball_extra : int, number of ball channels beyond x, y (reported, not used)
    """
    length, width = C.PITCH
    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ps = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)
    per_slot = None
    ball_extra = 0
    n_read = 0
    n_bad_players = 0

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(":")
            if len(parts) < 2 or not parts[0].strip():
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue
            groups = parts[1].split(";")
            if per_slot is None:
                per_slot = len(groups[0].split(","))
            values = np.array(parts[1].replace(";", ",").split(","), dtype=np.float32)
            if values.size == per_slot * len(groups):
                values = values.reshape(len(groups), per_slot)
                k = min(len(groups), n_slots)
                px[frame, :k] = values[:k, 0] * length
                py[frame, :k] = values[:k, 1] * width
                if per_slot >= 3:
                    ps[frame, :k] = values[:k, 2]
            else:
                n_bad_players += 1
            if len(parts) >= 3 and parts[2].strip():
                ball = parts[2].split(",")
                ball_extra = max(ball_extra, len(ball) - 2)
                try:
                    bx[frame] = float(ball[0]) * length
                    by[frame] = float(ball[1]) * width
                except (ValueError, IndexError):
                    pass
            n_read += 1

    # Speed channel missing: derive it from positions.
    if per_slot is not None and per_slot < 3:
        vx = np.gradient(px, axis=0) * C.FPS
        vy = np.gradient(py, axis=0) * C.FPS
        ps = np.hypot(vx, vy).astype(np.float32)

    return {"px": px, "py": py, "ps": ps, "bx": bx, "by": by,
            "ball_extra": ball_extra, "n_records": n_read, "n_bad_player_lines": n_bad_players}


def read_smart_tagging(path, fps=C.FPS):
    """[{code, start, end, tags}] with start/end as video frames (end exclusive)."""
    if not path or not os.path.isfile(path):
        return []
    out = []
    for inst in ET.parse(path).iter("instance"):
        try:
            start = float(inst.findtext("start"))
            end = float(inst.findtext("end"))
        except (TypeError, ValueError):
            continue
        tags = {}
        for label in inst.findall("label"):
            group = (label.findtext("group") or "").strip()
            text = (label.findtext("text") or "").strip()
            if group:
                tags[group.lower()] = text
        out.append({"code": (inst.findtext("code") or "").strip().upper(),
                    "start": int(round(start * fps)), "end": int(round(end * fps)),
                    "tags": tags})
    return out


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    return path
