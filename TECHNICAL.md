# Technical document: team possession from automatic tracking

*More than a Hack 2027, Phase 1. Team: Quantix.*

## 1. Approach

The ranking metric (`spellF1`) rewards recovering whole possessions as intervals, so we treat
the task as **segmentation**, not per-frame classification. Every frame gets *evidence* scores in
log-odds units, and a 3-state (H, A, D) Viterbi decoder finds the best piecewise-constant labeling.
Each state change costs a fixed penalty:

| transition | penalty | meaning |
|---|---|---|
| H ↔ A | 9 | a change of possession needs about 0.4 s of clear, controlled evidence for the other team |
| alive ↔ D | 6 | a stoppage has to be supported for a sustained stretch |

This is how the "controlled possession" definition becomes an algorithm. A single touch,
deflection or duel produces a few frames of evidence. That is less than the switch penalty, so it
does not flip possession. A player who receives and keeps the ball does flip it. When evidence is
absent (the ball is untracked), the decoder simply keeps the current state, which matches the
rule that a loose or contested ball stays with the last team in possession.

The decoder runs twice:

1. **Pass 1** uses ball, dead-ball and tagging evidence.
2. Possessions longer than 5 s from pass 1, trimmed by 1 s at each end, become **pseudo-labels**
   for a logistic regression on team-shape features. No human labels are involved: the model is
   self-supervised.
3. **Pass 2** adds the shape model's output as extra team evidence. It is strongest where the
   ball is not tracked (41% of graded frames).

## 2. Assets used

| asset | used | how |
|---|---|---|
| Tactical ATD (players and ball) | yes | the main signal |
| Smart Tagging | yes | soft evidence only, never copied as labels |
| Homography | no | it maps the same positions to pixels and adds no possession information to the ATD |
| Tactical / broadcast video | no | out of scope for a tracking-geometry approach |
| Broadcast ATD | no | not synchronised to tactical, and the alignment risk outweighs the extra ball coverage |

Published match constants are used: the graded half blocks (DATA.md) and the pitch size
(games.py). Nothing is tied to specific frames of the game.

## 3. Evidence

All positions are converted to metres (105.3 × 68 m). Ball gaps of up to 0.32 s are bridged by
linear interpolation.

**Ball control (team evidence).** Per frame, let `d_H` and `d_A` be the distances from the ball to
the nearest tracked home and away players:

```
team = tanh((d_A − d_H) / 1.5 m) · control(min distance) · flight(ball speed)
control = exp(−max(d − 1 m, 0) / 1.5 m)      ≈1 at the feet, ≈0.1 at 4.5 m
flight  = 1 / (1 + (speed / 10 m/s)²)          a ball in flight (a pass or shot) says little
```

**Dead ball.**

- The ball is tracked more than 1 m outside the lines: strong D evidence.
- The ball has been static (< 0.6 m/s) for 1 s: D evidence. In live play the ball is almost never
  still.
- Low collective activity: a 2 s rolling mean of the median player speed. The match's own
  distribution is split by a 2-component Gaussian mixture on log-speed, fitted without labels, and
  the posterior log-odds of the low-activity component count as D evidence. The evidence is
  boosted when the ball has also been unseen for more than 3 s.
- A ball unseen for more than 3 s is D evidence in itself, growing with the length of the gap
  (0.3 per frame, plus 0.1 for every further second, capped at 1.0). In live play, ball gaps
  rarely last that long. During stoppages the ball is often off-camera or being carried.
- A moving ball, in bounds and near a player: alive evidence.

**Dead-ball model (self-supervised).** A logistic regression on collective player motion. It
uses six descriptors smoothed over 2 s: log median speed, log 80th-percentile speed, the share of
players walking (< 1 m/s), the share running (> 4 m/s), and the length and width spread of all
players. It is trained on algorithmic pseudo-labels:

- *dead:* set-piece tag start to detected restart, the pre-kickoff stretch, and ball out of the
  pitch
- *alive:* a moving ball within 1.5 m of a player, away from set pieces

Ball features are excluded on purpose, because the alive examples all have a visible ball. Its
clipped log-odds × 0.6 is added to the dead-ball evidence.

**Team shape (self-supervised).** These are home-minus-away differences of five descriptors,
smoothed over 1 s, with x measured along each team's attacking direction:

- centroid advance
- length spread
- width spread
- mean forward velocity
- position of the second most advanced player

The team in possession typically pushes up, stretches and moves forward. Attacking direction is
estimated per half from each team's deepest player (the goalkeeper), using the median over the
half.

**Smart Tagging.**

- *Set pieces.* D evidence runs from 10 s before the tag (a weaker prior: stoppages usually last
  that long) and more strongly from the tag start until the detected restart, which is the first
  frame where the ball accelerates from rest inside the tag window. For a *Kick Off (Start)*, the
  whole stretch from the start of the graded half to the restart is D, which handles the 2nd-half
  pre-kickoff frames. After the restart, team evidence goes to the restarting team for 2 s when
  the tag names one. Throw-ins are tagged `N/A` and get no team evidence.
- *Phase codes* (Build Up, Progression, Defending in …) give weak team evidence. Whether a code's
  `Team` tag names the attacking or the defending side is **estimated per code** from its
  agreement with the ball evidence. Codes with weak agreement (|a| < 0.15) are ignored.

## 4. Possession rules as implemented

- **Loose balls and challenges.** The previous team keeps the ball until the other team has
  accumulated controlled-ball evidence above the switch penalty. Contested frames produce little
  net evidence (`tanh` of a small distance margin), so a duel does not flip possession.
- **Deflections and single touches.** These last only a few frames, which is below the penalty, so
  there is no change.
- **Aerial duels.** The ball is often untracked or flagged as in flight, so the evidence is weak and
  there is no change unless control follows.
- **Restarts.** D until the ball moves from rest. The restarting team comes from the tag if it names
  one, otherwise from whoever controls the ball next.
- **Dead ball.** Always `D` and never a possession, as the challenge requires. Dead-ball ownership
  is not modelled.

## 5. Training data

None. No labelled data from any game was used, and the public 5-minute ground truth was not used
for training or parameter selection (RULES.md). All parameters are football and physics priors,
listed with their rationale in `possession/config.py`. The only quantities fitted are fitted
**without labels** on this match's tracking:

- the activity mixture
- the attacking direction
- the tag polarity
- the shape model, from algorithmic pseudo-labels

## 6. Validation

- **Synthetic match** (`tools/make_synthetic.py`). This is a simulator in the bundle format with
  ~40% ball dropout, identity switches, set pieces and noisy tags, scored with the official
  scorer. On it, the pipeline gets spellF1 0.63 against 0.42 for `baseline_smooth50`. This
  validates correctness, not real-world accuracy.
- **Format.** Output passes `validate_submission.py`.
- **Determinism.** Output is byte-identical across runs and across numpy 1.26 and 2.x.
- **Public 5 minutes.** Used only to confirm alignment and label meaning.

## 7. Calibration history

Version 1 used larger penalties (12 / 10) and no missing-ball or set-piece lookback cues. On the
real match it labelled only about 11% of graded frames as dead ball. Effective playing time in
professional football is roughly 55 to 60% of the clock, so that was far too low. Missing
stoppages merge neighbouring possessions, which shows up as too few, too long possessions (about
100 per half, averaging 25 to 30 s). Version 2 strengthens the dead-ball cues and lowers the
penalties. The change was decided from this football prior and the run's own aggregate counts,
not from the public ground truth.

Version 3 adds the self-supervised dead-ball model above, because v2 still labelled only 16 to
22% of graded frames dead.

## 8. Limitations and next steps

- Parameters are priors, not learned. With labelled data from *other* games, the evidence weights
  and penalties could be fitted, for example as a CRF.
- Dead-ball detection rests on activity and ball cues. Quick restarts (fast throw-ins) and long
  in-play stoppages (injuries while play continues) are hard cases.
- Identity switches are handled implicitly, because only team membership and the nearest player
  are used. Duplicate tracks near the ball can still bias the distance margin.
- Possible extensions: duration-aware decoding (a hidden semi-Markov model), using the broadcast
  ATD to fill ball gaps after temporal alignment, and ball-height cues from the video for aerial
  play.

## 9. Reproduction

```bash
pip install -r requirements.txt
python run.py --data-dir data --out submissions/Quantix.csv
```
