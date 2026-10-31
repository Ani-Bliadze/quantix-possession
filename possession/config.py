"""Constants and model parameters.

Two kinds of numbers live here:

* Match constants published by the organisers (frame count, graded half blocks, pitch size).
  These are boundaries, not answers, and DATA.md says they may be used.
* Model parameters. Every one is a football / physics prior chosen *before* looking at any
  labels -- none was fitted to the public 5-minute ground truth (RULES.md forbids that). Where a
  quantity depends on this match's data distribution (activity level, attack direction, tag
  polarity, shape model) it is estimated from the tracking itself, without labels.
"""

# --- match constants (BAR-ATH, from the challenge repository) --------------------------------
N_FRAMES = 199319
FPS = 25
PITCH = (105.3, 68.0)                       # metres; games.py value
GRADED_HALVES = ((11185, 90651), (110808, 190197))   # inclusive video-frame blocks (DATA.md)

TACTICAL_STEM = "BAR-ATH_tactical"

# --- tracking preprocessing --------------------------------------------------------------------
BALL_INTERP_MAX_GAP = 8          # frames; linearly bridge ball gaps up to 0.32 s
BALL_SPEED_SMOOTH = 5            # frames, median filter on ball speed
OUT_OF_BOUNDS_MARGIN = 1.0       # metres beyond the lines before we call the ball out

# --- ball-based team evidence (per frame, log-odds units) ---------------------------------------
TEAM_GAIN = 1.0                  # max log-odds per frame from a controlled ball
TEAM_MARGIN_SCALE = 1.5          # metres: tanh scale on (d_away - d_home)
CONTROL_RADIUS = 1.0             # metres: inside this the nearest player has the ball at his feet
CONTROL_DECAY = 1.5              # metres: e-folding of control beyond CONTROL_RADIUS
BALL_FLIGHT_SPEED = 10.0         # m/s: above this the ball is travelling, evidence fades

# --- dead-ball evidence -------------------------------------------------------------------------
OUT_GAIN = 2.0                   # per frame, ball tracked outside the pitch
STATIC_BALL_SPEED = 0.6          # m/s
STATIC_BALL_WINDOW = 25          # frames: ball static for 1 s
STATIC_GAIN = 0.6                # per frame
ALIVE_MOVING_GAIN = 0.4          # per frame, ball visible, in bounds, moving, near a player
ACTIVITY_WINDOW = 50             # frames: rolling mean of median player speed (2 s)
ACTIVITY_GAIN = 0.5              # scales the GMM posterior log-odds of "low activity"
ACTIVITY_CLIP = 2.0
MISSING_LONG = 75                # frames: ball unseen for 3 s strengthens activity evidence
MISSING_LONG_BOOST = 1.5

# --- Smart Tagging ------------------------------------------------------------------------------
SETPIECE_DEAD_GAIN = 0.5         # per frame, D evidence from tag start until detected restart
SETPIECE_TEAM_GAIN = 0.5         # per frame, team evidence for the restarting team
SETPIECE_TEAM_FRAMES = 50        # 2 s after the restart
PHASE_TEAM_GAIN = 0.12           # per frame, team evidence from phase tags
PHASE_MIN_AGREEMENT = 0.15       # |mean signed agreement| needed to trust a code's polarity

# --- self-supervised team-shape model -------------------------------------------------------------
SHAPE_GAIN_MISSING = 0.25        # per frame, multiplier on shape log-odds when ball is unseen
SHAPE_GAIN_VISIBLE = 0.05        # ... when the ball is seen
SHAPE_CLIP = 3.0                 # clip shape log-odds before the gain
PSEUDO_MIN_SPELL = 125           # frames: only possessions >= 5 s from pass 1 feed pseudo-labels
PSEUDO_TRIM = 25                 # frames trimmed from each end of those spells
PSEUDO_MAX_SAMPLES = 60000

# --- segmentation (Viterbi) ---------------------------------------------------------------------
SWITCH_PENALTY = 12.0            # log-odds cost of H<->A: needs ~0.5 s of clear control to flip
DEAD_PENALTY = 10.0              # cost of entering / leaving D
