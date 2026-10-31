# Team possession estimator: More than a Hack 2027, Phase 1

Labels every frame of the BAR-ATH tactical video as `H` / `A` / `D` (`X` outside the graded
halves), from the provided automatic tracking (ATD) and Smart Tagging. numpy only, no training
labels, deterministic.

The approach is described in [TECHNICAL.md](TECHNICAL.md).

## Install

Python 3.8 or newer.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data

The match data is gated and is **not** in this repository. Unpack the bundle you received after
enrolment so that it looks like this (only the `tactical/` files are used):

```
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_smart_tagging.xml
```

You can also keep it elsewhere and pass `--data-dir` (or set `MTAH_DATA_DIR`).

## Reproduce the submission

```bash
python run.py --data-dir data --out submissions/Quantix.csv
```

That one command regenerates the submitted file, byte for byte. It takes a minute or so and
about 1.5 GB of RAM.

To also run the official validator and the public 5-minute scorer, clone the challenge repository
next to this one and point at it:

```bash
git clone https://github.com/metrica-sports/more-than-a-hack-2027
python run.py --data-dir data --out submissions/Quantix.csv --challenge-repo more-than-a-hack-2027
```

The public 5 minutes are a pipeline check only. Per the rules, nothing in this repository was
tuned on them.

## Layout

```
run.py                     entry point: data in, submission CSV out
possession/config.py       every constant and model parameter, with its rationale
possession/data_io.py      ATD / Smart Tagging readers, CSV writer
possession/features.py     per-frame evidence: ball control, dead ball, team shape
possession/tagging.py      Smart Tagging as soft evidence (set pieces, phase tags)
possession/model.py        3-state Viterbi segmentation + self-supervised shape model
tools/make_synthetic.py    synthetic match in the bundle format, for testing without the data
```

## Testing without the real data

```bash
python tools/make_synthetic.py --out /tmp/synth
python run.py --data-dir /tmp/synth --out /tmp/synth_out/ours.csv
python more-than-a-hack-2027/evaluation/run_evaluation.py --gt /tmp/synth/gt.csv --results /tmp/synth_out
```

The synthetic match is a crude simulator. It checks that the code is correct, not how well it
scores on real football.
