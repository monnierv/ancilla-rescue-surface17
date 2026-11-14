# Ancilla Rescue: Surface-17

A small single-logical-qubit memory simulation using the distance-3 rotated surface code. Stim generates the noisy syndrome-measurement circuit, and PyMatching decodes its detection events. The project is a starting point for exploring syndrome analysis and ancilla-assisted error correction.

## Setup

```bash
python -m pip install -e .
```

## Run

```bash
python -m surface17 --shots 1000 --rounds 3 --p 0.001 --seed 7
```

The report includes detection-event counts, raw logical measurement flips, and the logical failure rate after decoding. The `--plot` option displays a node map; `--save-map surface17.png` saves one without opening a window.

Run the checks with:

```bash
python -m unittest discover -s tests
```