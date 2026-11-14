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

Add `--ancilla-errors` to include explicit syndrome-ancilla reset and measurement bit-flip errors:

```bash
python -m surface17 --p 0.001 --ancilla-errors
```

## Square-8 Perimeter Demo

The separate `square8.py` script models eight data qubits around a square and four edge ancillas that repeatedly measure commuting X- and Z-type checks. It reports temporal syndrome changes and tracks one logical observable, `Z0 Z2`, including raw flips and PyMatching-decoded logical failures. The four checks encode multiple logical degrees of freedom, so this is not a complete decoder for the entire code.

```bash
python square8.py --shots 1000 --rounds 3 --p 0.001 --seed 7 --plot
```

Use `--save-map square8.png` to save the node map without opening a plot window.
Add `--ancilla-errors` to include explicit syndrome-ancilla reset and measurement bit-flip errors. The usual depolarizing noise on two-qubit data-ancilla gates remains controlled by `--p` either way.

## Code Configuration

The default layouts are defined in `surface17.json` and `square8.json`. Each file lists qubits with `id`, `kind`, and `x`/`y` coordinates, checks with an ancilla and X/Z basis, ancilla-to-data `connections`, and Z-basis `logical_observables`. Both programs validate that mixed X/Z checks commute before building the Stim circuit. Supply `--config PATH` to either command to run a different compatible layout.

```bash
python -m surface17 --config surface17.json
python square8.py --config square8.json
```

Run the checks with:

```bash
python -m unittest discover -s tests
```