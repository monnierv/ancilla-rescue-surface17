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

Use `--stress-test` to inject one guaranteed X fault and one guaranteed Z fault per configured node, running 1,000 trials for each node/error combination by default. It shows decoded logical-failure percentages on the node map. The configured `--p` noise is also active, so these are conditional failure rates in that background-noise model. At 1,000 trials the rates change in 0.1-percentage-point increments; increase `--stress-trials` for finer estimates. `--save-map` saves the annotated map.

```bash
python -m surface17 --stress-test --rounds 3 --p 0.001 --seed 7 --save-map stress.png
```

## Square-8 Perimeter Demo

The separate `square8.py` script models eight data qubits around a square and four edge ancillas that repeatedly measure commuting X- and Z-type checks. It reports temporal syndrome changes and tracks one logical observable, `Z0 Z2`, including raw flips and PyMatching-decoded logical failures. The four checks encode multiple logical degrees of freedom, so this is not a complete decoder for the entire code.

```bash
python square8.py --shots 1000 --rounds 3 --p 0.001 --seed 7 --plot
```

Use `--save-map square8.png` to save the node map without opening a plot window.
Add `--ancilla-errors` to include explicit syndrome-ancilla reset and measurement bit-flip errors. The usual depolarizing noise on two-qubit data-ancilla gates remains controlled by `--p` either way.

Square-8 Plus uses eight data qubits, eight data-check ancillas, and a ninth ancilla (A9) to measure the Y parity of the other eight ancillas in a separate diagnostic cycle. Its four X checks act on data pairs; its four weight-4 Z checks include one dependency, leaving one encoded logical qubit. The A9 diagnostic contributes its own detector and does not replace the data-syndrome readout.

```bash
python -m square8 --config square8_plus.json --plot
```

The same `--stress-test` mode is available here:

```bash
python square8.py --stress-test --rounds 3 --p 0.001 --seed 7 --save-map square8-stress.png
```

## Code Comparison

Compare decoded logical error rate against physical error probability for Surface-17 and Square-8 Plus:

```bash
python -m compare_codes --shots 5000 --rounds 3 --p-min 0.0001 --p-max 0.1 --points 20 --seed 7 --save logical-error-rate.png
```

The physical-error sweep is logarithmically spaced. The plot uses a logarithmic x-axis, reports logical error rates as percentages, and includes binomial standard-error bars. Use `--show` with `--save` to display and save the graph in one run; omit `--save` to display it interactively.

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