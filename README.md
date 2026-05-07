# cellular-complexity

> Japanese version: [README.ja.md](https://github.com/kanekoyuichi/cellular-complexity/blob/main/README.ja.md)

A tool that exhaustively searches all **262,144** rule combinations of two-dimensional binary cellular automata (Life-like CA) and extracts structurally complex rules as a Pareto-optimal set.

## Background

Cellular automata such as Conway's Game of Life produce complex behavior from simple rules. The number of possible Birth/Survival condition combinations is 512×512 = 262,144. This tool evaluates all of them and searches for rules that exhibit complexity similar to Conway's Life.

## Metrics

Three metrics are simultaneously maximized via multi-objective optimization (Pareto optimality).

| Metric | Description | When high |
|--------|-------------|-----------|
| **C** (Clustering) | Spatial autocorrelation. Tendency of live cells to cluster together | Cells aggregate and form patterns |
| **O** (Organization) | Self-organization. Degree to which the rate of change decreases in the latter half of simulation | Transition from chaos to order |
| **S** (Structure) | Pattern structure. Degree to which a small number of patterns dominate | Specific patterns appear in an organized form |

These metrics are designed to be low for rules that maintain random noise and high for structure-generating rules like Conway's Life.

## Installation

```bash
pip install numpy matplotlib pillow
```

Python 3.10 or higher and NumPy 1.24 or higher are recommended.

## Usage

### Full Rule Search

```bash
cd src/cellular-complexity
python3 full_rule_pareto_search.py --output-dir ../../output
```

Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--size` | 32 | Board size (square) |
| `--steps` | 50 | Maximum steps per rule |
| `--trials` | 1 | Number of trials per rule |
| `--init-live-prob` | 0.5 | Initial live cell probability |
| `--workers` | CPU count | Number of parallel processes |
| `--output-dir` | . | Output directory |

Output files:
- `all_rules_results.csv` — Metric values for all 262,144 rules
- `pareto_front_results.csv` — Pareto-optimal rules

### Visualization

```bash
python3 visualize.py B3/S23
python3 visualize.py B3/S23 --size 64 --steps 200
python3 visualize.py B3/S23 --output output/conway.gif
```

In environments with a display, shows an interactive viewer (Space: pause, R: reset).
In headless environments (e.g. SSH), automatically saves a GIF to `output/`.

## File Structure

```
cellular-complexity/
  src/cellular-complexity/
    full_rule_pareto_search.py   # Search engine (metric calculation, Pareto extraction)
    visualize.py                 # Rule visualization and GIF generation
  output/                        # Results (not tracked by git)
```

## Implementation Notes

- **Performance**: Neighbor counting via `np.pad` + slice addition; 3x3 pattern aggregation via `sliding_window_view` + bit packing
- **Parallelism**: `multiprocessing.Pool` utilizing all CPU cores
- **Memory efficiency**: Streaming CSV writes avoid holding all results in memory
- **Reproducibility**: Fixed random seed ensures reproducible results

## Pareto Optimality Definition

Rule `a` dominates rule `b` if:
- `a >= b` on all three metrics (C, O, S)
- and `a > b` on at least one metric

The Pareto-optimal set consists of all rules that are not dominated by any other rule.
