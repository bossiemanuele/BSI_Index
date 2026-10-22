# Boltzmann–Shannon Index: Revised Implementation

Python implementation and reproducible experiments for **Boltzmann–Shannon Index: A Measure of Density Balance in Clustered Data**.

The Boltzmann–Shannon Index (BSI) compares the probability mass assigned to each labeled state with its normalized geometric extent:

$$
\mathrm{BSI}(\mathbf p,\mathbf q)=1-\mathrm{JSD}_2(\mathbf p\Vert\mathbf q).
$$

The index lies in [0, 1] and equals one when the two distributions agree. It measures density balance relative to the specified labels and extent estimator; it does not measure general clustering quality, separation, or agreement with ground-truth classes.

## Project files

| File | Purpose |
| --- | --- |
| `bsi.py` | BSI calculation using supplied distributions, supplied extents, or labeled observations. |
| `run_experiments.py` | Runs the three studies and generates figures and numerical outputs. |
| `test_bsi.py` | Tests boundaries, zero probabilities, agreement with SciPy, covariance calculations, invariance, invalid inputs, and additive coarsening. |
| `readme.md` | Setup, usage, and methodological notes. |
| `requirements.txt` | Exact dependency versions used for the reported experiments. |

Keep these five files in the same directory.

## Installation

The reference environment uses **Python 3.12**. Open a terminal in the project directory and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

A virtual environment is recommended. For example, create one with `python -m venv .venv` and activate it before installing dependencies.

## Run the tests

```bash
python test_bsi.py
```

A successful run reports five passing test groups and ends with `OK`.

## Reproduce the experiments

```bash
python run_experiments.py
```

No separate data download is required after dependencies are installed. Synthetic samples are generated locally, and Iris is loaded from scikit-learn. Figures are saved automatically; no interactive plotting window is required.

The script runs:

1. **Controlled density imbalance:** known Gaussian population targets, 100 replicates at each of 61 spread settings, a comparison with raw SVD products, a sample-size analysis, and a separation control.
2. **Partition sensitivity:** exact interval masses and lengths under nested refinement, changing boundaries at fixed cluster count, and 30 K-means initializations for each K from 2 through 8.
3. **Iris case study:** species labels and K-means labels, covariance versus convex-hull extents, 50 K-means initializations, and 1,000 fixed-label stratified bootstrap samples for each labeling.

Seeds are specified in the script. Small floating-point differences or clustering ties may occur across platforms or numerical-library versions.

## Output location

The supplied script defines its output root as:

```python
ROOT = Path(__file__).resolve().parents[1]
```

This creates `fig/` and `results/` **in the parent directory of the directory containing the script**, regardless of the terminal's working directory.

For a standalone folder containing the five files listed above, change that line to the following if you want generated outputs inside that same folder:

```python
ROOT = Path(__file__).resolve().parent
```

The script creates the output directories automatically. Re-running it overwrites matching generated files.

| Directory | Contents |
| --- | --- |
| `fig/` | Three multi-panel figures, each in vector PDF and high-resolution PNG format. |
| `results/` | Replicate-level CSVs, Iris observations and selected labels, `summary.json`, and `environment.json` with package versions and an Iris data hash. |

The three figure basenames are `study1_controlled`, `study2_partition`, and `study3_iris`.

## Use BSI in your own analysis

For two normalized distributions:

```python
from bsi import distributions_bsi

score = distributions_bsi([0.7, 0.2, 0.1], [1 / 3, 1 / 3, 1 / 3])
print(score)  # approximately 0.892077
```

For state counts and supplied geometric extents:

```python
from bsi import from_extents

result = from_extents([70, 20, 10], [7, 2, 1])
print(result['bsi'])  # 1.0
print(result['p'])
print(result['q'])
```

For labeled observations, let `X` be an N-by-d numeric array and `labels` a length-N label vector:

```python
from bsi import labeled_bsi

result = labeled_bsi(X, labels, extent='covariance')
print(result['bsi'])
print(result['p'])
print(result['q'])
```

Use `extent='hull'` for a convex-hull sensitivity analysis. The selected estimator is part of the interpretation of the result.

## Methodological notes

- Covariance extents are `sqrt(det(sample covariance))`, computed through centered singular values with the normalization `(N_k - 1)**(d / 2)`. Raw singular-value products include an additional sample-count factor and are not equivalent.
- The covariance implementation uses all features and requires at least d+1 observations spanning the ambient space in every state. It rejects undersampled or numerically rank-deficient states.
- No automatic dimensionality truncation or covariance regularization is applied.
- Zero-probability terms are handled using the usual JSD convention. Cells where only one distribution is positive are retained. Supplied extents must have a positive total.
- Covariance extents summarize second moments. They are sensitive to outliers and do not generally estimate support volumes.
- For fixed labels, covariance-based BSI is invariant under a common invertible affine transformation. Re-fitting a clustering algorithm after transformation can change the labels.
- Additive coarsening cannot decrease BSI when masses and extents are both summed. Recomputed covariance extents are not additive and need not satisfy this ordering.
- Iris K-means is selected by minimum inertia, not maximum BSI. Bootstrap ranges condition on the observed labels and counts; they do not include clustering-selection uncertainty or establish confidence-interval coverage.

## Data attribution

Iris: R. A. Fisher, *Iris*, UCI Machine Learning Repository, [doi:10.24432/C56C76](https://doi.org/10.24432/C56C76), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The experiments use the corrected version distributed with scikit-learn; see the [dataset documentation](https://scikit-learn.org/1.8/modules/generated/sklearn.datasets.load_iris.html).

This folder contains the revised covariance-normalized implementation. Results from the earlier raw-SVD implementation should not be treated as interchangeable with its outputs.
