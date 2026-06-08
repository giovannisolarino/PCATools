# pca_tools

Scikit-learn-native cross-validated selection of the number of PCA components.

`PCACV` is a `*CV`-style self-validating PCA transformer: it chooses the number of
components by cross-validated PRESS, then behaves as a fitted PCA
(`transform` / `inverse_transform`).

## Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -e .
```

This installs `numpy`, `matplotlib` and `scikit-learn` (>=1.6). The editable
(`-e`) install means edits to `pca_tools.py` take effect without reinstalling.

## Usage

```python
import numpy as np
from pca_tools import PCACV

X = np.random.default_rng(0).normal(size=(50, 10))

# Cross-validated component selection, then use as a fitted PCA
model = PCACV(val_procedure="ekf", cv=5).fit(X)
print(model.n_components_)     # selected number of components
scores = model.transform(X)    # project onto n_components_
model.plot()                   # CV error (+/- 1 SE) vs # components
```

## Validation procedures (`val_procedure`)

`"rkf"` : row-wise k-fold; monotone PRESS, good as a quick visual guide.

`"ekf"` : element-wise k-fold (trimmed-score); for missing-data recovery.

`"cekf"` : corrected element-wise k-fold; for compression / dimensionality reduction.

`"em"` : EM missing-data cross-validation; robust, independent predictions.

Other useful options : `selection="min"` or `"1se"`, `scale=True/False`,
`n_components_values=...`, `n_var_blocks=...`, `random_state=...`.
See the docstrings in `pca_tools.py` for the full API.
