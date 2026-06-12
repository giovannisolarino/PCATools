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

`"rkf"` : row-wise k-fold; monotone PRESS

`"ekf"` : element-wise k-fold (trimmed-score)

`"cekf"` : corrected element-wise k-fold;

`"em"` : EM missing-data cross-validation;

Other useful options : `selection="min"`, `scale=True/False`,
`n_components_values=...`, `n_var_blocks=...`, `random_state=...`.
See the docstrings in `pca_tools.py` for the full API.

## References
- Camacho & Ferrer, J. Chemometrics 26(7), 2012, 361-373 https://doi.org/10.1002/cem.2440 (ekf, theory).
- Camacho & Ferrer, Chemom. Intell. Lab. Syst. 131, 2014, 37-50 https://doi.org/10.1016/j.chemolab.2013.12.003 (cekf).
- Bro, R., Kjeldahl, K., Smilde, A. K., & Kiers, H. A. L. (2008), 390(5), 1241-1251. https://link.springer.com/article/10.1007/s00216-007-1790-1 (EM cross-validation).

## Cite

If you use this package, please cite the repository and the papers in the References section.