"""
pca_tools
=========
Scikit-learn-native model selection for PCA.

Public API
----------
* `PCACV` -- a `*CV`-style self-validating PCA transformer that
  chooses the number of components by cross-validated PRESS. Supports
  row-wise (`rkf`), closed-form element-wise (`ekf`), corrected
  element-wise (`cekf`) and EM missing-data (`em`) validation. After
  fitting it behaves as a fitted PCA (`transform`/`inverse_transform`).

Conventions
-----------
`PCACV` follows the scikit-learn `RidgeCV`/`LassoCV` pattern: the
cross-validation is internal, the chosen hyper-parameter is exposed as
`n_components_`, and (when `refit=True`) the model is refit on the full
data and exposed through the transformer interface. Splitting is delegated to
scikit-learn splitters; PCA mean-centers internally as usual. Preprocessing is
owned by the estimator and refit inside every fold (no leakage): pass any
transformer (or transformer `Pipeline`) via `preprocessor`, or use the
`scale` shortcut for unit-variance scaling. PRESS is the only selection
criterion.

References
----------
- Camacho & Ferrer, J. Chemometrics 26(7), 2012, 361-373 (ekf, theory).
- Camacho & Ferrer, Chemom. Intell. Lab. Syst. 131, 2014, 37-50 (cekf).
- Bro, R., Kjeldahl, K., Smilde, A. K., & Kiers, H. A. L. (2008), 390(5), 1241-1251
  (EM cross-validation).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted, validate_data


# ===========================================================================
#                       EM missing-data PCA helper
# ===========================================================================

def _empca(Xp: np.ndarray, n_components: int,
           max_iter: int = 2000, tol: float = 1e-9) -> np.ndarray:
    """
    EM (iterative-imputation) PCA reconstruction of a centered/scaled matrix
    containing NaNs. Missing entries are filled with 0 (the centered mean),
    PCA is refit, missing entries are replaced by the rank-`n_components`
    reconstruction, and the loop repeats to convergence. This is the
    least-squares missing-data fit underlying the EM cross-validation of
    Bro et al. (2008). Returns the full reconstruction.
    """
    mask = np.isnan(Xp)
    X = np.where(mask, 0.0, Xp)
    if not mask.any():
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        P = Vt[:n_components]
        return (X @ P.T) @ P
    prev = None
    for _ in range(max_iter):
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        P = Vt[:n_components]
        Xrec = (X @ P.T) @ P
        X = np.where(mask, Xrec, X)
        cur = X[mask]
        if prev is not None and np.linalg.norm(cur - prev) <= tol * (
                np.linalg.norm(prev) + 1e-12):
            break
        prev = cur.copy()
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    P = Vt[:n_components]
    return (X @ P.T) @ P


# ===========================================================================
#           PCACV -- *CV-style self-validating PCA transformer
# ===========================================================================

class PCACV(TransformerMixin, BaseEstimator):
    r"""
    PCA with cross-validated selection of the number of components.

    A `*CV`-style estimator: cross-validation is internal, the selected
    component count is exposed as `n_components_`, and (with `refit=True`)
    the estimator behaves as a fitted PCA transformer over `n_components_`
    components.

    Parameters
    ----------
    n_components_values : iterable of int or None, default None
        Candidate component counts (>= 1). If None, uses
        `range(1, max_feasible + 1)`.
    val_procedure : {'rkf', 'ekf', 'cekf', 'em'}, default 'rkf'
        Cross-validation criterion:

        - `'rkf'`  : row-wise k-fold. Whole samples are held out and
          reconstructed. The held-out scores use the held-out samples
          themselves, so the PRESS curve is (near-)monotone in the number of
          components and has no interior minimum -- combine with
          `selection='1se'` or inspect `cv_results_`.
        - `'ekf'`  : closed-form element-wise k-fold with the trimmed-score
          (direct) imputation correction (Camacho & Ferrer 2012). Cheap.
        - `'cekf'` : corrected element-wise k-fold (Camacho & Ferrer 2014).
          Augments the calibration block with the model reconstruction of the
          held-out variables, refits PCA, and predicts. Recovers
          non-redundant information; valley-shaped PRESS. Expensive.
        - `'em'`   : element-wise EM missing-data cross-validation
          (Bro et al. 2008). Held-out element blocks are imputed by a
          least-squares PCA fit that excludes them, so predictions are
          independent of the predicted entity. Preprocessing is recomputed
          inside each fold from observed entries only.
    cv : int or scikit-learn splitter, default 5
        Row cross-validation. An int `k` means `KFold(k, shuffle=True,
        random_state=random_state)`. Defines the held-out samples for every
        procedure.
    n_var_blocks : int, default 7
        Number of variable (column) folds for `ekf`/`cekf`/`em`
        (>= 2). Ignored by `rkf`. Columns are split with `KFold`.
    preprocessor : scikit-learn transformer or None, default None
        Preprocessing applied before PCA. It is cloned and refit inside every
        fold (and for the final refit), so no information leaks from held-out
        samples into the preprocessing. May be a single transformer or a
        transformer `Pipeline` (e.g. `make_pipeline(SNV(), StandardScaler())`).
        Supersedes `scale` when not None. Not supported with
        `val_procedure='em'` (see `scale`).
    scale : bool, default True
        Shortcut used only when `preprocessor is None`: `True` prepends a
        `StandardScaler` (unit-variance scaling), `False` applies no
        preprocessing. For `val_procedure='em'` this flag controls whether
        the in-fold, observed-only preprocessing is autoscaling (`True`) or
        mean-centering only (`False`). PCA mean-centers in either case.
    selection : {'min', '1se'}, default 'min'
        `'min'` picks the component count minimising the mean CV error.
        `'1se'` picks the smallest count whose mean CV error is within one
        standard error of the minimum (parsimonious; useful for the monotone
        `rkf` curve).
    refit : bool, default True
        If True, refit on the full data with `n_components_` components and
        expose the transformer interface and `best_estimator_`.
    em_max_iter, em_tol : int, float
        EM convergence controls (`em` only).
    random_state : int or None, default None
        Seed for `KFold` shuffling (row and column folds) when ints are
        passed.

    Attributes
    ----------
    n_components_ : int
        Selected number of components.
    n_components_values_ : ndarray of int
        Candidate counts actually evaluated (sorted).
    mean_cv_error_, std_cv_error_ : ndarray
        Mean and std (across folds) of the per-sample PRESS for each candidate.
    cv_results_ : dict
        `{'n_components', 'mean_cv_error', 'std_cv_error'}`.
    best_estimator_ : Pipeline
        Full-data refit (if `refit`).
    components_, mean_, explained_variance_, explained_variance_ratio_ :
        Delegated from the refit PCA (if `refit`).
    n_features_in_ : int
        Number of features seen during fit.
    """

    def __init__(
        self,
        n_components_values=None,
        val_procedure: Literal["rkf", "ekf", "cekf", "em"] = "rkf",
        cv=5,
        n_var_blocks: int = 7,
        preprocessor=None,
        scale: bool = True,
        selection: Literal["min", "1se"] = "min",
        refit: bool = True,
        em_max_iter: int = 2000,
        em_tol: float = 1e-9,
        random_state: int | None = None,
    ) -> None:
        self.n_components_values = n_components_values
        self.val_procedure = val_procedure
        self.cv = cv
        self.n_var_blocks = n_var_blocks
        self.preprocessor = preprocessor
        self.scale = scale
        self.selection = selection
        self.refit = refit
        self.em_max_iter = em_max_iter
        self.em_tol = em_tol
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, X, y=None) -> "PCACV":
        """Run the cross-validation, select `n_components_`, optionally refit."""
        X = validate_data(self, X, accept_sparse=False, ensure_2d=True,
                           dtype=float, reset=True)
        N, M = X.shape

        if self.val_procedure not in ("rkf", "ekf", "cekf", "em"):
            raise ValueError("'val_procedure' must be 'rkf', 'ekf', 'cekf' or 'em'.")
        if self.selection not in ("min", "1se"):
            raise ValueError("'selection' must be 'min' or '1se'.")
        if self.preprocessor is not None:
            if not (hasattr(self.preprocessor, "fit")
                    and hasattr(self.preprocessor, "transform")):
                raise TypeError(
                    "'preprocessor' must be a scikit-learn transformer "
                    "(with fit/transform) or None.")
            if self.val_procedure == "em":
                raise ValueError(
                    "'preprocessor' is not supported with val_procedure='em'. "
                    "The EM procedure recomputes column centering/scaling from "
                    "observed entries inside each fold, which a generic "
                    "transformer cannot satisfy. Use 'scale' (mean-centering or "
                    "autoscaling) for 'em', or pick another procedure.")

        splitter = self._resolve_cv()
        folds = list(splitter.split(X))
        n_folds = len(folds)
        if n_folds < 2:
            raise ValueError("cv must yield at least 2 folds.")

        if self.val_procedure == "em":
            max_feasible = min(N - 1, M - 1)
        else:
            min_train = min(len(tr) for tr, _ in folds)
            max_feasible = min(min_train - 1, M - 1)
        if max_feasible < 1:
            raise ValueError("Not enough samples/features for any component.")

        if self.n_components_values is None:
            grid = np.arange(1, max_feasible + 1)
        else:
            grid = np.unique(np.asarray(self.n_components_values, dtype=int))
            if np.any(grid < 1):
                raise ValueError("'n_components_values' must be >= 1.")
            if grid.max() > max_feasible:
                raise ValueError(
                    f"max(n_components_values)={grid.max()} exceeds the feasible "
                    f"limit {max_feasible} (procedure='{self.val_procedure}', "
                    f"cv folds={n_folds}, N={N}, M={M})."
                )

        if self.val_procedure in ("ekf", "cekf", "em"):
            if self.n_var_blocks < 2 or self.n_var_blocks > M:
                raise ValueError("'n_var_blocks' must be in [2, M].")
            col_blocks = self._col_blocks(M)
        else:
            col_blocks = None

        errors = np.empty((len(grid), n_folds))   # per-sample PRESS per fold

        for f, (train, test) in enumerate(folds):
            n_test = len(test)
            if self.val_procedure == "em":
                press = self._press_fold_em(X, test, grid, col_blocks)
            else:
                press = self._press_fold_projection(X, train, test, grid,
                                                     col_blocks)
            errors[:, f] = press.sum(axis=1) / n_test

        self.n_components_values_ = grid
        self.mean_cv_error_ = errors.mean(axis=1)
        self.std_cv_error_ = errors.std(axis=1, ddof=1) / np.sqrt(n_folds)
        self.cv_results_ = {
            "n_components": grid,
            "mean_cv_error": self.mean_cv_error_,
            "std_cv_error": self.std_cv_error_,
        }
        self.n_components_ = self._select(grid)

        if self.refit:
            self.best_estimator_ = self._make_pipeline(self.n_components_).fit(X)
            pca = self.best_estimator_["pca"]
            self.components_ = pca.components_
            self.mean_ = pca.mean_
            self.explained_variance_ = pca.explained_variance_
            self.explained_variance_ratio_ = pca.explained_variance_ratio_
        return self

    # ------------------------------------------------------------------
    # Per-fold PRESS: projection-based procedures (rkf / ekf / cekf)
    # ------------------------------------------------------------------
    def _press_fold_projection(self, X, train, test, grid, col_blocks):
        pipe = self._make_pipeline(int(grid.max())).fit(X[train])
        pca = pipe["pca"]
        ccs = self._pre(pipe, X[train])
        scs = self._pre(pipe, X[test])
        mu = pca.mean_
        Pfull = pca.components_.T                         # (M, kmax)
        M = X.shape[1]

        press = np.zeros((len(grid), M))
        for a, k in enumerate(grid):
            p2 = Pfull[:, :k]
            T = (scs - mu) @ p2
            srec = T @ p2.T + mu

            if self.val_procedure == "rkf":
                press[a] = np.sum((scs - srec) ** 2, axis=0)

            elif self.val_procedure == "ekf":
                erec = scs - srec
                base = scs - mu
                if self.n_var_blocks == M:
                    term1p = base * np.sum(p2 * p2, axis=1)[None, :]
                else:
                    term1p = np.zeros_like(erec)
                    for cb in col_blocks:
                        term1p[:, cb] = base[:, cb] @ (p2[cb] @ p2[cb].T)
                press[a] = np.sum((term1p + erec) ** 2, axis=0)

            elif self.val_procedure == "cekf":
                press[a] = self._cekf_pem(ccs, scs, p2, mu, int(k), col_blocks)
        return press

    def _cekf_pem(self, ccs, scs, p2, mu, k, col_blocks):
        """Corrected element-wise per-variable PRESS (Camacho & Ferrer 2014)."""
        tcest = (ccs - mu) @ p2
        tsest = (scs - mu) @ p2
        rec = tcest @ p2.T + mu
        recsam = tsest @ p2.T + mu
        srec = (scs - mu) @ p2 @ p2.T + mu
        for cb in col_blocks:
            aug = np.hstack([ccs, rec[:, cb]])
            pca3 = PCA(n_components=k, svd_solver="full").fit(aug)
            p3 = pca3.components_.T
            mu3 = pca3.mean_
            scs2 = np.hstack([scs, recsam[:, cb]])
            scs2[:, cb] = mu[cb]
            pred = ((scs2 - mu3) @ p3) @ p3.T + mu3
            srec[:, cb] = pred[:, cb]
        return np.sum((scs - srec) ** 2, axis=0)

    # ------------------------------------------------------------------
    # Per-fold PRESS: EM missing-data procedure
    # ------------------------------------------------------------------
    def _press_fold_em(self, X, test, grid, col_blocks):
        M = X.shape[1]
        press = np.zeros((len(grid), M))
        for cb in col_blocks:
            Xm = X.copy()
            Xm[np.ix_(test, cb)] = np.nan
            mu = np.nanmean(Xm, axis=0)
            if self.scale:
                sd = np.nanstd(Xm, axis=0, ddof=1)
                sd[~np.isfinite(sd) | (sd == 0)] = 1.0
            else:
                sd = np.ones(M)
            Xp = (Xm - mu) / sd
            true = (X[np.ix_(test, cb)] - mu[cb]) / sd[cb]
            for a, k in enumerate(grid):
                recon = _empca(Xp, int(k), self.em_max_iter, self.em_tol)
                pred = recon[np.ix_(test, cb)]
                press[a, cb] += np.sum((true - pred) ** 2, axis=0)
        return press

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def _select(self, grid):
        m = self.mean_cv_error_
        kmin = int(np.argmin(m))
        if self.selection == "min":
            return int(grid[kmin])
        thresh = m[kmin] + self.std_cv_error_[kmin]
        within = np.where(m <= thresh)[0]
        return int(grid[within.min()])      # smallest count within 1 SE

    # ------------------------------------------------------------------
    # Transformer interface (delegates to the refit PCA)
    # ------------------------------------------------------------------
    def transform(self, X):
        """Project `X` onto `n_components_` components."""
        check_is_fitted(self, "best_estimator_")
        X = validate_data(self, X, reset=False, dtype=float)
        return self.best_estimator_.transform(X)

    def inverse_transform(self, X):
        """Map scores back to the original feature space."""
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.inverse_transform(X)

    def score(self, X, y=None):
        """Negative mean squared reconstruction error (higher is better)."""
        check_is_fitted(self, "best_estimator_")
        X = validate_data(self, X, reset=False, dtype=float)
        Xrec = self.best_estimator_.inverse_transform(
            self.best_estimator_.transform(X))
        return -float(np.mean((np.asarray(X, float) - Xrec) ** 2))

    def plot(self):
        """Plot mean CV error (+/- 1 SE) vs number of components."""
        check_is_fitted(self, "n_components_")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(self.n_components_values_, self.mean_cv_error_,
                    yerr=self.std_cv_error_, marker="o", lw=1.8, capsize=3)
        ax.set_xlabel("Nr. of components")
        ax.set_ylabel("PRESS")
        ax.set_title(f"PCACV ({self.val_procedure})")
        ax.set_xticks(self.n_components_values_)
        fig.tight_layout()
        plt.close(fig)
        return fig

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _resolve_cv(self):
        if isinstance(self.cv, int):
            return KFold(n_splits=self.cv, shuffle=True,
                         random_state=self.random_state)
        return self.cv

    def _col_blocks(self, M):
        kf = KFold(n_splits=self.n_var_blocks, shuffle=True,
                   random_state=self.random_state)
        return [c for _, c in kf.split(np.arange(M))]

    def _resolve_preprocessor(self):
        """Return a fresh, unfitted preprocessing transformer, or None."""
        if self.preprocessor is not None:
            return clone(self.preprocessor)
        if self.scale:
            return StandardScaler()
        return None

    def _make_pipeline(self, k):
        steps = []
        pre = self._resolve_preprocessor()
        if pre is not None:
            steps.append(("pre", pre))
        steps.append(("pca", PCA(n_components=int(k), svd_solver="full")))
        return Pipeline(steps)

    def _pre(self, pipe, data):
        """Transform through the pre-PCA step of a fitted pipeline, if present."""
        if "pre" in pipe.named_steps:
            return np.asarray(pipe["pre"].transform(data), dtype=float)
        return np.asarray(data, dtype=float)
