# SPDX-License-Identifier: GPL-3.0-or-later
"""
Prediction utilities mirroring `fastLPR_predict.m`.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .structures import RegressionOutput


def fastlpr_predict(
    regression: RegressionOutput, x_new: Iterable[Iterable[float]] | np.ndarray
) -> np.ndarray:
    """
    Evaluate a fitted fastLPR regression object at new predictor locations.

    Port from MATLAB's fastLPR_predict.m. Uses the fitted interpolator from
    cv_fastlpr to predict response values at arbitrary new points within
    the data range. Supports real and complex-valued predictions.

    Parameters
    ----------
    regression : RegressionOutput
        Result from :func:`fastlpr.api.cv_fastlpr`.
        Must contain an interpolant (automatically created by cv_fastlpr).
    x_new : array-like
        Predictor locations at which to evaluate the fitted regression.
        Shape: (n_samples, n_features).
        Should be within the range of the training data for reliable predictions.

    Returns
    -------
    np.ndarray
        Predicted values at ``x_new``.
        Shape: (n_samples,) for scalar responses.
        Preserves complex values if the fitted model is complex-valued.

    Examples
    --------
    1D regression prediction at new points:

    >>> import numpy as np
    >>> from fastlpr import cv_fastlpr, fastlpr_predict, get_hlist
    >>>
    >>> # Generate training data
    >>> np.random.seed(42)
    >>> x_train = np.random.rand(500, 1)
    >>> y_train = np.sin(2*np.pi*x_train).ravel() + 0.1*np.random.randn(500)
    >>>
    >>> # Fit regression with local linear (order=1)
    >>> hlist = get_hlist(20, [[0.01, 0.5]])
    >>> opt = {'order': 1}
    >>> regs = cv_fastlpr(x_train, y_train, hlist, opt)
    >>>
    >>> # Predict at new points
    >>> x_new = np.linspace(0, 1, 100).reshape(-1, 1)
    >>> y_pred = fastlpr_predict(regs, x_new)
    >>> print(f"Predictions shape: {y_pred.shape}")  # (100,)

    2D regression prediction on a grid:

    >>> # Generate 2D training data
    >>> np.random.seed(42)
    >>> x_train = np.random.rand(1000, 2)
    >>> y_train = (np.sin(2*np.pi*x_train[:, 0]) *
    ...            np.cos(2*np.pi*x_train[:, 1]) +
    ...            0.1*np.random.randn(1000))
    >>>
    >>> # Fit 2D regression
    >>> hlist = get_hlist([10, 10], [[0.05, 0.5], [0.05, 0.5]])
    >>> regs = cv_fastlpr(x_train, y_train, hlist)
    >>>
    >>> # Predict on a 50x50 evaluation grid
    >>> x1_grid = np.linspace(0, 1, 50)
    >>> x2_grid = np.linspace(0, 1, 50)
    >>> X1, X2 = np.meshgrid(x1_grid, x2_grid)
    >>> x_new = np.column_stack([X1.ravel(), X2.ravel()])
    >>> y_pred = fastlpr_predict(regs, x_new)
    >>> y_pred_grid = y_pred.reshape(50, 50)

    Complex-valued regression (e.g., for spectral analysis):

    >>> # Generate complex-valued training data
    >>> np.random.seed(42)
    >>> x_train = np.random.rand(500, 1)
    >>> y_train = (np.exp(1j*2*np.pi*x_train.ravel()) +
    ...            0.1*(np.random.randn(500) + 1j*np.random.randn(500)))
    >>>
    >>> # Fit complex regression
    >>> regs = cv_fastlpr(x_train, y_train, get_hlist(20, [[0.01, 0.5]]))
    >>>
    >>> # Predict (preserves complex values)
    >>> x_new = np.array([[0.25], [0.5], [0.75]])
    >>> y_pred = fastlpr_predict(regs, x_new)
    >>> print(f"Complex prediction: {y_pred[0]}")  # complex value

    MATLAB equivalence:

    >>> # MATLAB: yhat_new = fastLPR_predict(regs, x_new);
    >>> # Python:
    >>> yhat_new = fastlpr_predict(regs, x_new)

    Notes
    -----
    - Predictions outside the training data range use extrapolation
      (linear by default, matching MATLAB's griddedInterpolant)
    - For reliable predictions, keep x_new within training data bounds
    - Interpolation uses cubic splines by default for smooth results
    - Complex-valued predictions are supported (magnitude and phase preserved)
    - No refitting is needed - uses precomputed interpolant from cv_fastlpr

    See Also
    --------
    cv_fastlpr : Fit regression model with bandwidth selection
    fastlpr_interval : Compute intervals (CI and PI) at new points
    """

    if regression.fpp_yhat is None:
        raise ValueError("Regression output does not contain an interpolant (fpp).")

    points = np.asarray(x_new, dtype=float)
    if points.ndim == 1:
        points = points[:, None]

    predictions = regression.fpp_yhat(points)
    # Preserve complex values if present
    return np.asarray(predictions)
