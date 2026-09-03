import os
import time

import numpy as np
from numpy.fft import fft, ifft
from scipy.io import loadmat

try:
    import mat73
except ImportError:
    mat73 = None


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "datasets",
)
# =============================================================================
# Data Loading
# =============================================================================

def load_mat(filename):
    """
    Load a MATLAB .mat file.

    Standard MATLAB files are loaded using scipy.io.loadmat.
    MATLAB v7.3 files are loaded using mat73 when necessary.

    Parameters
    ----------
    filename : str
        Name of the MATLAB data file.

    Returns
    -------
    dict
        Dictionary containing variables stored in the .mat file.
    """
    path = os.path.join(DATA_DIR, filename)

    try:
        data = loadmat(path)
        return {
            key: value
            for key, value in data.items()
            if not key.startswith("__")
        }

    except NotImplementedError:
        if mat73 is None:
            raise RuntimeError(
                "MATLAB v7.3 files require the 'mat73' package."
            )

        return mat73.loadmat(path)


def load_data(dataset):
    """
    Load and construct the tensor corresponding to a given dataset.

    Parameters
    ----------
    dataset : str
        Dataset identifier.

    Returns
    -------
    ndarray
        Third-order tensor used by TC-RLA.
    """
    dataset = dataset.lower()

    if dataset in ("abilene", "a"):
        original_data = load_mat("Abilene.mat")["Abilene"]

        temp_data = np.reshape(
            original_data,
            (12, 12, 288, 7),
            order="F",
        )

        temp_data = np.transpose(
            temp_data,
            (2, 3, 0, 1),
        )

        x0 = np.reshape(
            temp_data,
            (288, 7, 144),
            order="F",
        )

        return x0.astype(float) * 1000.0

    if dataset in ("guangzhou", "g"):
        guangzhou = load_mat("guangzhou.mat")["guangzhou"]

        x0 = np.transpose(
            guangzhou,
            (1, 2, 0),
        )

        return x0.astype(float)

    if dataset in ("nasdaq", "nasdaq-100", "n"):
        x0 = load_mat("nasdaq.mat")["X"]
        return x0.astype(float)

    if dataset in ("solar", "s"):
        x0 = load_mat("solar.mat")["X"]
        return x0.astype(float)

    raise ValueError(f"Unknown dataset: {dataset}")


# =============================================================================
# Missing-Pattern Generation
# =============================================================================

def build_mask(dim, missing_rate, missing_way, seed):
    """
    Generate the observation mask.

    Parameters
    ----------
    dim : tuple
        Tensor dimensions.
    missing_rate : float
        Fraction of missing entries.
    missing_way : str
        Missing pattern. Supported options are:
        "Random" and "Non-random2".
    seed : int
        Random seed.

    Returns
    -------
    ndarray
        Binary observation mask, where 1 denotes an observed entry.
    """
    rng = np.random.default_rng(seed)

    if missing_way == "Random":
        pomega = np.round(
            rng.random(dim) + 0.5 - missing_rate
        )

    elif missing_way == "Non-random2":
        base_mask = np.round(
            rng.random((dim[0], dim[2]))
            + 0.5
            - missing_rate
        )

        expanded_mask = np.kron(
            base_mask,
            np.ones((1, dim[1])),
        )

        pomega = np.reshape(
            expanded_mask,
            dim,
            order="F",
        )

    else:
        raise ValueError(
            f"Unknown missing pattern: {missing_way}"
        )

    return pomega


# =============================================================================
# Circulant Matrix Construction
# =============================================================================

def circvet2mat(x):
    """
    Construct a circulant matrix from a one-dimensional vector.

    Parameters
    ----------
    x : array_like
        Input vector.

    Returns
    -------
    ndarray
        Circulant matrix generated from the input vector.
    """
    x = np.asarray(x, dtype=float).ravel()

    return np.vstack(
        [
            np.roll(x, shift)
            for shift in range(len(x))
        ]
    )


# =============================================================================
# Tensor Unfolding and Folding
# =============================================================================

def unfold(x, mode):
    """
    Unfold a tensor along the specified mode.

    Parameters
    ----------
    x : ndarray
        Input tensor.
    mode : int
        Tensor mode, indexed from 1.

    Returns
    -------
    ndarray
        Mode-wise unfolding matrix.
    """
    dim = x.shape
    mode -= 1

    axes = (
        list(range(mode, x.ndim))
        + list(range(mode))
    )

    x_unfold = np.reshape(
        np.transpose(x, axes),
        (dim[mode], -1),
        order="F",
    )

    return x_unfold


def fold(mat, dim, mode):
    """
    Fold a mode-wise matrix back into a tensor.

    Parameters
    ----------
    mat : ndarray
        Unfolded matrix.
    dim : tuple
        Target tensor dimensions.
    mode : int
        Tensor mode, indexed from 1.

    Returns
    -------
    ndarray
        Reconstructed tensor.
    """
    mode -= 1

    new_dim = (
        list(dim[mode:])
        + list(dim[:mode])
    )

    tensor = np.reshape(
        mat,
        new_dim,
        order="F",
    )

    axes = (
        list(range(mode, len(dim)))
        + list(range(mode))
    )

    tensor = np.transpose(
        tensor,
        np.argsort(axes),
    )

    return tensor


# =============================================================================
# Proximal Operator for the Sum of Nuclear Norms
# =============================================================================

def prox_snn(
    y,
    threshold,
    weights=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
):
    """
    Compute the weighted proximal-average update for the SNN regularizer.

    Singular value thresholding is applied independently to each
    mode unfolding, followed by weighted averaging.

    Parameters
    ----------
    y : ndarray
        Input third-order tensor.
    threshold : float
        Singular value threshold.
    weights : tuple of float, optional
        Weights associated with the three tensor modes.

    Returns
    -------
    ndarray
        Tensor obtained from the weighted SNN proximal update.
    """
    dim = y.shape
    result = np.zeros_like(y, dtype=float)

    for mode in range(1, 4):
        y_mat = unfold(y, mode)

        u, s, vh = np.linalg.svd(
            y_mat,
            full_matrices=False,
        )

        s = np.maximum(
            s - threshold,
            0.0,
        )

        temp = (u * s) @ vh
        temp = fold(temp, dim, mode)

        result += weights[mode - 1] * temp

    return result


# =============================================================================
# MATLAB Burg Autoregressive Estimation
# =============================================================================

class MatlabBurg:
    """
    MATLAB interface for Burg autoregressive coefficient estimation.

    The MATLAB function `arburg` is called through MATLAB Engine for Python.
    """

    def __init__(self):
        import matlab
        import matlab.engine

        self.matlab = matlab
        self.engine = matlab.engine.start_matlab()

    def arburg(self, x, order):
        """
        Estimate autoregressive coefficients using MATLAB `arburg`.

        Parameters
        ----------
        x : ndarray
            Input matrix.
        order : int
            Autoregressive order.

        Returns
        -------
        ndarray
            Estimated AR coefficient matrix.
        """
        x_matlab = self.matlab.double(
            np.asarray(
                x,
                dtype=float,
            ).tolist()
        )

        coeff = self.engine.arburg(
            x_matlab,
            float(order),
        )

        coeff = np.asarray(
            coeff,
            dtype=float,
        )

        if coeff.ndim == 1:
            coeff = coeff[None, :]

        return coeff

    def close(self):
        """Terminate the MATLAB Engine session."""
        self.engine.quit()


# =============================================================================
# TC-RLA
# =============================================================================

def tc_rla(
    dataset,
    missing_rate,
    missing_way,
    seed=49,
    detail=True,
):
    """
    Perform tensor completion using TC-RLA.

    Parameters
    ----------
    dataset : str
        Dataset name or abbreviation.
    missing_rate : float
        Fraction of missing entries.
    missing_way : str
        Missing pattern.
    seed : int, optional
        Random seed used to generate the observation mask.
    detail : bool, optional
        If True, print iteration information.

    Returns
    -------
    x : ndarray
        Recovered tensor.
    mae : float
        Mean absolute error.
    rmse : float
        Root mean squared error.
    runtime : float
        Total running time in seconds.
    """

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    x0 = load_data(dataset)
    dim = x0.shape

    # -------------------------------------------------------------------------
    # Model and optimization parameters
    # -------------------------------------------------------------------------

    p = 7

    max_iter = 200
    outer_max_iter = 20

    rho = 1.1

    mu1 = 1e-6
    mu2 = 1e-6
    max_mu = 1e10

    tol = 1e-2
    outer_tol = 1e-1

    eps = 1e-6

    # -------------------------------------------------------------------------
    # Observation mask
    # -------------------------------------------------------------------------

    pomega = build_mask(
        dim,
        missing_rate,
        missing_way,
        seed,
    )

    pomegac = 1.0 - pomega

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    x = pomega * x0
    px0 = pomega * x0

    # AR operator D and its spectral representation
    d = np.zeros(
        (dim[0], dim[0], dim[2])
    )

    t_spec = np.zeros(dim)

    # -------------------------------------------------------------------------
    # Start MATLAB Engine
    # -------------------------------------------------------------------------

    burg = MatlabBurg()
    start_time = time.perf_counter()

    try:
        outer_iter = 0

        # =====================================================================
        # Outer alternating loop
        # =====================================================================

        while outer_iter < outer_max_iter:
            outer_iter += 1

            if detail:
                print(
                    f"\n=== Outer Iteration {outer_iter} ==="
                )

            x_prev = x.copy()

            # Reset ADMM parameters and auxiliary variables
            # at the beginning of each outer iteration.
            mu1_current = mu1
            mu2_current = mu2

            e = np.zeros(dim)
            lam = np.zeros(dim)
            gamma = np.zeros(dim)
            g = np.zeros(dim)

            # -----------------------------------------------------------------
            # Update autoregressive coefficients
            #
            # MATLAB equivalent:
            #
            # MMM = reshape(X, [dim(1) * dim(2), dim(3)]);
            # vv  = fliplr(arburg(MMM, p));
            # -----------------------------------------------------------------

            mat = np.reshape(
                x,
                (dim[0] * dim[1], dim[2]),
                order="F",
            )

            vv = burg.arburg(mat, p)
            vv = np.fliplr(vv)

            # -----------------------------------------------------------------
            # Construct the AR operator D and its Fourier spectrum
            # -----------------------------------------------------------------

            for i in range(dim[2]):
                row = np.concatenate(
                    [
                        vv[i, :],
                        np.zeros(dim[0] - p - 1),
                    ]
                )

                d[:, :, i] = circvet2mat(row)

                spectrum = (
                    np.abs(
                        fft(
                            row.reshape(-1, 1),
                            axis=0,
                        )
                    )
                    ** 2
                )

                t_spec[:, :, i] = np.tile(
                    spectrum,
                    (1, dim[1]),
                )

            # =================================================================
            # Inner ADMM loop
            # =================================================================

            inner_iter = 0

            while inner_iter < max_iter:
                inner_iter += 1

                xk = x.copy()
                ek = e.copy()

                # -------------------------------------------------------------
                # Update X
                # -------------------------------------------------------------

                for i in range(dim[2]):
                    w = d[:, :, i].T @ (
                        g[:, :, i]
                        - gamma[:, :, i] / mu2_current
                    )

                    w += (
                        px0[:, :, i]
                        - e[:, :, i]
                        + lam[:, :, i] / mu1_current
                    )

                    x[:, :, i] = np.real(
                        ifft(
                            fft(w, axis=0)
                            / (1.0 + t_spec[:, :, i]),
                            axis=0,
                        )
                    )

                # -------------------------------------------------------------
                # Compute D X
                # -------------------------------------------------------------

                dx = np.zeros(dim)

                for i in range(dim[2]):
                    dx[:, :, i] = (
                        d[:, :, i] @ x[:, :, i]
                    )

                # -------------------------------------------------------------
                # Update G
                # -------------------------------------------------------------

                j = np.transpose(
                    dx + gamma / mu2_current,
                    (2, 0, 1),
                )

                jj = prox_snn(
                    j,
                    1.0 / mu2_current,
                )

                g = np.transpose(
                    jj,
                    (1, 2, 0),
                )

                # -------------------------------------------------------------
                # Update E
                # -------------------------------------------------------------

                e = (
                    -x
                    + lam / mu1_current
                )

                e = pomegac * e

                # -------------------------------------------------------------
                # Stopping criterion
                # -------------------------------------------------------------

                dy = px0 - x - e

                chg_x = np.max(
                    np.abs(xk - x)
                )

                chg_e = np.max(
                    np.abs(ek - e)
                )

                chg = max(
                    chg_x,
                    chg_e,
                    np.max(np.abs(dy)),
                )

                # -------------------------------------------------------------
                # Update Lagrange multipliers and penalty parameters
                # -------------------------------------------------------------

                lam = (
                    lam
                    + mu1_current * dy
                )

                gamma = (
                    gamma
                    + mu2_current * (dx - g)
                )

                mu1_current = min(
                    rho * mu1_current,
                    max_mu,
                )

                mu2_current = min(
                    rho * mu2_current,
                    max_mu,
                )

                # Terminate the current inner loop once converged.
                if chg < tol:
                    break

            # =================================================================
            # Outer-loop stopping criterion
            # =================================================================

            rel_change = (
                np.linalg.norm(x - x_prev)
                / (
                    np.linalg.norm(x_prev)
                    + eps
                )
            )

            if detail:
                mae = np.mean(
                    np.abs(x0 - x)
                )

                rmse = np.sqrt(
                    np.mean(
                        (x0 - x) ** 2
                    )
                )

                print(
                    f"Outer iteration : {outer_iter:3d} | "
                    f"Inner iteration : {inner_iter:3d} | "
                    f"Relative change : {rel_change:.4e} | "
                    f"MAE : {mae:.6f} | "
                    f"RMSE : {rmse:.6f}"
                )

            if rel_change < outer_tol:
                break

    finally:
        burg.close()

    # =========================================================================
    # Final evaluation
    # =========================================================================

    diff = x0 - x

    mae = np.mean(
        np.abs(diff)
    )

    rmse = np.sqrt(
        np.mean(diff ** 2)
    )

    runtime = (
        time.perf_counter()
        - start_time
    )

    return x, mae, rmse, runtime


# =============================================================================
# Example Experiment
# =============================================================================

if __name__ == "__main__":

    dataset = "nasdaq"
    missing_way = "Random"
    missing_rate = 0.2
    seed = 1000

    x, mae, rmse, runtime = tc_rla(
        dataset=dataset,
        missing_rate=missing_rate,
        missing_way=missing_way,
        seed=seed,
        detail=True,
    )

    print("\n" + "=" * 60)
    print("TC-RLA Experiment Summary")
    print("=" * 60)
    print(f"Dataset       : {dataset}")
    print(f"Missing type  : {missing_way}")
    print(f"Missing rate  : {missing_rate:.2f}")
    print(f"Random seed   : {seed}")
    print(f"MAE           : {mae:.6f}")
    print(f"RMSE          : {rmse:.6f}")
    print(f"Runtime       : {runtime:.3f} s")
    print("=" * 60)
