"""
Beam model implementations. These are different ways of modeling the
temperature and polarization beams. They inherit from a base class and
are intended to present a uniform beam model for use in the fitter and plotter.
"""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
from scipy import interpolate
from scipy.integrate import simpson

from .utils import linear_interp_differentiable


class BeamModelBase(ABC):
    """Abstract base class for all beam models."""

    def __init__(self, config, y_grid, x_grid, band):
        self.config = config
        self.y_grid = y_grid
        self.x_grid = x_grid
        self.band = band
        self.band_GHz = self.band.replace("GHz", "")

    @abstractmethod
    def get_initial_physical_params(self):
        """Returns a dictionary of initial beam parameters in physical space."""
        pass

    @abstractmethod
    def evaluate_beam_maps(self, params, dy, dx):
        """
        Evaluates the T and P beam maps.

        Args:
            params (dict): Dictionary containing the beam parameters.
            dy (float): Source y-offset.
            dx (float): Source x-offset.

        Returns:
            tuple: A tuple containing the T and P beam maps.
        """
        pass

    @abstractmethod
    def get_human_readable_params(self, params):
        """Returns a human-readable dictionary of parameters."""
        pass

    @abstractmethod
    def get_profiles_for_plotting(self, params):
        """
        Returns (r_fine, profile_T, profile_P, info_dict) for plotting.
        info_dict should include model-specific label strings and normalization info.
        """
        pass

    def _calculate_r_map(self, dy, dx, eps=1e-12):
        """
        Calculate radial distance map from source position.

        A tiny epsilon is added to make sure this is differentiable.
        """
        dx_diff = self.x_grid - dx
        dy_diff = self.y_grid - dy
        r_sq = dx_diff * dx_diff + dy_diff * dy_diff
        return jnp.sqrt(r_sq + eps) * self.config.reso_arcmin


class BeamModelGaussian(BeamModelBase):
    """
    A simple Gaussian beam model with separate T and P widths.
    """

    def __init__(self, config, y_grid, x_grid, band):
        super().__init__(config, y_grid, x_grid, band)
        self.param_names = ["T_width_arcmin", "P_width_arcmin"]
        self.n_fittable_coeffs = 2

    def get_initial_physical_params(self):
        """Returns the initial guess for T and P widths."""
        # Use band FWHM as initial guess for both T and P
        band_fwhm = self.config.band_fwhm_arcmin[self.band]
        return {"T_width_arcmin": band_fwhm, "P_width_arcmin": band_fwhm}

    def evaluate_beam_maps(self, params, dy, dx):
        """
        Evaluates the T and P beam maps for the Gaussian model.
        """
        T_width_arcmin = params["T_width_arcmin"]
        P_width_arcmin = params["P_width_arcmin"]

        sigma_T_arcmin = T_width_arcmin / (2 * jnp.sqrt(2 * jnp.log(2)))
        sigma_P_arcmin = P_width_arcmin / (2 * jnp.sqrt(2 * jnp.log(2)))

        r_map_arcmin = self._calculate_r_map(dy, dx)
        r_sq_arcmin = r_map_arcmin**2

        beam_T_map = jnp.exp(-0.5 * r_sq_arcmin / sigma_T_arcmin**2)
        beam_P_map = jnp.exp(-0.5 * r_sq_arcmin / sigma_P_arcmin**2)
        return beam_T_map, beam_P_map

    def get_human_readable_params(self, params, prefix=""):
        """Returns a human-readable dictionary of the parameters."""
        return {
            f"{prefix}T_width_arcmin": float(params["T_width_arcmin"]),
            f"{prefix}P_width_arcmin": float(params["P_width_arcmin"]),
        }

    def get_profiles_for_plotting(self, params):
        r_max = 10.0
        r_fine = np.linspace(0, r_max, 1000)
        T_width_arcmin = params["T_width_arcmin"]
        P_width_arcmin = params["P_width_arcmin"]
        T_sigma = T_width_arcmin / (2 * np.sqrt(2 * np.log(2)))
        P_sigma = P_width_arcmin / (2 * np.sqrt(2 * np.log(2)))
        profile_T_fine = np.exp(-0.5 * (r_fine / T_sigma) ** 2)
        profile_P_fine = np.exp(-0.5 * (r_fine / P_sigma) ** 2)
        info = {
            "t_label": f"Best-Fit T-Beam Profile (FWHM={T_width_arcmin:.2f} arcmin)",
            "p_label": f"Best-Fit P-Beam Profile (FWHM={P_width_arcmin:.2f} arcmin)",
            "ylabel": "Beam Amplitude",
            "normalization": None,
        }
        return r_fine, profile_T_fine, profile_P_fine, info


class BeamModelBetaPol(BeamModelBase):
    """
    A beam model that interpolates between
     - BT_r(r), the AGN+Saturn stitched beam
     - Bmain_r(r), the physical model fit to the multiband inner 0.75 arcmin of the stitched beam
    Uses a single free parameter, beta_pol.
    Temperature beam(r) = BT_r(r)
    Polarization beam(r) = Bmain_r(r) + beta_pol * (BT_r(r) - Bmain_r(r))
    """

    def __init__(self, config, y_grid, x_grid, band):
        super().__init__(config, y_grid, x_grid, band)
        self.param_names = ["beta_pol"]
        self.n_fittable_coeffs = 1

        # Load the pre-calculated beam profiles
        data_path = config.betapol_data_path
        try:
            beam_data = np.load(data_path)
            print(f"Loaded betapol data from {data_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find the betapol data file at {data_path}.")

        # Check if the requested band is available
        bt_key = f"BT_r_norm_{self.band_GHz}"
        bmain_key = f"Bmain_r_norm_{self.band_GHz}"

        if bt_key not in beam_data:
            available_bands = [key.split("_")[-1] for key in beam_data.keys() if key.startswith("BT_r_norm_")]
            raise KeyError(f"Band {self.band_GHz} GHz not found in betapol data. Available bands: {available_bands} GHz")

        # Store JAX arrays for fast computation
        self.r_fine_jax = jax.device_put(beam_data["r_fine_arcmin"].astype(self.config.dtype_jax_real))
        self.BT_r_norm_jax = jax.device_put(beam_data[bt_key].astype(self.config.dtype_jax_real))
        self.Bmain_r_norm_jax = jax.device_put(beam_data[bmain_key].astype(self.config.dtype_jax_real))

        # Store numpy copies to avoid importing JAX in plotting
        self.r_fine_np = beam_data["r_fine_arcmin"]
        self.BT_r_norm_np = beam_data[bt_key]
        self.Bmain_r_norm_np = beam_data[bmain_key]

    def get_initial_physical_params(self):
        """Returns the initial guess for beta_pol."""
        return {"beta_pol": 0.5}  # Start in the middle

    def evaluate_beam_profile_P(self, params, r_values):
        """
        Calculates the P beam profile for a given beta_pol.
        P beam = Bmain_r + beta_pol * (BT_r - Bmain_r)

        Args:
            params (dict): Dictionary containing beam parameters.
            r_values (jnp.ndarray): The radial points at which to evaluate the profile.
        """
        beta_pol = params["beta_pol"]

        # P beam: interpolate between Bmain and BT
        profile_fine = self.Bmain_r_norm_jax + beta_pol * (self.BT_r_norm_jax - self.Bmain_r_norm_jax)

        # Interpolate the final profile onto the requested r_values
        return linear_interp_differentiable(r_values, self.r_fine_jax, profile_fine, self.config)

    def evaluate_beam_maps(self, params, dy, dx):
        r_map_arcmin = self._calculate_r_map(dy, dx)
        beam_T_map = linear_interp_differentiable(r_map_arcmin, self.r_fine_jax, self.BT_r_norm_jax, self.config)
        beam_P_map = self.evaluate_beam_profile_P(params, r_map_arcmin)
        return beam_T_map, beam_P_map

    def get_human_readable_params(self, params, prefix=""):
        """Returns a human-readable dictionary of the parameters."""
        return {f"{prefix}beta_pol": float(params["beta_pol"])}

    def get_profiles_for_plotting(self, params):
        beta_pol = params["beta_pol"]
        r_fine = self.r_fine_np
        profile_T_fine = self.BT_r_norm_np
        profile_P_fine = self.Bmain_r_norm_np + beta_pol * (self.BT_r_norm_np - self.Bmain_r_norm_np)
        info = {
            "t_label": "T-Beam Profile (BT_r)",
            "p_label": f"P-Beam Profile (Bmain + β_pol×(BT-Bmain), β_pol={beta_pol:.3f})",
            "ylabel": "Beam Amplitude",
            "normalization": "pre-normalized",
        }
        return r_fine, profile_T_fine, profile_P_fine, info


class BeamModelBetaTest(BeamModelBase):
    """
    Pretty much a copy of BeamModelBetaPol but swapping the T and P
    models. This is a test suggested by SWLH to see if we recover
    beta_T=1.0.

    Uses a single parameter, beta_T, for interpolation.
    B_T_model = Bmain_r(r) + beta_T * (BT_r(r) - Bmain_r(r))
    B_P_model = Bmain_r(r)
    """

    def __init__(self, config, y_grid, x_grid, band):
        super().__init__(config, y_grid, x_grid, band)
        self.param_names = ["beta_T"]
        self.n_fittable_coeffs = 1

        # Load the pre-calculated beam profiles
        data_path = config.betapol_data_path
        try:
            beam_data = np.load(data_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find the betapol data file at {data_path}.")

        # Check if the requested band is available
        bt_key = f"BT_r_norm_{self.band_GHz}"
        bmain_key = f"Bmain_r_norm_{self.band_GHz}"

        if bt_key not in beam_data:
            available_bands = [key.split("_")[-1] for key in beam_data.keys() if key.startswith("BT_r_norm_")]
            raise KeyError(f"Band {self.band_GHz} GHz not found in betapol data. Available bands: {available_bands} GHz")

        print(f"Loading betatest beam model for {self.band_GHz} GHz")

        # Store JAX arrays for fast computation
        self.r_fine_jax = jax.device_put(beam_data["r_fine_arcmin"].astype(self.config.dtype_jax_real))
        self.BT_r_norm_jax = jax.device_put(beam_data[bt_key].astype(self.config.dtype_jax_real))
        self.Bmain_r_norm_jax = jax.device_put(beam_data[bmain_key].astype(self.config.dtype_jax_real))

        # Store numpy copies to avoid importing JAX in plotting
        self.r_fine_np = beam_data["r_fine_arcmin"]
        self.BT_r_norm_np = beam_data[bt_key]
        self.Bmain_r_norm_np = beam_data[bmain_key]

    def get_initial_physical_params(self):
        """Returns the initial guess for beta_T."""
        return {"beta_T": 0.5}  # Start in the middle

    def evaluate_beam_profile_T(self, params, r_values):
        """
        Calculates the T beam profile for a given beta_T.
        T beam = Bmain_r + beta_T * (BT_r - Bmain_r)

        Args:
            params (dict): Dictionary containing beam parameters.
            r_values (jnp.ndarray): The radial points at which to evaluate the profile.
        """
        beta_T = params["beta_T"]

        # T beam: interpolate between Bmain and BT
        profile_fine = self.Bmain_r_norm_jax + beta_T * (self.BT_r_norm_jax - self.Bmain_r_norm_jax)

        # Interpolate the final profile onto the requested r_values
        return linear_interp_differentiable(r_values, self.r_fine_jax, profile_fine, self.config)

    def evaluate_beam_maps(self, params, dy, dx):
        r_map_arcmin = self._calculate_r_map(dy, dx)
        beam_T_map = self.evaluate_beam_profile_T(params, r_map_arcmin)
        beam_P_map = linear_interp_differentiable(r_map_arcmin, self.r_fine_jax, self.Bmain_r_norm_jax, self.config)
        return beam_T_map, beam_P_map

    def get_human_readable_params(self, params, prefix=""):
        """Returns a human-readable dictionary of the parameters."""
        return {f"{prefix}beta_T": float(params["beta_T"])}

    def get_profiles_for_plotting(self, params):
        beta_T = params["beta_T"]
        r_fine = self.r_fine_np
        profile_T_fine = self.Bmain_r_norm_np + beta_T * (self.BT_r_norm_np - self.Bmain_r_norm_np)
        profile_P_fine = self.Bmain_r_norm_np
        info = {
            "t_label": f"T-Beam Profile (Bmain + β_T×(BT-Bmain), β_T={beta_T:.3f})",
            "p_label": "P-Beam Profile (Bmain_r)",
            "ylabel": "Beam Amplitude",
            "normalization": "pre-normalized",
        }
        return r_fine, profile_T_fine, profile_P_fine, info


class BeamModelBSplinesGaussian(BeamModelBase):
    """
    A beam model combining a central Gaussian with area-normalized B-splines.

    The model is: B(r) = Gaussian(r; sigma) + sum_i c_i * phi_i(r)

    where:
    - Gaussian has variable sigma
    - B-splines start at 0.5 arcmin (no central region)
    - B-splines are orthonormalized with 2D area normalization: ∫phi_i(r)2πr dr = 1
    - Endpoint constraints on B-splines
    """

    def __init__(self, config, y_grid, x_grid, band):
        super().__init__(config, y_grid, x_grid, band)
        self.param_names = [
            "gaussian_sigma_arcmin",
            "bspline_coeffs_T",
            "bspline_coeffs_P",
        ]
        self.spline_k = config.spline_k
        self.bspline_rmax_arcmin = config.bspline_rmax_arcmin
        self.spline_rmax_arcmin = self.bspline_rmax_arcmin
        self.knot_spacing_arcmin = config.knot_spacing_arcmin
        self.spline_rmin_arcmin = config.bsplines_gaussian_rmin_arcmin
        bspline_rmax_value = config.bspline_rmax_values[self.band]
        self.bspline_rmax_value = None if bspline_rmax_value is None else float(bspline_rmax_value)
        self.bspline_force_rmax_derivative_zero = config.bspline_force_rmax_derivative_zero

        # Setup the B-spline basis (area-normalized, no boundary conditions)
        (
            r_fine_jax,
            self.ortho_basis_funcs_jax,
            self.n_bspline_coeffs,
            self.r_knots,
        ) = self._setup_bspline_lut()

        # Total number of fittable coefficients: 1 (Gaussian sigma) + N (B-spline coeffs)
        self.n_fittable_coeffs = 1 + self.n_bspline_coeffs
        self.r_fine_jax = r_fine_jax
        beam_data = np.load(config.betapol_data_path)
        r_measured = beam_data["r_fine_arcmin"].astype(self.config.dtype_np_real)
        beam_measured = beam_data[f"BT_r_norm_{self.band_GHz}"].astype(self.config.dtype_np_real)
        r_target = np.array(self.r_fine_jax, dtype=self.config.dtype_np_real)
        self.measured_beam_profile = np.interp(r_target, r_measured, beam_measured, left=beam_measured[0], right=beam_measured[-1])

    def _bspline_rmax_offset(self, r_values, gaussian_sigma=None):
        if self.bspline_rmax_value is None:
            return jnp.zeros_like(jnp.asarray(r_values, dtype=self.config.dtype_jax_real))

        target = self.bspline_rmax_value
        if gaussian_sigma is not None:
            r_max = self.bspline_rmax_arcmin
            target -= jnp.exp(-0.5 * (r_max / gaussian_sigma) ** 2)

        profile = self.bspline_rmax_profile_jax * target
        return linear_interp_differentiable(r_values, self.r_fine_jax, profile, self.config)

    def _bspline_rmax_offset_np(self, r_values, gaussian_sigma=None):
        if self.bspline_rmax_value is None:
            return np.zeros_like(np.asarray(r_values, dtype=self.config.dtype_np_real))

        target = self.bspline_rmax_value
        if gaussian_sigma is not None:
            r_max = self.bspline_rmax_arcmin
            target -= np.exp(-0.5 * (r_max / gaussian_sigma) ** 2)

        profile = self.bspline_rmax_profile_np * target
        r_fine = np.asarray(self.r_fine_jax, dtype=self.config.dtype_np_real)
        return np.interp(r_values, r_fine, profile, left=0.0, right=profile[-1])

    def _setup_bspline_lut(self):
        """
        Setup area-normalized orthogonal B-splines with endpoint constraints.
        We always enforce B(r_min) = B'(r_min) = 0. If the configured
        bspline_rmax_value is numeric, we also enforce B(r_max) = 0 and,
        by default, B'(r_max) = 0. If bspline_rmax_value is None, no r_max
        value or derivative constraint is applied.
        """
        print("Setting up area-normalized orthogonal B-splines...")
        degree = self.spline_k - 1  # degree = 3 for cubic spline
        r_min = self.spline_rmin_arcmin
        r_max = self.bspline_rmax_arcmin

        # Interior knots
        delta_r = self.knot_spacing_arcmin
        interior_knots = np.arange(r_min + delta_r, r_max, delta_r)

        # Clamped knot vector
        self.r_knots = np.concatenate([np.full(degree + 1, r_min), interior_knots, np.full(degree + 1, r_max)])

        n_coeffs_total = len(self.r_knots) - degree - 1

        # Create standard B-spline basis
        identity_coeffs = np.eye(n_coeffs_total)
        basis_functions = []
        for i in range(n_coeffs_total):
            basis_functions.append(interpolate.BSpline(self.r_knots, identity_coeffs[i], degree))

        # Always join smoothly to the central model at r_min. The r_max value
        # constraint is optional; None means leave the outer endpoint free.
        constrain_rmax_value = self.bspline_rmax_value is not None
        value_row_rmin = np.array([bf(r_min) for bf in basis_functions])
        deriv_row_rmin = np.array([bf.derivative()(r_min) for bf in basis_functions])
        constraint_rows = [value_row_rmin, deriv_row_rmin]
        if constrain_rmax_value:
            value_row_rmax = np.array([bf(r_max) for bf in basis_functions])
            constraint_rows.append(value_row_rmax)
            if self.bspline_force_rmax_derivative_zero:
                deriv_row_rmax = np.array([bf.derivative()(r_max) for bf in basis_functions])
                constraint_rows.append(deriv_row_rmax)
        constraint_matrix = np.vstack(constraint_rows)

        rmax_coeffs = np.zeros(n_coeffs_total)
        if constrain_rmax_value:
            rmax_targets = np.zeros(len(constraint_rows))
            rmax_targets[2] = 1.0
            rmax_coeffs = np.linalg.lstsq(constraint_matrix, rmax_targets, rcond=None)[0]

        print(f"Constraint matrix condition number: {np.linalg.cond(constraint_matrix):.2e}")

        _, singular_values, vh = np.linalg.svd(constraint_matrix)
        if singular_values.size == 0:
            raise ValueError("Failed to build endpoint constraint null space for B-splines.")
        tol = max(1e-10, singular_values[0] * 1e-12)
        rank = np.sum(singular_values > tol)
        null_space_basis = vh[rank:].T
        if null_space_basis.shape[1] == 0:
            raise ValueError("Endpoint constraints remove all B-spline degrees of freedom.")
        print(f"Null space dimension: {null_space_basis.shape[1]} free parameters")

        # Define evaluation grid for area normalization
        n_eval = 2000
        r_eval = np.linspace(r_min, r_max, n_eval)

        # Evaluate all basis functions on the grid
        basis_matrix = np.zeros((n_eval, n_coeffs_total))
        for i, basis_func in enumerate(basis_functions):
            basis_matrix[:, i] = basis_func(r_eval)

        # 2D area normalization: ∫b(r)2πr dr = 1
        # Weight function is 2πr for cylindrical coordinates
        weight_2d = 2 * np.pi * r_eval

        # Compute Gram matrix in the constrained coefficient space
        n_free = null_space_basis.shape[1]
        gram_matrix = np.zeros((n_free, n_free))

        for i in range(n_free):
            func_i = basis_matrix @ null_space_basis[:, i]
            for j in range(i, n_free):
                func_j = basis_matrix @ null_space_basis[:, j]
                # 2D area inner product: ∫f_i(r) * f_j(r) * 2π*r dr
                inner_prod = simpson(func_i * func_j * weight_2d, x=r_eval)
                gram_matrix[i, j] = inner_prod
                gram_matrix[j, i] = inner_prod

        # Orthogonalize using eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(gram_matrix)

        # Keep only positive eigenvalues
        mask = eigvals > 1e-10
        if not np.any(mask):
            raise ValueError("Gram matrix is singular - check B-spline setup")

        # Create orthonormal basis with 2D area normalization
        orthogonal_coeffs = null_space_basis @ (eigvecs[:, mask] @ np.diag(1.0 / np.sqrt(eigvals[mask])))

        # Verify orthonormality
        test_gram = np.zeros((orthogonal_coeffs.shape[1], orthogonal_coeffs.shape[1]))
        for i in range(orthogonal_coeffs.shape[1]):
            func_i = basis_matrix @ orthogonal_coeffs[:, i]
            for j in range(orthogonal_coeffs.shape[1]):
                func_j = basis_matrix @ orthogonal_coeffs[:, j]
                test_gram[i, j] = simpson(func_i * func_j * weight_2d, x=r_eval)

        off_diag_error = np.max(np.abs(test_gram - np.eye(test_gram.shape[0])))
        print(f"2D area orthonormality verification: max off-diagonal = {off_diag_error:.2e}")

        # Verify boundary conditions are satisfied
        boundary_check_rmin = basis_matrix[0, :] @ orthogonal_coeffs
        max_rmin_violation = np.max(np.abs(boundary_check_rmin))

        # Check derivative at boundary
        derivative_at_boundary = np.array([bf.derivative()(r_min) for bf in basis_functions])
        boundary_check_derivative = derivative_at_boundary @ orthogonal_coeffs
        max_derivative_violation = np.max(np.abs(boundary_check_derivative))

        print(f"Boundary condition B(r_min) = 0: max violation = {max_rmin_violation:.2e}")
        print(f"Boundary condition B'(r_min) = 0: max violation = {max_derivative_violation:.2e}")
        if constrain_rmax_value:
            boundary_check_rmax = basis_matrix[-1, :] @ orthogonal_coeffs
            max_rmax_violation = np.max(np.abs(boundary_check_rmax))
            print(f"Boundary condition B(r_max) = 0: max violation = {max_rmax_violation:.2e}")
            if self.bspline_force_rmax_derivative_zero:
                derivative_at_rmax = np.array([bf.derivative()(r_max) for bf in basis_functions])
                boundary_check_derivative_rmax = derivative_at_rmax @ orthogonal_coeffs
                max_derivative_rmax_violation = np.max(np.abs(boundary_check_derivative_rmax))
                print(f"Boundary condition B'(r_max) = 0: max violation = {max_derivative_rmax_violation:.2e}")
            else:
                print("Boundary condition B'(r_max): not enforced")
        else:
            print("Boundary condition B(r_max): not enforced")
            print("Boundary condition B'(r_max): not enforced")
        # Store coefficients for reconstruction
        self.n_bspline_coeffs = orthogonal_coeffs.shape[1]
        self.orthogonal_coeffs = orthogonal_coeffs

        # Create fine-resolution lookup tables for JAX
        r_fine = np.linspace(0, r_max, 1000)  # Start from 0 for the full beam model

        # Orthonormal B-spline basis functions (only defined for r >= r_min)
        ortho_basis_funcs = np.zeros((len(r_fine), self.n_bspline_coeffs))
        rmax_profile = np.zeros(len(r_fine))

        # Only evaluate B-splines where r >= r_min
        mask_spline_region = r_fine >= r_min
        r_spline = r_fine[mask_spline_region]

        if len(r_spline) > 0:
            # Evaluate basis on spline region
            spline_basis_matrix = np.zeros((len(r_spline), n_coeffs_total))
            for i, basis_func in enumerate(basis_functions):
                spline_basis_matrix[:, i] = basis_func(r_spline)

            # Apply orthogonalization to get final basis functions
            for j in range(self.n_bspline_coeffs):
                ortho_func_spline = spline_basis_matrix @ orthogonal_coeffs[:, j]
                ortho_basis_funcs[mask_spline_region, j] = ortho_func_spline
            rmax_profile[mask_spline_region] = spline_basis_matrix @ rmax_coeffs

        # Convert to JAX arrays
        r_fine_jax = jax.device_put(r_fine.astype(self.config.dtype_jax_real))
        ortho_basis_funcs_jax = jax.device_put(ortho_basis_funcs.astype(self.config.dtype_jax_real))
        self.bspline_rmax_profile_np = rmax_profile.astype(self.config.dtype_np_real)
        self.bspline_rmax_profile_jax = jax.device_put(rmax_profile.astype(self.config.dtype_jax_real))

        print("Area-normalized B-spline basis setup complete:")
        print(f"  {len(self.r_knots)} knots, {n_coeffs_total} total B-spline coefficients")
        print(f"  {self.n_bspline_coeffs} orthonormal basis functions")
        if self.bspline_rmax_value is None:
            print("  Endpoint constraints: B=B'=0 at r_min, no constraints at r_max")
        elif self.bspline_force_rmax_derivative_zero:
            print("  Endpoint constraints: B=B'=0 at r_min, B=B'=0 at r_max")
        else:
            print("  Endpoint constraints: B=B'=0 at r_min, B=0 at r_max")
        print(f"  B-splines active over r = [{r_min:.1f}, {r_max:.1f}] arcmin")

        return r_fine_jax, ortho_basis_funcs_jax, self.n_bspline_coeffs, self.r_knots

    def get_initial_physical_params(self):
        """Returns initial guess for all parameters."""
        # Use band FWHM for initial Gaussian sigma (shared between T and P)
        band_fwhm = self.config.band_fwhm_arcmin[self.band]
        initial_sigma = band_fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma

        r_fine_np = np.array(self.r_fine_jax, dtype=self.config.dtype_np_real)
        gaussian_component_np = np.exp(-0.5 * (r_fine_np / initial_sigma) ** 2)
        basis_np = np.array(self.ortho_basis_funcs_jax, dtype=self.config.dtype_np_real)
        rmax_offset_np = self._bspline_rmax_offset_np(r_fine_np, initial_sigma)

        target_profile = self.measured_beam_profile - gaussian_component_np - rmax_offset_np

        weight = 2.0 * np.pi * r_fine_np
        integrand = target_profile[:, None] * basis_np * weight[:, None]
        coeffs_solution = simpson(integrand, x=r_fine_np, axis=0)
        bounds = self.config.active_beam_model_bounds["bspline_coeffs_T"]
        eps = 1e-6
        coeffs_solution = np.clip(coeffs_solution, bounds[0] + eps, bounds[1] - eps)
        coeffs_from_measurement = jnp.asarray(coeffs_solution, dtype=self.config.dtype_jax_real)
        bspline_coeffs_T = coeffs_from_measurement
        bspline_coeffs_P = coeffs_from_measurement

        return {
            "gaussian_sigma_arcmin": initial_sigma,  # Shared between T and P
            "bspline_coeffs_T": bspline_coeffs_T,  # T beam B-spline coefficients
            "bspline_coeffs_P": bspline_coeffs_P,  # P beam B-spline coefficients
        }

    def evaluate_beam_profile(self, gaussian_sigma, bspline_coeffs, r_values):
        """
        Evaluate the combined beam profile at given radial values.

        Parameters:
        -----------
        gaussian_sigma : float
            Gaussian sigma parameter (shared between T and P)
        bspline_coeffs : array_like
            B-spline coefficient array (specific to T or P beam)
        r_values : array_like
            Radial values to evaluate at

        Returns:
        --------
        array_like
            Combined beam profile values
        """
        # Ensure inputs are JAX arrays to avoid device transfers
        r_values = jnp.asarray(r_values, dtype=self.config.dtype_jax_real)
        bspline_coeffs = jnp.asarray(bspline_coeffs, dtype=self.config.dtype_jax_real)

        # Gaussian component (shared)
        gaussian_component = jnp.exp(-0.5 * (r_values / gaussian_sigma) ** 2)

        # B-spline component (beam-specific)
        bspline_profile_fine = jnp.dot(self.ortho_basis_funcs_jax, bspline_coeffs)
        bspline_component = linear_interp_differentiable(r_values, self.r_fine_jax, bspline_profile_fine, self.config)
        bspline_component = bspline_component + self._bspline_rmax_offset(r_values, gaussian_sigma)
        # If an rmax value is configured, set the total beam to that value for r >= rmax
        if self.bspline_rmax_value is not None:
            bspline_component = jnp.where(
                r_values >= self.bspline_rmax_arcmin, self.bspline_rmax_value - gaussian_component, bspline_component
            )

        # Combined beam
        total_beam = gaussian_component + bspline_component

        return total_beam

    def evaluate_beam_maps(self, params, dy, dx):
        r_map_arcmin = self._calculate_r_map(dy, dx)
        beam_T_map = self.evaluate_beam_profile(params["gaussian_sigma_arcmin"], params["bspline_coeffs_T"], r_map_arcmin)
        beam_P_map = self.evaluate_beam_profile(params["gaussian_sigma_arcmin"], params["bspline_coeffs_P"], r_map_arcmin)
        return beam_T_map, beam_P_map

    def get_human_readable_params(self, params, prefix=""):
        """Returns a human-readable dictionary of the parameters."""
        result = {f"{prefix}gaussian_sigma_arcmin": float(params["gaussian_sigma_arcmin"])}
        for i, c in enumerate(params["bspline_coeffs_T"]):
            result[f"{prefix}bspline_coeff_T_{i}"] = float(c)
        for i, c in enumerate(params["bspline_coeffs_P"]):
            result[f"{prefix}bspline_coeff_P_{i}"] = float(c)
        return result

    def get_profiles_for_plotting(self, params):
        gaussian_sigma = params["gaussian_sigma_arcmin"]
        bspline_coeffs_T = params["bspline_coeffs_T"]
        bspline_coeffs_P = params["bspline_coeffs_P"]
        r_fine = np.array(self.r_fine_jax)
        gaussian_component = np.exp(-0.5 * (r_fine / gaussian_sigma) ** 2)
        ortho_funcs = np.array(self.ortho_basis_funcs_jax)
        rmax_offset = self._bspline_rmax_offset_np(r_fine, gaussian_sigma)
        bspline_component_T = ortho_funcs @ bspline_coeffs_T + rmax_offset
        bspline_component_P = ortho_funcs @ bspline_coeffs_P + rmax_offset
        profile_T_fine = gaussian_component + bspline_component_T
        profile_P_fine = gaussian_component + bspline_component_P
        info = {
            "t_label": f"T-Beam Profile (Gaussian σ={gaussian_sigma:.3f} arcmin + B-splines)",
            "p_label": f"P-Beam Profile (Gaussian σ={gaussian_sigma:.3f} arcmin + B-splines)",
            "ylabel": "Beam Amplitude",
            "normalization": None,
        }
        return r_fine, profile_T_fine, profile_P_fine, info


class BeamModelBSplinesPlusMain(BeamModelBase):
    """
    Hybrid beam model using the betapol "full" beam for temperature and
    the "main" beam plus B-splines (starting at 0.75 arcmin) for polarization.
    """

    def __init__(self, config, y_grid, x_grid, band):
        super().__init__(config, y_grid, x_grid, band)
        self.param_names = ["bspline_coeffs_P"]
        self.spline_k = config.spline_k
        self.bspline_rmax_arcmin = config.bspline_rmax_arcmin
        self.spline_rmax_arcmin = self.bspline_rmax_arcmin
        self.knot_spacing_arcmin = config.knot_spacing_arcmin
        self.spline_rmin_arcmin = config.bsplines_main_rmin_arcmin
        bspline_rmax_value = config.bspline_rmax_values[self.band]
        self.bspline_rmax_value = None if bspline_rmax_value is None else float(bspline_rmax_value)
        self.bspline_force_rmax_derivative_zero = config.bspline_force_rmax_derivative_zero

        # Load the pre-calculated beam profiles
        data_path = config.betapol_data_path
        try:
            beam_data = np.load(data_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find the betapol data file at {data_path}.")

        full_key = f"BT_r_norm_{self.band_GHz}"
        main_key = f"Bmain_r_norm_{self.band_GHz}"
        r_key = "r_fine_arcmin"

        if full_key not in beam_data or main_key not in beam_data or r_key not in beam_data:
            available_bands = [key.split("_")[-1] for key in beam_data.keys() if key.startswith("BT_r_norm_")]
            raise KeyError(
                f"Band {self.band_GHz} GHz not found in betapol data. "
                f"Available bands: {available_bands} GHz, required keys: {full_key}, {main_key}, {r_key}"
            )

        self.r_fine_betapol_jax = jax.device_put(beam_data[r_key].astype(self.config.dtype_jax_real))
        self.BT_r_norm_jax = jax.device_put(beam_data[full_key].astype(self.config.dtype_jax_real))
        self.Bmain_r_norm_jax = jax.device_put(beam_data[main_key].astype(self.config.dtype_jax_real))

        self.r_fine_np = beam_data[r_key]
        self.BT_r_norm_np = beam_data[full_key]
        self.Bmain_r_norm_np = beam_data[main_key]

        (
            r_fine_jax,
            self.ortho_basis_funcs_jax,
            self.n_bspline_coeffs,
            self.r_knots,
        ) = BeamModelBSplinesGaussian._setup_bspline_lut(self)
        self.r_fine_jax = r_fine_jax  # B-spline lookup grid
        self.n_fittable_coeffs = self.n_bspline_coeffs

    def get_initial_physical_params(self):
        """Initialize B-spline coefficients to zero (perturbations around the main beam)."""
        return {"bspline_coeffs_P": jnp.zeros(self.n_bspline_coeffs, dtype=self.config.dtype_jax_real)}

    def evaluate_beam_profile_P(self, params, r_values):
        """
        Calculate the polarization beam profile: Bmain(r) + Σ c_i φ_i(r).
        """
        r_values = jnp.asarray(r_values, dtype=self.config.dtype_jax_real)
        bspline_coeffs = jnp.asarray(params["bspline_coeffs_P"], dtype=self.config.dtype_jax_real)

        bspline_profile_fine = jnp.dot(self.ortho_basis_funcs_jax, bspline_coeffs)
        bspline_component = linear_interp_differentiable(r_values, self.r_fine_jax, bspline_profile_fine, self.config)
        bspline_component = bspline_component + BeamModelBSplinesGaussian._bspline_rmax_offset(self, r_values)
        base_profile = linear_interp_differentiable(r_values, self.r_fine_betapol_jax, self.Bmain_r_norm_jax, self.config)

        return base_profile + bspline_component

    def evaluate_beam_maps(self, params, dy, dx):
        r_map_arcmin = self._calculate_r_map(dy, dx)
        beam_T_map = linear_interp_differentiable(r_map_arcmin, self.r_fine_betapol_jax, self.BT_r_norm_jax, self.config)
        beam_P_map = self.evaluate_beam_profile_P(params, r_map_arcmin)
        return beam_T_map, beam_P_map

    def get_human_readable_params(self, params, prefix=""):
        """Returns a human-readable dictionary of the B-spline coefficients."""
        result = {}
        for i, c in enumerate(params["bspline_coeffs_P"]):
            result[f"{prefix}bspline_coeff_P_{i}"] = float(c)
        return result

    def get_profiles_for_plotting(self, params):
        bspline_coeffs = np.array(params["bspline_coeffs_P"])
        r_fine = np.array(self.r_fine_jax)
        ortho_funcs = np.array(self.ortho_basis_funcs_jax)
        bspline_component = ortho_funcs @ bspline_coeffs + BeamModelBSplinesGaussian._bspline_rmax_offset_np(self, r_fine)
        base_profile = np.interp(r_fine, self.r_fine_np, self.Bmain_r_norm_np)
        profile_P_fine = base_profile + bspline_component
        profile_T_fine = np.interp(r_fine, self.r_fine_np, self.BT_r_norm_np)
        info = {
            "t_label": "T-Beam Profile (BT_r, fixed)",
            "p_label": "P-Beam Profile (Bmain + B-splines above 0.75 arcmin)",
            "ylabel": "Beam Amplitude",
            "normalization": "pre-normalized",
        }
        return r_fine, profile_T_fine, profile_P_fine, info


def create_beam_model(config, y_grid, x_grid, band):
    """Factory function to create a beam model instance."""
    if config.beam_model_type == "gaussian":
        return BeamModelGaussian(config, y_grid, x_grid, band)
    elif config.beam_model_type == "beta_pol":
        return BeamModelBetaPol(config, y_grid, x_grid, band)
    elif config.beam_model_type == "beta_T":
        return BeamModelBetaTest(config, y_grid, x_grid, band)
    elif config.beam_model_type == "bsplines_plus_gaussian":
        return BeamModelBSplinesGaussian(config, y_grid, x_grid, band)
    elif config.beam_model_type == "bsplines_plus_main":
        return BeamModelBSplinesPlusMain(config, y_grid, x_grid, band)
    else:
        raise ValueError(f"Unknown beam model type: {config.beam_model_type}")
