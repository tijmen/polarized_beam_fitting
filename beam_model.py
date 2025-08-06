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

    def _calculate_r_map(self, dy, dx):
        """Calculate radial distance map from source position."""
        dx_diff = self.x_grid - dx
        dy_diff = self.y_grid - dy
        return jnp.sqrt(dx_diff * dx_diff + dy_diff * dy_diff) * self.config.reso_arcmin


class BeamModelBspline(BeamModelBase):
    """
    Beam model using orthonormal B-spline basis with boundary conditions.

    This class sets up an orthonormal B-spline basis that naturally satisfies
    boundary conditions for the beam profile.
    """

    def __init__(
        self,
        config,
        y_grid,
        x_grid,
        band,
        spline_k=4,
        spline_rmax_arcmin=10.0,
        knot_spacing_arcmin=0.5,
    ):
        """
        Initialize the beam model.

        Parameters:
        -----------
        spline_k : int
            B-spline order (4 for cubic)
        spline_rmax_arcmin : float
            Maximum radius in arcminutes
        knot_spacing_arcmin : float
            Spacing between knots in arcminutes
        """
        super().__init__(config, y_grid, x_grid, band)
        self.spline_k = spline_k
        self.spline_rmax_arcmin = spline_rmax_arcmin
        self.knot_spacing_arcmin = knot_spacing_arcmin

        # Get FWHM from config
        self.fwhm_arcmin = self.config.band_fwhm_arcmin[self.band]

        # Setup the orthonormal basis
        (
            r_fine_jax,
            self.ortho_basis_funcs_jax,
            self.n_fittable_coeffs,
            self.r_knots,
        ) = self._setup_spline_lut()
        self.r_fine_jax = r_fine_jax

        # Define parameter names for the base class methods
        self.param_names = ["T_coeffs", "P_coeffs"]

    def _setup_spline_lut(self):
        """
        Setup B-splines with an orthonormal basis that naturally satisfies boundary conditions.

        Strategy:
        1. Start with standard B-spline basis
        2. Apply boundary constraints to get null space
        3. Orthonormalize the null space basis
        4. Create efficient lookup tables for JAX

        Returns:
        --------
        tuple
            (r_fine_jax, ortho_basis_funcs_jax, n_fittable_coeffs, r_knots)
        """
        print("Setting up orthonormal B-spline basis with boundary conditions...")
        degree = self.spline_k - 1  # degree = 3 for cubic spline
        r_min = 0.0
        r_max = self.spline_rmax_arcmin

        # Interior knots
        delta_r = self.knot_spacing_arcmin
        interior_knots = np.arange(r_min + delta_r, r_max, delta_r)

        # Clamped knot vector for better boundary behavior
        self.r_knots = np.concatenate([np.full(degree + 1, r_min), interior_knots, np.full(degree + 1, r_max)])

        n_coeffs_total = len(self.r_knots) - degree - 1

        # Create standard B-spline basis
        identity_coeffs = np.eye(n_coeffs_total)
        basis_functions = []
        for i in range(n_coeffs_total):
            basis_functions.append(interpolate.BSpline(self.r_knots, identity_coeffs[i], degree))

        # Build constraint matrix for boundary conditions
        boundary_conditions = [
            (0.0, 1.0, False),  # B(0) = 1
            (0.0, 0.0, True),  # B'(0) = 0
            (r_max, 0.0, False),  # B(r_max) = 0
            (r_max, 0.0, True),  # B'(r_max) = 0
        ]

        constraints, constraint_values = [], []
        for r_pos, target_val, use_derivative in boundary_conditions:
            if use_derivative:
                constraint_row = np.array([bf.derivative()(r_pos) for bf in basis_functions])
            else:
                constraint_row = np.array([bf(r_pos) for bf in basis_functions])
            constraints.append(constraint_row)
            constraint_values.append(target_val)

        A = np.vstack(constraints)
        b = np.array(constraint_values)

        # Find particular solution that satisfies constraints exactly using constrained least squares
        sigma_arcmin = self.fwhm_arcmin / (2 * np.sqrt(2 * np.log(2)))

        # Create target Gaussian on evaluation grid
        r_eval_target = np.linspace(r_min, r_max, 500)
        target_gaussian = np.exp(-0.5 * (r_eval_target / sigma_arcmin) ** 2)

        # Evaluate basis functions on target grid
        target_basis_matrix = np.zeros((len(r_eval_target), n_coeffs_total))
        for i, basis_func in enumerate(basis_functions):
            target_basis_matrix[:, i] = basis_func(r_eval_target)

        # Weight for cylindrical coordinates
        target_weight = r_eval_target.copy()
        target_weight[0] = target_weight[1] * 0.5  # Avoid zero weight at origin
        W_target = np.diag(target_weight)

        # Use constrained least squares: min ||W * (C @ x - d)||^2 subject to A @ x = b
        # This ensures constraints are satisfied exactly
        C = W_target @ target_basis_matrix
        d = W_target @ target_gaussian

        # Solve using the method of Lagrange multipliers
        # The solution is: x = A_pinv @ b + (I - A_pinv @ A) @ C_pinv @ d
        # where A_pinv is the pseudoinverse of A

        # Compute pseudoinverse of constraint matrix
        A_pinv = np.linalg.pinv(A)

        # Project the Gaussian fitting onto the null space of constraints
        P_null = np.eye(n_coeffs_total) - A_pinv @ A  # Projection onto null space

        # Solve the projected least squares problem
        C_projected = C @ P_null
        C_proj_pinv = np.linalg.pinv(C_projected)

        # Particular solution: exact constraint satisfaction + best Gaussian fit in null space
        x_constraint = A_pinv @ b  # Exact solution to constraints
        x_gaussian = P_null @ C_proj_pinv @ (d - C @ x_constraint)  # Gaussian fit in null space
        particular_solution = x_constraint + x_gaussian

        print("Particular solution fitting:")
        print("  Constraint satisfaction: exact (using pseudoinverse)")

        # Evaluate Gaussian approximation quality
        approx_gaussian = target_basis_matrix @ particular_solution
        gaussian_rms = np.sqrt(np.mean((approx_gaussian - target_gaussian) ** 2))
        print(f"  Gaussian RMS error: {gaussian_rms:.6f}")

        print(f"Constraint matrix condition number: {np.linalg.cond(A):.2e}")

        # Verify constraints are satisfied
        constraint_check = A @ particular_solution - b
        print(f"Constraint violations: {np.abs(constraint_check)}")
        max_violation = np.max(np.abs(constraint_check))
        if max_violation > 1e-10:
            print(f"WARNING: Large constraint violation: {max_violation:.2e}")
        else:
            print("Constraints satisfied to machine precision")

        # Find null space of A (these are the free directions)
        U, s, Vt = np.linalg.svd(A)
        tol = max(1e-10, s[0] * 1e-12)  # Relative tolerance
        rank = np.sum(s > tol)
        null_space_basis = Vt[rank:].T  # Columns are null space vectors

        print(f"Null space dimension: {null_space_basis.shape[1]} free parameters")

        # Define evaluation grid for inner products
        n_eval = 2000
        r_eval = np.linspace(r_min, r_max, n_eval)

        # Evaluate all basis functions on the grid
        basis_matrix = np.zeros((n_eval, n_coeffs_total))
        for i, basis_func in enumerate(basis_functions):
            basis_matrix[:, i] = basis_func(r_eval)

        # Weight function for cylindrical coordinates
        weight = r_eval.copy()
        weight[0] = weight[1] * 0.5  # Avoid zero weight at origin

        # Compute Gram matrix for the null space vectors
        n_null = null_space_basis.shape[1]
        gram_matrix = np.zeros((n_null, n_null))

        for i in range(n_null):
            func_i = basis_matrix @ null_space_basis[:, i]
            for j in range(i, n_null):
                func_j = basis_matrix @ null_space_basis[:, j]
                inner_prod = np.trapz(func_i * func_j * weight, r_eval)
                gram_matrix[i, j] = inner_prod
                gram_matrix[j, i] = inner_prod

        # Orthogonalize using eigendecomposition (more stable than Cholesky)
        eigvals, eigvecs = np.linalg.eigh(gram_matrix)

        # Keep only positive eigenvalues
        mask = eigvals > 1e-10
        if not np.any(mask):
            raise ValueError("Gram matrix is singular - check null space computation")

        orthogonal_coeffs = null_space_basis @ (eigvecs[:, mask] @ np.diag(1.0 / np.sqrt(eigvals[mask])))

        # Verify orthonormality
        test_gram = np.zeros((orthogonal_coeffs.shape[1], orthogonal_coeffs.shape[1]))
        for i in range(orthogonal_coeffs.shape[1]):
            func_i = basis_matrix @ orthogonal_coeffs[:, i]
            for j in range(orthogonal_coeffs.shape[1]):
                func_j = basis_matrix @ orthogonal_coeffs[:, j]
                test_gram[i, j] = np.trapz(func_i * func_j * weight, r_eval)

        off_diag_error = np.max(np.abs(test_gram - np.eye(test_gram.shape[0])))
        print(f"Orthonormality verification: max off-diagonal = {off_diag_error:.2e}")

        # Store everything we need for reconstruction
        self.n_fittable_coeffs = orthogonal_coeffs.shape[1]
        self.particular_solution = particular_solution
        self.orthogonal_coeffs = orthogonal_coeffs

        # Create fine-resolution lookup tables for JAX
        r_fine = np.linspace(r_min, r_max, 1000)

        # Particular solution function
        particular_func = np.zeros_like(r_fine)
        for i, coeff in enumerate(particular_solution):
            particular_func += coeff * basis_functions[i](r_fine)

        # Orthonormal basis functions
        ortho_basis_funcs = np.zeros((len(r_fine), self.n_fittable_coeffs))
        for j in range(self.n_fittable_coeffs):
            for i, coeff in enumerate(orthogonal_coeffs[:, j]):
                ortho_basis_funcs[:, j] += coeff * basis_functions[i](r_fine)

        # Verify boundary conditions on particular solution
        print("\nParticular solution boundary conditions:")
        print(f"  B(0) = {particular_func[0]:.6f} (target: 1.0)")
        h = r_fine[1] - r_fine[0]
        B_prime_0_estimate = (particular_func[1] - particular_func[0]) / h
        print(f"  B'(0) ≈ {B_prime_0_estimate:.6f} (target: 0.0)")
        print(f"  B(r_max) = {particular_func[-1]:.6f} (target: 0.0)")
        B_prime_rmax_estimate = (particular_func[-1] - particular_func[-2]) / h
        print(f"  B'(r_max) ≈ {B_prime_rmax_estimate:.6f} (target: 0.0)")

        # Convert to JAX arrays
        r_fine_jax = jax.device_put(r_fine.astype(self.config.dtype_jax_real))
        particular_func_jax = jax.device_put(particular_func.astype(self.config.dtype_jax_real))
        ortho_basis_funcs_jax = jax.device_put(ortho_basis_funcs.astype(self.config.dtype_jax_real))

        # Store for later use
        self.particular_func_jax = particular_func_jax

        print("\nOrthonormal B-spline basis setup complete:")
        print(f"  {len(self.r_knots)} knots, {n_coeffs_total} total B-spline coefficients")
        print(f"  {self.n_fittable_coeffs} orthonormal basis functions")

        return r_fine_jax, ortho_basis_funcs_jax, self.n_fittable_coeffs, self.r_knots

    def get_initial_physical_params(self):
        """Returns initial parameters."""
        fwhm_arcmin = self.config.band_fwhm_arcmin[self.band]
        initial_coeffs = self.fit_gaussian_coefficients(fwhm_arcmin)
        return {"T_coeffs": initial_coeffs.copy(), "P_coeffs": initial_coeffs.copy()}

    def evaluate_beam_profile(self, coeffs, r_values):
        """
        Evaluate the beam profile at given radial values.

        Parameters:
        -----------
        coeffs : array_like
            Coefficients for the orthonormal basis functions
        r_values : array_like
            Radial values to evaluate at

        Returns:
        --------
        array_like
            Beam profile values
        """
        # Construct 1D beam profile
        # B(r) = particular + sum of orthonormal basis functions
        profile_fine = self.particular_func_jax + jnp.dot(self.ortho_basis_funcs_jax, coeffs)

        # Ensure r_values is a JAX array
        r_values = jnp.asarray(r_values, dtype=self.config.dtype_jax_real)

        # Interpolate to desired radial values
        beam_profile = linear_interp_differentiable(r_values, self.r_fine_jax, profile_fine, self.config)

        return beam_profile

    def fit_gaussian_coefficients(self, fwhm_arcmin):
        """
        Find coefficients that best approximate a Gaussian beam with given FWHM.

        Parameters:
        -----------
        fwhm_arcmin : float
            Full-width at half-maximum in arcminutes

        Returns:
        --------
        numpy.ndarray
            Coefficients for the orthonormal basis functions
        """
        # Convert FWHM to sigma
        sigma_arcmin = fwhm_arcmin / (2 * np.sqrt(2 * np.log(2)))

        # Create target Gaussian profile on the fine grid
        target_gaussian = np.exp(-0.5 * (self.r_fine_jax / sigma_arcmin) ** 2)

        # We want to find coeffs such that:
        # particular_func + ortho_basis_funcs @ coeffs ≈ target_gaussian
        #
        # This gives us: ortho_basis_funcs @ coeffs ≈ target_gaussian - particular_func
        #
        # Solve the least squares problem
        residual = target_gaussian - self.particular_func_jax

        # Use numpy arrays for the least squares solve
        ortho_basis_np = np.array(self.ortho_basis_funcs_jax)
        residual_np = np.array(residual)

        # Solve weighted least squares using 2D polar coordinates
        r_fine_np = np.array(self.r_fine_jax)
        dr = r_fine_np[1] - r_fine_np[0]
        weight = r_fine_np.copy()
        weight[0] = weight[1] * 0.5
        w_sqrt = np.sqrt(weight * dr)
        A = ortho_basis_np * w_sqrt[:, np.newaxis]
        b = residual_np * w_sqrt
        coeffs, residual_norm, rank, _ = np.linalg.lstsq(A, b, rcond=None)

        print(f"Gaussian fitting for FWHM={fwhm_arcmin:.2f}' (σ={sigma_arcmin:.2f}):")
        print(f"  Residual norm: {residual_norm[0] if len(residual_norm) > 0 else 'exact'}")
        print(f"  Matrix rank: {rank}/{self.n_fittable_coeffs}")

        # Verify the fit quality
        fitted_profile = self.particular_func_jax + np.dot(ortho_basis_np, coeffs)
        max_error = np.max(np.abs(fitted_profile - target_gaussian))
        rms_error = np.sqrt(np.mean((fitted_profile - target_gaussian) ** 2))
        print(f"  Max error: {max_error:.6f}, RMS error: {rms_error:.6f}")

        return coeffs

    def evaluate_beam_maps(self, params, dy, dx):
        r_map_arcmin = self._calculate_r_map(dy, dx)
        beam_T_map = self.evaluate_beam_profile(params["T_coeffs"], r_map_arcmin)
        beam_P_map = self.evaluate_beam_profile(params["P_coeffs"], r_map_arcmin)
        return beam_T_map, beam_P_map

    def get_human_readable_params(self, params):
        return {
            "beam_T_coeffs": np.array(params["T_coeffs"]),
            "beam_P_coeffs": np.array(params["P_coeffs"]),
        }

    def get_profiles_for_plotting(self, params):
        fitted_coeffs_T = params["T_coeffs"]
        fitted_coeffs_P = params["P_coeffs"]
        r_fine = np.array(self.r_fine_jax)
        particular = np.array(self.particular_func_jax)
        ortho_funcs = np.array(self.ortho_basis_funcs_jax)
        profile_T_fine = particular + ortho_funcs @ fitted_coeffs_T
        profile_P_fine = particular + ortho_funcs @ fitted_coeffs_P
        # Area normalization (as in plotting.py)
        band_fwhm_arcmin = self.config.band_fwhm_arcmin[self.band]
        sigma_arcmin = band_fwhm_arcmin / (2 * np.sqrt(2 * np.log(2)))
        r_max_integration = 2.0 * sigma_arcmin
        mask = r_fine <= r_max_integration
        r_integration = r_fine[mask]
        profile_T_integration = profile_T_fine[mask]
        profile_P_integration = profile_P_fine[mask]
        area_T = 2 * np.pi * np.trapz(profile_T_integration * r_integration, r_integration)
        area_P = 2 * np.pi * np.trapz(profile_P_integration * r_integration, r_integration)
        profile_T_fine_normalized = profile_T_fine / area_T
        profile_P_fine_normalized = profile_P_fine / area_P
        info = {
            "t_label": "Best-Fit T-Beam Profile (Area Normalized)",
            "p_label": "Best-Fit P-Beam Profile (Area Normalized)",
            "ylabel": "Area-Normalized Amplitude [arcmin⁻²]",
            "normalization": "area",
            "area_T": area_T,
            "area_P": area_P,
        }
        return r_fine, profile_T_fine_normalized, profile_P_fine_normalized, info


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
    Temperature beam(r) = BT_r(t)
    Polarization beam(r) = Bmain_r(r) + beta_pol * (BT_r(r) - Bmain_r(r))
    """

    def __init__(self, config, y_grid, x_grid, band):
        super().__init__(config, y_grid, x_grid, band)
        self.param_names = ["beta_pol"]
        self.n_fittable_coeffs = 1

        # Load the pre-calculated beam profiles
        import os

        data_path = os.path.join(os.path.dirname(__file__), "data", "betapol_TdH.npz")
        try:
            beam_data = np.load(data_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find the betapol data file at {data_path}. Run create_betapol_data.py to generate it.")

        # Check if the requested band is available
        bt_key = f"BT_r_norm_{self.band_GHz}"
        bmain_key = f"Bmain_r_norm_{self.band_GHz}"

        if bt_key not in beam_data:
            available_bands = [key.split("_")[-1] for key in beam_data.keys() if key.startswith("BT_r_norm_")]
            raise KeyError(f"Band {self.band_GHz} GHz not found in betapol data. Available bands: {available_bands} GHz")

        print(f"Loading betapol beam model for {self.band_GHz} GHz")

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
        import os

        data_path = os.path.join(os.path.dirname(__file__), "data", "betapol_TdH.npz")
        try:
            beam_data = np.load(data_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find the betapol data file at {data_path}. Run create_betapol_data.py to generate it.")

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
    - No boundary conditions on B-splines
    """

    def __init__(self, config, y_grid, x_grid, band):
        super().__init__(config, y_grid, x_grid, band)
        self.param_names = [
            "gaussian_sigma_arcmin",
            "bspline_coeffs_T",
            "bspline_coeffs_P",
        ]
        self.spline_k = config.spline_k
        self.spline_rmax_arcmin = config.spline_rmax_arcmin
        self.knot_spacing_arcmin = config.knot_spacing_arcmin
        self.spline_rmin_arcmin = config.bsplines_gaussian_rmin_arcmin

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

    def _count_bspline_coeffs(self):
        """Count how many B-spline coefficients we'll have."""
        degree = self.spline_k - 1
        r_min = self.spline_rmin_arcmin
        r_max = self.spline_rmax_arcmin

        # Interior knots
        delta_r = self.knot_spacing_arcmin
        interior_knots = np.arange(r_min + delta_r, r_max, delta_r)

        # Clamped knot vector
        r_knots = np.concatenate([np.full(degree + 1, r_min), interior_knots, np.full(degree + 1, r_max)])

        n_coeffs_total = len(r_knots) - degree - 1
        return n_coeffs_total

    def _setup_bspline_lut(self):
        """
        Setup area-normalized orthogonal B-splines with boundary conditions B(r_min) = 0 and B'(r_min) = 0.

        The first knot is at r_min, but the B-spline basis functions are constrained
        to evaluate to zero and have zero derivative at r_min, ensuring C1 continuity
        with the Gaussian component.
        """
        print("Setting up area-normalized orthogonal B-splines with B(r_min) = B'(r_min) = 0 constraints...")
        degree = self.spline_k - 1  # degree = 3 for cubic spline
        r_min = self.spline_rmin_arcmin
        r_max = self.spline_rmax_arcmin

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

        # Build constraint matrix for boundary conditions B(r_min) = 0 and B'(r_min) = 0
        constraint_rows = []
        constraint_values = []

        # B(r_min) = 0
        constraint_rows.append(np.array([bf(r_min) for bf in basis_functions]))
        constraint_values.append(0.0)

        # B'(r_min) = 0 (first derivative)
        constraint_rows.append(np.array([bf.derivative()(r_min) for bf in basis_functions]))
        constraint_values.append(0.0)

        A = np.vstack(constraint_rows)
        _ = np.array(constraint_values)  # we don't actually need this if the boundary conditions are all zero

        print(f"Constraint matrix condition number: {np.linalg.cond(A):.2e}")

        # Find null space of A (these are the free directions)
        U, s, Vt = np.linalg.svd(A)
        tol = max(1e-10, s[0] * 1e-12)  # Relative tolerance
        rank = np.sum(s > tol)
        null_space_basis = Vt[rank:].T  # Columns are null space vectors

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

        # Compute Gram matrix for the null space vectors
        n_null = null_space_basis.shape[1]
        gram_matrix = np.zeros((n_null, n_null))

        for i in range(n_null):
            func_i = basis_matrix @ null_space_basis[:, i]
            for j in range(i, n_null):
                func_j = basis_matrix @ null_space_basis[:, j]
                # 2D area inner product: ∫f_i(r) * f_j(r) * 2π*r dr
                inner_prod = np.trapz(func_i * func_j * weight_2d, r_eval)
                gram_matrix[i, j] = inner_prod
                gram_matrix[j, i] = inner_prod

        # Orthogonalize using eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(gram_matrix)

        # Keep only positive eigenvalues
        mask = eigvals > 1e-10
        if not np.any(mask):
            raise ValueError("Gram matrix is singular - check null space computation")

        # Create orthonormal basis with 2D area normalization
        orthogonal_coeffs = null_space_basis @ (eigvecs[:, mask] @ np.diag(1.0 / np.sqrt(eigvals[mask])))

        # Verify orthonormality
        test_gram = np.zeros((orthogonal_coeffs.shape[1], orthogonal_coeffs.shape[1]))
        for i in range(orthogonal_coeffs.shape[1]):
            func_i = basis_matrix @ orthogonal_coeffs[:, i]
            for j in range(orthogonal_coeffs.shape[1]):
                func_j = basis_matrix @ orthogonal_coeffs[:, j]
                test_gram[i, j] = np.trapz(func_i * func_j * weight_2d, r_eval)

        off_diag_error = np.max(np.abs(test_gram - np.eye(test_gram.shape[0])))
        print(f"2D area orthonormality verification: max off-diagonal = {off_diag_error:.2e}")

        # Verify boundary conditions are satisfied
        boundary_check_value = basis_matrix[0, :] @ orthogonal_coeffs
        max_value_violation = np.max(np.abs(boundary_check_value))

        # Check derivative at boundary
        derivative_at_boundary = np.array([bf.derivative()(r_min) for bf in basis_functions])
        boundary_check_derivative = derivative_at_boundary @ orthogonal_coeffs
        max_derivative_violation = np.max(np.abs(boundary_check_derivative))

        print(f"Boundary condition B(r_min) = 0: max violation = {max_value_violation:.2e}")
        print(f"Boundary condition B'(r_min) = 0: max violation = {max_derivative_violation:.2e}")

        # Store coefficients for reconstruction
        self.n_bspline_coeffs = orthogonal_coeffs.shape[1]
        self.orthogonal_coeffs = orthogonal_coeffs

        # Create fine-resolution lookup tables for JAX
        r_fine = np.linspace(0, r_max, 1000)  # Start from 0 for the full beam model

        # Orthonormal B-spline basis functions (only defined for r >= r_min)
        ortho_basis_funcs = np.zeros((len(r_fine), self.n_bspline_coeffs))

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

        # Convert to JAX arrays
        r_fine_jax = jax.device_put(r_fine.astype(self.config.dtype_jax_real))
        ortho_basis_funcs_jax = jax.device_put(ortho_basis_funcs.astype(self.config.dtype_jax_real))

        print("Area-normalized B-spline basis setup complete:")
        print(f"  {len(self.r_knots)} knots, {n_coeffs_total} total B-spline coefficients")
        print(f"  {self.n_bspline_coeffs} orthonormal basis functions")
        print(f"  B-splines start at r = {r_min:.1f} arcmin with B(r_min) = B'(r_min) = 0 constraints")

        return r_fine_jax, ortho_basis_funcs_jax, self.n_bspline_coeffs, self.r_knots

    def get_initial_physical_params(self):
        """Returns initial guess for all parameters."""
        # Use band FWHM for initial Gaussian sigma (shared between T and P)
        band_fwhm = self.config.band_fwhm_arcmin[self.band]
        initial_sigma = band_fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma

        # Initialize B-spline coefficients to small random values (separate for T and P)
        key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
        key1, key2 = jax.random.split(key)
        bspline_coeffs_T = jax.random.normal(key1, (self.n_bspline_coeffs,), dtype=self.config.dtype_jax_real) * 0.01
        bspline_coeffs_P = jax.random.normal(key2, (self.n_bspline_coeffs,), dtype=self.config.dtype_jax_real) * 0.01

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
        bspline_component_T = ortho_funcs @ bspline_coeffs_T
        bspline_component_P = ortho_funcs @ bspline_coeffs_P
        profile_T_fine = gaussian_component + bspline_component_T
        profile_P_fine = gaussian_component + bspline_component_P
        info = {
            "t_label": f"T-Beam Profile (Gaussian σ={gaussian_sigma:.3f} arcmin + B-splines)",
            "p_label": f"P-Beam Profile (Gaussian σ={gaussian_sigma:.3f} arcmin + B-splines)",
            "ylabel": "Beam Amplitude",
            "normalization": None,
        }
        return r_fine, profile_T_fine, profile_P_fine, info


def create_beam_model(config, y_grid, x_grid, band):
    """Factory function to create a beam model instance."""
    if config.beam_model_type == "b_spline":
        return BeamModelBspline(
            config,
            y_grid,
            x_grid,
            band,
            config.spline_k,
            config.spline_rmax_arcmin,
            config.knot_spacing_arcmin,
        )
    elif config.beam_model_type == "gaussian":
        return BeamModelGaussian(config, y_grid, x_grid, band)
    elif config.beam_model_type == "betapol":
        return BeamModelBetaPol(config, y_grid, x_grid, band)
    elif config.beam_model_type == "betatest":
        return BeamModelBetaTest(config, y_grid, x_grid, band)
    elif config.beam_model_type == "bsplines_plus_gaussian":
        return BeamModelBSplinesGaussian(config, y_grid, x_grid, band)
    else:
        raise ValueError(f"Unknown beam model type: {config.beam_model_type}")
