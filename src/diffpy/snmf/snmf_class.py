import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, csc_matrix, diags


class SNMFOptimizer:
    """A implementation of stretched NMF (sNMF), including sparse stretched NMF.

    Instantiating the SNMFOptimizer class runs all the analysis immediately.
    The results matrices can then be accessed as instance attributes
    of the class (comps, weights, and stretch).

    For more information on sNMF, please reference:
    Gu, R., Rakita, Y., Lan, L. et al. Stretched non-negative matrix factorization.
    npj Comput Mater 10, 193 (2024). https://doi.org/10.1038/s41524-024-01377-5

    Attributes
    ----------
    source_matrix : ndarray
        The original, unmodified data to be decomposed and later, compared against.
        Shape is (length_of_signal, number_of_conditions).
    stretch : ndarray
        The best guess (or while running, the current guess) for the stretching
        factor matrix.
    comps : ndarray
        The best guess (or while running, the current guess) for the matrix of
        component intensities.
    weights : ndarray
        The best guess (or while running, the current guess) for the matrix of
        component weights.
    rho : float
            The stretching factor that influences the decomposition. Zero corresponds to no
            stretching present. Relatively insensitive and typically adjusted in powers of 10.
    eta : float
        The sparsity factor that influences the decomposition. Should be set to zero for
        non-sparse data such as PDF. Can be used to improve results for sparse data such
        as XRD, but due to instability, should be used only after first selecting the
        best value for rho. Suggested adjustment is by powers of 2.
    max_iter : int
            The maximum number of times to update each of stretch, comps, and weights before stopping
            the optimization.
    tol : float
        The convergence threshold. This is the minimum fractional improvement in the
        objective function to allow without terminating the optimization. Note that
        a minimum of 20 updates are run before this parameter is checked.
    n_components : int
        The number of components to extract from source_matrix. Must be provided when and only when
        Y0 is not provided.
    random_state : int
        The seed for the initial guesses at the matrices (A, X, and Y) created by
        the decomposition.
    num_updates : int
        The total number of times that any of (A, X, and Y) have had their values changed.
        If not terminated by other means, this value is used to stop when reaching max_iter.
    objective_function: float
        The value corresponding to the minimization of the difference between the source_matrix and the
        products of A, X, and Y. For full details see the sNMF paper. Smaller corresponds to
        better agreement and is desirable.
    objective_difference : float
        The change in the objective function value since the last update. A negative value
        means that the result improved.
    """

    def __init__(
        self,
        source_matrix,
        init_weights=None,
        init_comps=None,
        init_stretch=None,
        rho=1e12,
        eta=610,
        max_iter=500,
        tol=5e-7,
        n_components=None,
        random_state=None,
    ):
        """Initialize an instance of SNMF and run the optimization.

        Parameters
        ----------
        source_matrix : ndarray
            The data to be decomposed. Shape is (length_of_signal, number_of_conditions).
        init_weights : ndarray
            The initial guesses for the component weights at each stretching condition.
            Shape is (number_of_components, number_of_conditions) Must provide exactly one
            of this or n_components.
        init_comps : ndarray
            The initial guesses for the intensities of each component per
            row/sample/angle. Shape is (length_of_signal, number_of_components).
        init_stretch : ndarray
            The initial guesses for the stretching factor for each component, at each
            condition. Shape is (number_of_components, number_of_conditions).
        rho : float
            The stretching factor that influences the decomposition. Zero corresponds to no
            stretching present. Relatively insensitive and typically adjusted in powers of 10.
        eta : float
            The sparsity factor that influences the decomposition. Should be set to zero for
            non-sparse data such as PDF. Can be used to improve results for sparse data such
            as XRD, but due to instability, should be used only after first selecting the
            best value for rho. Suggested adjustment is by powers of 2.
        max_iter : int
            The maximum number of times to update each of A, X, and Y before stopping
            the optimization.
        tol : float
            The convergence threshold. This is the minimum fractional improvement in the
            objective function to allow without terminating the optimization. Note that
            a minimum of 20 updates are run before this parameter is checked.
        n_components : int
            The number of components to extract from source_matrix. Must be provided when and only when
            Y0 is not provided.
        random_state : int
            The seed for the initial guesses at the matrices (A, X, and Y) created by
            the decomposition.
        """

        self.source_matrix = source_matrix
        self.rho = rho
        self.eta = eta
        # Capture matrix dimensions
        self._signal_len, self._num_conditions = source_matrix.shape
        self.num_updates = 0
        self._rng = np.random.default_rng(random_state)

        # Enforce exclusive specification of n_components or Y0
        if (n_components is None and init_weights is None) or (
            n_components is not None and init_weights is not None
        ):
            raise ValueError("Must provide exactly one of init_weights or n_components, but not both.")

        # Initialize weights and determine number of components
        if init_weights is None:
            self._n_components = n_components
            self.weights = self._rng.beta(a=2.5, b=1.5, size=(self._n_components, self._num_conditions))
        else:
            self._n_components = init_weights.shape[0]
            self.weights = init_weights

        # Initialize stretching matrix if not provided
        if init_stretch is None:
            self.stretch = np.ones((self._n_components, self._num_conditions)) + self._rng.normal(
                0, 1e-3, size=(self._n_components, self._num_conditions)
            )
        else:
            self.stretch = init_stretch

        # Initialize component matrix if not provided
        if init_comps is None:
            self.comps = self._rng.random((self._signal_len, self._n_components))
        else:
            self.comps = init_comps

        # Enforce non-negativity in our initial guesses
        self.comps = np.maximum(0, self.comps)
        self.weights = np.maximum(0, self.weights)

        # Second-order spline: Tridiagonal (-2 on diagonal, 1 on sub/superdiagonals)
        self.spline_smooth_operator = 0.25 * diags(
            [1, -2, 1], offsets=[0, 1, 2], shape=(self._num_conditions - 2, self._num_conditions)
        )
        self.spline_smooth_penalty = self.spline_smooth_operator.T @ self.spline_smooth_operator

        # Set up residual matrix, objective function, and history
        self.residuals = self.get_residual_matrix()
        self.objective_function = self.get_objective_function()
        self.objective_difference = None
        self._objective_history = [self.objective_function]

        # Set up tracking variables for updateX()
        self._prev_comps = None
        self.grad_comps = np.zeros_like(self.comps)  # Gradient of X (zeros for now)
        self._prev_grad_comps = np.zeros_like(self.comps)  # Previous gradient of X (zeros for now)

        regularization_term = 0.5 * rho * np.linalg.norm(self.spline_smooth_operator @ self.stretch.T, "fro") ** 2
        sparsity_term = eta * np.sum(np.sqrt(self.comps))  # Square root penalty
        print(
            f"Start, Objective function: {self.objective_function:.5e}"
            f", Obj - reg/sparse: {self.objective_function - regularization_term - sparsity_term:.5e}"
        )

        # Main optimization loop
        for iter in range(max_iter):
            self.optimize_loop()
            # Print diagnostics
            regularization_term = (
                0.5 * rho * np.linalg.norm(self.spline_smooth_operator @ self.stretch.T, "fro") ** 2
            )
            sparsity_term = eta * np.sum(np.sqrt(self.comps))  # Square root penalty
            print(
                f"Num_updates: {self.num_updates}, "
                f"Obj fun: {self.objective_function:.5e}, "
                f"Obj - reg/sparse: {self.objective_function - regularization_term - sparsity_term:.5e}, "
                f"Iter: {iter}"
            )

            # Convergence check: decide when to terminate for small/no improvement
            print(self.objective_difference, " < ", self.objective_function * tol)
            if self.objective_difference < self.objective_function * tol and iter >= 20:
                break

        # Normalize our results
        weights_row_max = np.max(self.weights, axis=1, keepdims=True)
        stretch_row_max = np.max(self.stretch, axis=1, keepdims=True)
        self.weights = self.weights / weights_row_max
        self.stretch = self.stretch / stretch_row_max

        # loop to normalize components
        # effectively just re-running class with non-normalized comps, normalized wts/stretch as inputs,
        # then only update comps
        self._prev_comps = None
        self.grad_comps = np.zeros_like(self.comps)
        self._prev_grad_comps = np.zeros_like(self.comps)
        self.residuals = self.get_residual_matrix()
        self.objective_function = self.get_objective_function()
        self.objective_difference = None
        self._objective_history = [self.objective_function]
        for norm_iter in range(100):
            self.update_comps()
            self.residuals = self.get_residual_matrix()
            self.objective_function = self.get_objective_function()
            print(f"Objective function after normX: {self.objective_function:.5e}")
            self._objective_history.append(self.objective_function)
            self.objective_difference = self._objective_history[-2] - self._objective_history[-1]
            if self.objective_difference < self.objective_function * tol and norm_iter >= 20:
                break
        # end of normalization (and program)
        # note that objective function may not fully recover after normalization, this is okay
        print("Finished optimization.")

    def optimize_loop(self):
        # Update components first
        self._prev_grad_comps = self.grad_comps.copy()
        self.update_comps()
        self.num_updates += 1
        self.residuals = self.get_residual_matrix()
        self.objective_function = self.get_objective_function()
        print(f"Objective function after updateX: {self.objective_function:.5e}")
        self._objective_history.append(self.objective_function)
        if self.objective_difference is None:
            self.objective_difference = self._objective_history[-1] - self.objective_function

        # Now we update weights
        self.update_weights()
        self.num_updates += 1
        self.residuals = self.get_residual_matrix()
        self.objective_function = self.get_objective_function()
        print(f"Objective function after updateY2: {self.objective_function:.5e}")
        self._objective_history.append(self.objective_function)

        # Now we update stretch
        self.update_stretch()
        self.num_updates += 1
        self.residuals = self.get_residual_matrix()
        self.objective_function = self.get_objective_function()
        print(f"Objective function after update_stretch: {self.objective_function:.5e}")
        self._objective_history.append(self.objective_function)
        self.objective_difference = self._objective_history[-2] - self._objective_history[-1]

    def apply_interpolation(self, a, x, return_derivatives=False):
        """
        Applies an interpolation-based transformation to `x` based on scaling `a`.
        Also can compute first (`d_intr_x`) and second (`dd_intr_x`) derivatives.
        """
        x_len = len(x)

        # Ensure `a` is an array and reshape for broadcasting
        a = np.atleast_1d(np.asarray(a))  # Ensures a is at least 1D

        # Compute fractional indices, broadcasting over `a`
        ii = np.arange(x_len)[:, None] / a  # Shape (N, M)

        II = np.floor(ii).astype(int)  # Integer part (still (N, M))
        valid_mask = II < (x_len - 1)  # Ensure indices are within bounds

        # Apply valid_mask to keep correct indices
        idx_int = np.where(valid_mask, II, x_len - 2)  # Prevent out-of-bounds indexing (previously "I")
        idx_frac = np.where(valid_mask, ii, II)  # Keep aligned (previously "i")

        # Ensure x is a 1D array
        x = np.asarray(x).ravel()

        # Compute interpolated_x (linear interpolation)
        interpolated_x = x[idx_int] * (1 - idx_frac + idx_int) + x[np.minimum(idx_int + 1, x_len - 1)] * (
            idx_frac - idx_int
        )

        # Fill the tail with the last valid value
        intr_x_tail = np.full((x_len - len(idx_int), interpolated_x.shape[1]), interpolated_x[-1, :])
        interpolated_x = np.vstack([interpolated_x, intr_x_tail])

        if return_derivatives:
            # Compute first derivative (d_intr_x)
            di = -idx_frac / a
            d_intr_x = x[idx_int] * (-di) + x[np.minimum(idx_int + 1, x_len - 1)] * di
            d_intr_x = np.vstack([d_intr_x, np.zeros((x_len - len(idx_int), d_intr_x.shape[1]))])

            # Compute second derivative (dd_intr_x)
            ddi = -di / a + idx_frac * a**-2
            dd_intr_x = x[idx_int] * (-ddi) + x[np.minimum(idx_int + 1, x_len - 1)] * ddi
            dd_intr_x = np.vstack([dd_intr_x, np.zeros((x_len - len(idx_int), dd_intr_x.shape[1]))])
        else:
            # Make placeholders
            d_intr_x = np.empty(interpolated_x.shape)
            dd_intr_x = np.empty(interpolated_x.shape)

        return interpolated_x, d_intr_x, dd_intr_x

    def get_residual_matrix(self, comps=None, weights=None, stretch=None):
        # Initialize residual matrix as negative of source_matrix
        # In MATLAB this is getR
        if comps is None:
            comps = self.comps
        if weights is None:
            weights = self.weights
        if stretch is None:
            stretch = self.stretch
        residuals = -self.source_matrix.copy()
        # Compute transformed components for all (k, m) pairs
        for k in range(weights.shape[0]):
            stretched_comps, _, _ = self.apply_interpolation(stretch[k, :], comps[:, k])  # Only calculate Ax
            residuals += weights[k, :] * stretched_comps  # Element-wise scaling and sum
        return residuals

    def get_objective_function(self, residuals=None, stretch=None):
        if residuals is None:
            residuals = self.residuals
        if stretch is None:
            stretch = self.stretch
        residual_term = 0.5 * np.linalg.norm(residuals, "fro") ** 2
        regularization_term = 0.5 * self.rho * np.linalg.norm(self.spline_smooth_operator @ stretch.T, "fro") ** 2
        sparsity_term = self.eta * np.sum(np.sqrt(self.comps))  # Square root penalty
        # Final objective function value
        function = residual_term + regularization_term + sparsity_term
        return function

    def apply_interpolation_matrix(self, comps=None, weights=None, stretch=None, return_derivatives=False):
        """
        Applies an interpolation-based transformation to the matrix `comps` using `stretch`,
        weighted by `weights`. Optionally computes first (`d_str_cmps`) and second (`dd_str_comps`) derivatives.
        Equivalent to getAfun_matrix in MATLAB.
        """

        if comps is None:
            comps = self.comps
        if weights is None:
            weights = self.weights
        if stretch is None:
            stretch = self.stretch

        # Compute scaled indices (MATLAB: AA = repmat(reshape(A',1,M*K).^-1, N,1))
        stretch_flat = stretch.reshape(1, self._num_conditions * self._n_components) ** -1
        stretch_tiled = np.tile(stretch_flat, (self._signal_len, 1))

        # Compute `ii` (MATLAB: ii = repmat((0:N-1)',1,K*M).*tiled_stretch)
        ii = (
            np.tile(np.arange(self._signal_len)[:, None], (1, self._num_conditions * self._n_components))
            * stretch_tiled
        )

        # Weighting matrix (MATLAB: YY = repmat(reshape(Y',1,M*K), N,1))
        weights_flat = weights.reshape(1, self._num_conditions * self._n_components)
        weights_tiled = np.tile(weights_flat, (self._signal_len, 1))

        # Bias for indexing into reshaped X (MATLAB: bias = kron((0:K-1)*(N+1),ones(N,M)))
        # TODO break this up or describe what it does better
        bias = np.kron(
            np.arange(self._n_components) * (self._signal_len + 1),
            np.ones((self._signal_len, self._num_conditions), dtype=int),
        ).reshape(self._signal_len, self._n_components * self._num_conditions)

        # Handle boundary conditions for interpolation (MATLAB: X1=[X;X(end,:)])
        X1 = np.vstack([comps, comps[-1, :]])  # Duplicate last row (like MATLAB)

        # Compute floor indices (MATLAB: II = floor(ii); II1=min(II+1,N+1); II2=min(II1+1,N+1))
        II = np.floor(ii).astype(int)

        II1 = np.minimum(II + 1, self._signal_len)
        II2 = np.minimum(II1 + 1, self._signal_len)

        # Compute fractional part (MATLAB: iI = ii - II)
        iI = ii - II

        # Compute offset indices (MATLAB: II1_ = II1 + bias; II2_ = II2 + bias)
        II1_ = II1 + bias
        II2_ = II2 + bias

        # Extract values (MATLAB: XI1 = reshape(X1(II1_), N, K*M); XI2 = reshape(X1(II2_), N, K*M))
        # Note: this "-1" corrects an off-by-one error that may have originated in an earlier line
        XI1 = X1.flatten(order="F")[(II1_ - 1).ravel()].reshape(
            self._signal_len, self._n_components * self._num_conditions
        )  # order = F uses FORTRAN, column major order
        XI2 = X1.flatten(order="F")[(II2_ - 1).ravel()].reshape(
            self._signal_len, self._n_components * self._num_conditions
        )

        # Interpolation (MATLAB: Ax2=XI1.*(1-iI)+XI2.*(iI); stretched_comps=Ax2.*YY)
        Ax2 = XI1 * (1 - iI) + XI2 * iI
        stretched_comps = Ax2 * weights_tiled  # Apply weighting

        if return_derivatives:
            # Compute first derivative (MATLAB: Tx2=XI1.*(-di)+XI2.*di; d_str_cmps=Tx2.*YY)
            di = -ii * stretch_tiled
            d_x2 = XI1 * (-di) + XI2 * di
            d_str_cmps = d_x2 * weights_tiled

            # Compute second derivative (MATLAB: Hx2=XI1.*(-ddi)+XI2.*ddi; dd_str_comps=Hx2.*YY)
            ddi = -di * stretch_tiled * 2
            dd_x2 = XI1 * (-ddi) + XI2 * ddi
            dd_str_cmps = dd_x2 * weights_tiled
        else:
            shape = stretched_comps.shape
            d_str_cmps = np.empty(shape)
            dd_str_cmps = np.empty(shape)

        return stretched_comps, d_str_cmps, dd_str_cmps

    def apply_transformation_matrix(self, stretch=None, weights=None, residuals=None):
        """
        Computes the transformation matrix `stretch_transformed` for `residuals`,
        using scaling matrix `stretch` and coefficients `weights`.
        """

        if stretch is None:
            stretch = self.stretch
        if weights is None:
            weights = self.weights
        if residuals is None:
            residuals = self.residuals

        N, M = self.source_matrix.shape
        K = weights.shape[0]

        # Compute scaling matrix (MATLAB: AA = repmat(reshape(A,1,M*K).^-1,Nindex,1))
        AA = np.tile(stretch.reshape(1, M * K, order="F") ** -1, (N, 1))

        # Compute indices (MATLAB: ii = repmat((index-1)',1,K*M).*AA)
        ii = np.arange(N)[:, None] * AA  # Shape (N, M*K), replacing `index`

        # Weighting coefficients (MATLAB: YY = repmat(reshape(Y,1,M*K),Nindex,1))
        YY = np.tile(weights.reshape(1, M * K, order="F"), (N, 1))

        # Compute floor indices (MATLAB: II = floor(ii); II1 = min(II+1,N+1); II2 = min(II1+1,N+1))
        II = np.floor(ii).astype(int)
        II1 = np.minimum(II + 1, N)
        II2 = np.minimum(II1 + 1, N)

        # Assign directly (MATLAB: II1_ = II1; II2_ = II2)
        II1_ = II1
        II2_ = II2

        # Compute fractional part (MATLAB: iI = ii - II)
        iI = ii - II

        # Expand row indices (MATLAB: repm = repmat(1:K, Nindex, M))
        repm = np.tile(np.arange(K), (N, M))

        # Compute transformations (MATLAB: kro = kron(R(index,:), ones(1, K)))
        kron = np.kron(residuals, np.ones((1, K)))

        # (MATLAB: kroiI = kro .* (iI); iIYY = (iI-1) .* YY)
        kron_iI = kron * iI
        iIYY = (iI - 1) * YY

        # Construct sparse matrices (MATLAB: sparse(II1_,repm,kro.*-iIYY,(N+1),K))
        x2 = coo_matrix(((-kron * iIYY).flatten(), (II1_.flatten() - 1, repm.flatten())), shape=(N + 1, K)).tocsc()
        x3 = coo_matrix(((kron_iI * YY).flatten(), (II2_.flatten() - 1, repm.flatten())), shape=(N + 1, K)).tocsc()

        # Combine the last row into previous, then remove the last row
        x2[N - 1, :] += x2[N, :]
        x3[N - 1, :] += x3[N, :]
        x2 = x2[:-1, :]
        x3 = x3[:-1, :]

        stretch_transformed = x2 + x3

        return stretch_transformed

    def solve_quadratic_program(self, T, m, alg="trust-constr"):
        """
        Solves the quadratic program for updating y in stretched NMF using scipy.optimize:

            min J(y) = 0.5 * y^T Q y + d^T y
            subject to: 0 ≤ y ≤ 1

        Uses the 'trust-constr' solver with the analytical gradient and Hessian.
        Alternatively, can use scipy's L-BFGS-B algorithm, which supports bound
        constraints.

        Parameters:
        - T: (N, K) ndarray
            Matrix computed from getAfun(A(k, m), X[:, k]).
        - m: int
            Index of the current column in source_matrix.

        Returns:
        - y: (K,) ndarray
            Optimal solution for y, clipped to ensure non-negativity.
        """
        source_matrix_col = self.source_matrix[:, m]
        Q = T.T @ T
        d = -T.T @ source_matrix_col
        K = Q.shape[0]
        reg_factor = 1e-8 * np.linalg.norm(Q, ord="fro")
        Q += np.eye(K) * reg_factor

        def objective(y):
            return 0.5 * y @ Q @ y + d @ y

        def grad(y):
            return Q @ y + d

        if alg == "trust-constr":

            def hess(y):
                return csc_matrix(Q)  # sparse format for efficiency

            bounds = [(0, 1)] * K
            y0 = np.clip(-np.linalg.solve(Q + np.eye(K) * 1e-5, d), 0, 1)
            result = minimize(
                objective, y0, method="trust-constr", jac=grad, hess=hess, bounds=bounds, options={"verbose": 0}
            )
        elif alg == "L-BFGS-B":
            bounds = [(0, 1) for _ in range(K)]  # per-variable bounds
            y0 = np.clip(-np.linalg.solve(Q + np.eye(K) * 1e-5, d), 0, 1)  # Initial guess
            result = minimize(objective, y0, method="L-BFGS-B", jac=grad, bounds=bounds)

        return np.maximum(result.x, 0)

    def update_comps(self):
        """
        Updates `comps` using gradient-based optimization with adaptive step size L.
        """
        # Compute `stretched_comps` using the interpolation function
        stretched_comps, _, _ = self.apply_interpolation_matrix()  # Skip the other two outputs (derivatives)
        # Compute RA and RR
        intermediate_RA = stretched_comps.flatten(order="F").reshape(
            (self._signal_len * self._num_conditions, self._n_components), order="F"
        )
        RA = intermediate_RA.sum(axis=1).reshape((self._signal_len, self._num_conditions), order="F")
        RR = RA - self.source_matrix
        # Compute gradient `GraX`
        self.grad_comps = self.apply_transformation_matrix(
            residuals=RR
        ).toarray()  # toarray equivalent of full, make non-sparse

        # Compute initial step size `L0`
        L0 = np.linalg.eigvalsh(self.weights.T @ self.weights).max() * np.max(
            [self.stretch.max(), 1 / self.stretch.min()]
        )
        # Compute adaptive step size `L`
        if self._prev_comps is None:
            L = L0
        else:
            num = np.sum(
                (self.grad_comps - self._prev_grad_comps) * (self.comps - self._prev_comps)
            )  # Elem-wise multiply
            denom = np.linalg.norm(self.comps - self._prev_comps, "fro") ** 2  # Frobenius norm squared
            L = num / denom if denom > 0 else L0
            if L <= 0:
                L = L0

        # Store our old component matrix before updating because it is used in step selection
        self._prev_comps = self.comps.copy()

        while True:  # iterate updating components
            comps_step = self._prev_comps - self.grad_comps / L
            # Solve x^3 + p*x + q = 0 for the largest real root
            self.comps = np.square(cubic_largest_real_root(-comps_step, self.eta / (2 * L)))
            # Mask values that should be set to zero
            mask = self.comps**2 * L / 2 - L * self.comps * comps_step + self.eta * np.sqrt(self.comps) < 0
            self.comps = mask * self.comps

            objective_improvement = self._objective_history[-1] - self.get_objective_function(
                residuals=self.get_residual_matrix()
            )

            # Check if objective function improves
            if objective_improvement > 0:
                break
            # If not, increase L (step size)
            L *= 2
            if np.isinf(L):
                break

    def update_weights(self):
        """
        Updates weights using matrix operations, solving a quadratic program via to do so.
        """

        for m in range(self._num_conditions):
            T = np.zeros((self._signal_len, self._n_components))  # Initialize T as an (N, K) zero matrix

            # Populate T using apply_interpolation
            for k in range(self._n_components):
                T[:, k] = self.apply_interpolation(self.stretch[k, m], self.comps[:, k], return_derivatives=True)[
                    0
                ].squeeze()

            # Solve quadratic problem for y
            y = self.solve_quadratic_program(T=T, m=m)

            # Update Y
            self.weights[:, m] = y

    def regularize_function(self, stretch=None):
        """
        Computes the regularization function, gradient, and Hessian for optimization.
        Returns:
        - fun: Objective function value (scalar)
        - gra: Gradient (same shape as A)
        """
        if stretch is None:
            stretch = self.stretch

        # Compute interpolated matrices
        stretched_comps, d_str_cmps, dd_str_cmps = self.apply_interpolation_matrix(
            stretch=stretch, return_derivatives=True
        )

        # Compute residual
        intermediate_RA = stretched_comps.flatten(order="F").reshape(
            (self._signal_len * self._num_conditions, self._n_components), order="F"
        )
        RA = intermediate_RA.sum(axis=1).reshape((self._signal_len, self._num_conditions), order="F")
        RA = RA - self.source_matrix

        # Compute objective function
        reg_func = self.get_objective_function(RA, stretch)

        # Compute gradient
        tiled_derivative = np.sum(d_str_cmps * np.tile(RA, (1, self._n_components)), axis=0)
        der_reshaped = np.asarray(tiled_derivative).reshape((self._num_conditions, self._n_components), order="F")
        func_grad = (
            der_reshaped.T + self.rho * stretch @ self.spline_smooth_operator.T @ self.spline_smooth_operator
        )

        return reg_func, func_grad

    def update_stretch(self):
        """
        Updates stretching matrix using constrained optimization (equivalent to fmincon in MATLAB).
        """

        # Flatten stretch for compatibility with the optimizer (since SciPy expects 1D input)
        stretch_init_vec = self.stretch.flatten()

        # Define the optimization function
        def objective(stretch_vec):
            stretch_matrix = stretch_vec.reshape(self.stretch.shape)  # Reshape back to matrix form
            func, grad = self.regularize_function(stretch_matrix)
            grad = grad.flatten()
            return func, grad

        # Optimization constraints: lower bound 0.1, no upper bound
        bounds = [(0.1, None)] * stretch_init_vec.size  # Equivalent to 0.1 * ones(K, M)

        # Solve optimization problem (equivalent to fmincon)
        result = minimize(
            fun=lambda stretch_vec: objective(stretch_vec)[0],  # Objective function
            x0=stretch_init_vec,  # Initial guess
            method="trust-constr",  # Equivalent to 'trust-region-reflective'
            jac=lambda stretch_vec: objective(stretch_vec)[1],  # Gradient
            bounds=bounds,  # Lower bounds on stretch
            # TODO: A Hessian can be incorporated for better convergence.
        )

        # Update stretch with the optimized values
        self.stretch = result.x.reshape(self.stretch.shape)


def cubic_largest_real_root(p, q):
    """
    Vectorized solver for x^3 + p*x + q = 0.
    Returns the largest real root element-wise.
    """
    # calculate the discriminant
    delta = (q / 2) ** 2 + (p / 3) ** 3
    sqrt_delta = np.sqrt(np.abs(delta))

    # When delta >= 0: one real root
    A = np.cbrt(-q / 2 + sqrt_delta)
    B = np.cbrt(-q / 2 - sqrt_delta)
    root1 = A + B

    # When delta < 0: three real roots, use trigonometric method
    phi = np.arccos(-q / (2 * np.sqrt(-((p / 3) ** 3) + 1e-12)))
    r = 2 * np.sqrt(-p / 3)
    root2 = r * np.cos(phi / 3)

    # Choose correct root depending on sign of delta
    return np.where(delta >= 0, root1, root2)
