import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, csc_matrix, diags


class SNMFOptimizer:
    """A implementation of stretched NMF (sNMF), including sparse stretched NMF.

    Instantiating the SNMFOptimizer class runs all the analysis immediately.
    The results matrices can then be accessed as instance attributes
    of the class (X, Y, and A).

    For more information on sNMF, please reference:
    Gu, R., Rakita, Y., Lan, L. et al. Stretched non-negative matrix factorization.
    npj Comput Mater 10, 193 (2024). https://doi.org/10.1038/s41524-024-01377-5
    """

    def __init__(
        self,
        MM,
        Y0=None,
        X0=None,
        A=None,
        rho=1e12,
        eta=610,
        max_iter=500,
        tol=5e-7,
        n_components=None,
        random_state=None,
    ):
        """Initialize an instance of SNMF and run the optimization

        Parameters
        ----------
        MM : ndarray
            The data to be decomposed. Shape is (length_of_signal, number_of_conditions).
        Y0 : ndarray
            The initial guesses for the component weights at each stretching condition.
            Shape is (number of components, number ofconditions) Must be provided if
            n_components is not provided. Will override n_components if both are provided.
        X0 : ndarray
            The initial guesses for the intensities of each component per
            row/sample/angle. Shape is (length_of_signal, number_of_components).
        A : ndarray
            The initial guesses for the stretching factor for each component, at each
            condition. Shape is (number_of_components, number_of_conditions).
        rho : float
            The stretching factor that influences the decomposition. Zero corresponds to no
            stretching present. Relatively insensitive and typically adjusted in powers of 10.
        eta : float
            The sparsity factor than influences the decomposition. Should be set to zero for
            non-sparse data such as PDF. Can be used to improve results for sparse data such
            as XRD, but due to instability, should be used only after first selecting the
            best value for rho.
        max_iter : int
            The maximum number of times to update each of A, X, and Y before stopping
            the optimization.
        tol : float
            The convergence threshold. This is the minimum fractional improvement in the
            objective function to allow without terminating the optimization. Note that
            a minimum of 20 updates are run before this parameter is checked.
        n_components : int
            The number of components to attempt to extract from MM. Note that this will
            be overridden by Y0 if that is provided, but must be provided if no Y0 is
            provided.
        random_state : int
            The seed for the initial matrices used in the optimization.
        """

        self.MM = MM
        self.X0 = X0
        self.Y0 = Y0
        self.A = A
        self.rho = rho
        self.eta = eta
        # Capture matrix dimensions
        self.N, self.M = MM.shape
        self.num_updates = 0
        self.rng = np.random.default_rng(random_state)

        if Y0 is None:
            if n_components is None:
                raise ValueError("Must provide either Y0 or n_components.")
            else:
                self.K = n_components
                self.Y0 = self.rng.beta(a=2.5, b=1.5, size=(self.K, self.M))
        else:
            self.K = Y0.shape[0]

        if self.A is None:
            self.A = np.ones((self.K, self.M)) + self.rng.normal(0, 1e-3, size=(self.K, self.M))
        if self.X0 is None:
            self.X0 = self.rng.random((self.N, self.K))

        self.X = np.maximum(0, self.X0)
        self.Y = np.maximum(0, self.Y0)

        # Second-order spline: Tridiagonal (-2 on diagonal, 1 on sub/superdiagonals)
        self.P = 0.25 * diags([1, -2, 1], offsets=[0, 1, 2], shape=(self.M - 2, self.M))
        self.PP = self.P.T @ self.P

        # Set up residual matrix, objective function, and history
        self.R = self.get_residual_matrix()
        self.objective_function = self.get_objective_function()
        self.objective_difference = None
        self.objective_history = [self.objective_function]

        # Set up tracking variables for updateX()
        self.preX = None
        self.GraX = np.zeros_like(self.X)  # Gradient of X (zeros for now)
        self.preGraX = np.zeros_like(self.X)  # Previous gradient of X (zeros for now)

        regularization_term = 0.5 * rho * np.linalg.norm(self.P @ self.A.T, "fro") ** 2
        sparsity_term = eta * np.sum(np.sqrt(self.X))  # Square root penalty
        print(
            f"Start, Objective function: {self.objective_function:.5e}"
            f", Obj - reg/sparse: {self.objective_function - regularization_term - sparsity_term:.5e}"
        )

        # Main optimization loop
        for iter in range(max_iter):
            self.optimize_loop()
            # Print diagnostics
            regularization_term = 0.5 * rho * np.linalg.norm(self.P @ self.A.T, "fro") ** 2
            sparsity_term = eta * np.sum(np.sqrt(self.X))  # Square root penalty
            print(
                f"Num_updates: {self.num_updates}, "
                f"Obj fun: {self.objective_function:.5e}, "
                f"Obj - reg/sparse: {self.objective_function - regularization_term - sparsity_term:.5e}, "
                f"Iter: {iter}"
            )

            # Convergence check: Stop if diffun is small and at least 20 iterations have passed
            print(self.objective_difference, " < ", self.objective_function * tol)
            if self.objective_difference < self.objective_function * tol and iter >= 20:
                break

        # Normalize our results
        Y_row_max = np.max(self.Y, axis=1, keepdims=True)
        self.Y = self.Y / Y_row_max
        A_row_max = np.max(self.A, axis=1, keepdims=True)
        self.A = self.A / A_row_max
        # loop to normalize X
        # effectively just re-running class with non-normalized X, normalized Y/A as inputs, then only update X
        # reset difference trackers and initialize
        self.preX = None
        self.GraX = np.zeros_like(self.X)  # Gradient of X (zeros for now)
        self.preGraX = np.zeros_like(self.X)  # Previous gradient of X (zeros for now)
        self.R = self.get_residual_matrix()
        self.objective_function = self.get_objective_function()
        self.objective_difference = None
        self.objective_history = [self.objective_function]
        for norm_iter in range(100):
            self.updateX()
            self.R = self.get_residual_matrix()
            self.objective_function = self.get_objective_function()
            print(f"Objective function after normX: {self.objective_function:.5e}")
            self.objective_history.append(self.objective_function)
            self.objective_difference = self.objective_history[-2] - self.objective_history[-1]
            if self.objective_difference < self.objective_function * tol and norm_iter >= 20:
                break
        # end of normalization (and program)
        # note that objective function may not fully recover after normalization, this is okay
        print("Finished optimization.")

    def optimize_loop(self):
        self.preGraX = self.GraX.copy()
        self.updateX()
        self.num_updates += 1
        self.R = self.get_residual_matrix()
        self.objective_function = self.get_objective_function()
        print(f"Objective function after updateX: {self.objective_function:.5e}")
        self.objective_history.append(self.objective_function)
        if self.objective_difference is None:
            self.objective_difference = self.objective_history[-1] - self.objective_function

        # Now we update Y
        self.updateY2()
        self.num_updates += 1
        self.R = self.get_residual_matrix()
        self.objective_function = self.get_objective_function()
        print(f"Objective function after updateY2: {self.objective_function:.5e}")
        self.objective_history.append(self.objective_function)

        self.updateA2()

        self.num_updates += 1
        self.R = self.get_residual_matrix()
        self.objective_function = self.get_objective_function()
        print(f"Objective function after updateA2: {self.objective_function:.5e}")
        self.objective_history.append(self.objective_function)
        self.objective_difference = self.objective_history[-2] - self.objective_history[-1]

    def apply_interpolation(self, a, x, return_derivatives=False):
        """
        Applies an interpolation-based transformation to `x` based on scaling `a`.
        Also can compute first (`Tx`) and second (`Hx`) derivatives.
        """
        N = len(x)

        # Ensure `a` is an array and reshape for broadcasting
        a = np.atleast_1d(np.asarray(a))  # Ensures a is at least 1D

        # Compute fractional indices, broadcasting over `a`
        ii = np.arange(N)[:, None] / a  # Shape (N, M)

        II = np.floor(ii).astype(int)  # Integer part (still (N, M))
        valid_mask = II < (N - 1)  # Ensure indices are within bounds

        # Apply valid_mask to keep correct indices
        idx_int = np.where(valid_mask, II, N - 2)  # Prevent out-of-bounds indexing (previously "I")
        idx_frac = np.where(valid_mask, ii, II)  # Keep aligned (previously "i")

        # Ensure x is a 1D array
        x = np.asarray(x).ravel()

        # Compute Ax (linear interpolation)
        Ax = x[idx_int] * (1 - idx_frac + idx_int) + x[np.minimum(idx_int + 1, N - 1)] * (idx_frac - idx_int)

        # Fill the tail with the last valid value
        Ax_tail = np.full((N - len(idx_int), Ax.shape[1]), Ax[-1, :])
        Ax = np.vstack([Ax, Ax_tail])

        if return_derivatives:
            # Compute first derivative (Tx)
            di = -idx_frac / a
            Tx = x[idx_int] * (-di) + x[np.minimum(idx_int + 1, N - 1)] * di
            Tx = np.vstack([Tx, np.zeros((N - len(idx_int), Tx.shape[1]))])

            # Compute second derivative (Hx)
            ddi = -di / a + idx_frac * a**-2
            Hx = x[idx_int] * (-ddi) + x[np.minimum(idx_int + 1, N - 1)] * ddi
            Hx = np.vstack([Hx, np.zeros((N - len(idx_int), Hx.shape[1]))])
        else:
            # Make placeholders
            Tx = np.empty(Ax.shape)
            Hx = np.empty(Ax.shape)

        return Ax, Tx, Hx

    def get_residual_matrix(self, X=None, Y=None, A=None):
        # Initialize residual matrix as negative of MM
        # In MATLAB this is getR
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y
        if A is None:
            A = self.A
        R = -self.MM.copy()
        # Compute transformed X for all (k, m) pairs
        for k in range(Y.shape[0]):  # K
            Ax, _, _ = self.apply_interpolation(A[k, :], X[:, k])  # Only calculate Ax
            R += Y[k, :] * Ax  # Element-wise scaling and sum
        return R

    def get_objective_function(self, R=None, A=None):
        if R is None:
            R = self.R
        if A is None:
            A = self.A
        residual_term = 0.5 * np.linalg.norm(R, "fro") ** 2
        regularization_term = 0.5 * self.rho * np.linalg.norm(self.P @ A.T, "fro") ** 2
        sparsity_term = self.eta * np.sum(np.sqrt(self.X))  # Square root penalty
        # Final objective function value
        function = residual_term + regularization_term + sparsity_term
        return function

    def apply_interpolation_matrix(self, X=None, Y=None, A=None, return_derivatives=False):
        """
        Applies an interpolation-based transformation to the matrix `X` using `A`,
        weighted by `Y`. Optionally computes first (`Tx`) and second (`Hx`) derivatives.
        Equivalent to getAfun_matrix in MATLAB.
        """

        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y
        if A is None:
            A = self.A

        N, M = self.MM.shape
        K = Y.shape[0]

        # Compute scaled indices (MATLAB: AA = repmat(reshape(A',1,M*K).^-1, N,1))
        A_flat = A.reshape(1, M * K) ** -1
        AA = np.tile(A_flat, (N, 1))

        # Compute `ii` (MATLAB: ii = repmat((0:N-1)',1,K*M).*AA)
        ii = np.tile(np.arange(N)[:, None], (1, M * K)) * AA

        # Weighting matrix (MATLAB: YY = repmat(reshape(Y',1,M*K), N,1))
        Y_flat = Y.reshape(1, M * K)
        YY = np.tile(Y_flat, (N, 1))

        # Bias for indexing into reshaped X (MATLAB: bias = kron((0:K-1)*(N+1),ones(N,M)))
        # TODO break this up or describe what it does better
        bias = np.kron(np.arange(K) * (N + 1), np.ones((N, M), dtype=int)).reshape(N, K * M)

        # Handle boundary conditions for interpolation (MATLAB: X1=[X;X(end,:)])
        X1 = np.vstack([X, X[-1, :]])  # Duplicate last row (like MATLAB)

        # Compute floor indices (MATLAB: II = floor(ii); II1=min(II+1,N+1); II2=min(II1+1,N+1))
        II = np.floor(ii).astype(int)

        II1 = np.minimum(II + 1, N)
        II2 = np.minimum(II1 + 1, N)

        # Compute fractional part (MATLAB: iI = ii - II)
        iI = ii - II

        # Compute offset indices (MATLAB: II1_ = II1 + bias; II2_ = II2 + bias)
        II1_ = II1 + bias
        II2_ = II2 + bias

        # Extract values (MATLAB: XI1 = reshape(X1(II1_), N, K*M); XI2 = reshape(X1(II2_), N, K*M))
        # Note: this "-1" corrects an off-by-one error that may have originated in an earlier line
        XI1 = X1.flatten(order="F")[(II1_ - 1).ravel()].reshape(
            N, K * M
        )  # order = F uses FORTRAN, column major order
        XI2 = X1.flatten(order="F")[(II2_ - 1).ravel()].reshape(N, K * M)

        # Interpolation (MATLAB: Ax2=XI1.*(1-iI)+XI2.*(iI); Ax=Ax2.*YY)
        Ax2 = XI1 * (1 - iI) + XI2 * iI
        Ax = Ax2 * YY  # Apply weighting

        if return_derivatives:
            # Compute first derivative (MATLAB: Tx2=XI1.*(-di)+XI2.*di; Tx=Tx2.*YY)
            di = -ii * AA
            Tx2 = XI1 * (-di) + XI2 * di
            Tx = Tx2 * YY

            # Compute second derivative (MATLAB: Hx2=XI1.*(-ddi)+XI2.*ddi; Hx=Hx2.*YY)
            ddi = -di * AA * 2
            Hx2 = XI1 * (-ddi) + XI2 * ddi
            Hx = Hx2 * YY
        else:
            shape = Ax.shape
            Tx = np.empty(shape)
            Hx = np.empty(shape)

        return Ax, Tx, Hx

    def apply_transformation_matrix(self, A=None, Y=None, R=None):
        """
        Computes the transformation matrix `AT` for residual `R`,
        using scaling matrix `A` and weight coefficients `Y`.
        """

        if A is None:
            A = self.A
        if Y is None:
            Y = self.Y
        if R is None:
            R = self.R

        N, M = self.MM.shape
        K = Y.shape[0]

        # Compute scaling matrix (MATLAB: AA = repmat(reshape(A,1,M*K).^-1,Nindex,1))
        AA = np.tile(A.reshape(1, M * K, order="F") ** -1, (N, 1))

        # Compute indices (MATLAB: ii = repmat((index-1)',1,K*M).*AA)
        ii = np.arange(N)[:, None] * AA  # Shape (N, M*K), replacing `index`

        # Weighting coefficients (MATLAB: YY = repmat(reshape(Y,1,M*K),Nindex,1))
        YY = np.tile(Y.reshape(1, M * K, order="F"), (N, 1))

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
        kro = np.kron(R, np.ones((1, K)))

        # (MATLAB: kroiI = kro .* (iI); iIYY = (iI-1) .* YY)
        kroiI = kro * iI
        iIYY = (iI - 1) * YY

        # Construct sparse matrices (MATLAB: sparse(II1_,repm,kro.*-iIYY,(N+1),K))
        x2 = coo_matrix(((-kro * iIYY).flatten(), (II1_.flatten() - 1, repm.flatten())), shape=(N + 1, K)).tocsc()
        x3 = coo_matrix(((kroiI * YY).flatten(), (II2_.flatten() - 1, repm.flatten())), shape=(N + 1, K)).tocsc()

        # Combine the last row into previous, then remove the last row
        x2[N - 1, :] += x2[N, :]
        x3[N - 1, :] += x3[N, :]
        x2 = x2[:-1, :]
        x3 = x3[:-1, :]

        AT = x2 + x3

        return AT

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
            Index of the current column in MM.

        Returns:
        - y: (K,) ndarray
            Optimal solution for y, clipped to ensure non-negativity.
        """
        MM_col = self.MM[:, m]
        Q = T.T @ T
        d = -T.T @ MM_col
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

    def updateX(self):
        """
        Updates `X` using gradient-based optimization with adaptive step size L.
        """
        # Compute `AX` using the interpolation function
        AX, _, _ = self.apply_interpolation_matrix()  # Skip the other two outputs
        # Compute RA and RR
        intermediate_RA = AX.flatten(order="F").reshape((self.N * self.M, self.K), order="F")
        RA = intermediate_RA.sum(axis=1).reshape((self.N, self.M), order="F")
        RR = RA - self.MM
        # Compute gradient `GraX`
        self.GraX = self.apply_transformation_matrix(R=RR).toarray()  # toarray equivalent of full, make non-sparse

        # Compute initial step size `L0`
        L0 = np.linalg.eigvalsh(self.Y.T @ self.Y).max() * np.max([self.A.max(), 1 / self.A.min()])
        # Compute adaptive step size `L`
        if self.preX is None:
            L = L0
        else:
            num = np.sum((self.GraX - self.preGraX) * (self.X - self.preX))  # Element-wise multiplication
            denom = np.linalg.norm(self.X - self.preX, "fro") ** 2  # Frobenius norm squared
            L = num / denom if denom > 0 else L0
            if L <= 0:
                L = L0

        # Store our old X before updating because it is used in step selection
        self.preX = self.X.copy()

        while True:  # iterate updating X
            x_step = self.preX - self.GraX / L
            # Solve x^3 + p*x + q = 0 for the largest real root
            self.X = np.square(cubic_largest_real_root(-x_step, self.eta / (2 * L)))
            # Mask values that should be set to zero
            mask = self.X**2 * L / 2 - L * self.X * x_step + self.eta * np.sqrt(self.X) < 0
            self.X = mask * self.X

            objective_improvement = self.objective_history[-1] - self.get_objective_function(
                R=self.get_residual_matrix()
            )

            # Check if objective function improves
            if objective_improvement > 0:
                break
            # If not, increase L (step size)
            L *= 2
            if np.isinf(L):
                break

    def updateY2(self):
        """
        Updates Y using matrix operations, solving a quadratic program via `solve_mkr_box`.
        """

        K = self.K
        N = self.N
        M = self.M

        for m in range(M):
            T = np.zeros((N, K))  # Initialize T as an (N, K) zero matrix

            # Populate T using apply_interpolation
            for k in range(K):
                T[:, k] = self.apply_interpolation(self.A[k, m], self.X[:, k], return_derivatives=True)[
                    0
                ].squeeze()

            # Solve quadratic problem for y
            y = self.solve_quadratic_program(T=T, m=m)

            # Update Y
            self.Y[:, m] = y

    def regularize_function(self, A=None):
        """
        Computes the regularization function, gradient, and Hessian for optimization.
        Returns:
        - fun: Objective function value (scalar)
        - gra: Gradient (same shape as A)
        """
        if A is None:
            A = self.A

        K = self.K
        M = self.M
        N = self.N

        # Compute interpolated matrices
        AX, TX, HX = self.apply_interpolation_matrix(A=A, return_derivatives=True)

        # Compute residual
        intermediate_RA = AX.flatten(order="F").reshape((N * M, K), order="F")
        RA = intermediate_RA.sum(axis=1).reshape((N, M), order="F")
        RA = RA - self.MM

        # Compute objective function
        fun = self.get_objective_function(RA, A)

        # Compute gradient
        tiled_derivative = np.sum(TX * np.tile(RA, (1, K)), axis=0)
        der_reshaped = np.asarray(tiled_derivative).reshape((M, K), order="F")
        gra = der_reshaped.T + self.rho * A @ self.P.T @ self.P

        return fun, gra

    def updateA2(self):
        """
        Updates matrix A using constrained optimization (equivalent to fmincon in MATLAB).
        """

        # Flatten A for compatibility with the optimizer (since SciPy expects 1D input)
        A_initial = self.A.flatten()

        # Define the optimization function
        def objective(A_vec):
            A_matrix = A_vec.reshape(self.A.shape)  # Reshape back to matrix form
            fun, gra = self.regularize_function(A_matrix)
            gra = gra.flatten()
            return fun, gra

        # Optimization constraints: lower bound 0.1, no upper bound
        bounds = [(0.1, None)] * A_initial.size  # Equivalent to 0.1 * ones(K, M)

        # Solve optimization problem (equivalent to fmincon)
        result = minimize(
            fun=lambda A_vec: objective(A_vec)[0],  # Objective function
            x0=A_initial,  # Initial guess
            method="trust-constr",  # Equivalent to 'trust-region-reflective'
            jac=lambda A_vec: objective(A_vec)[1],  # Gradient
            bounds=bounds,  # Lower bounds on A
        )

        # Update A with the optimized values
        self.A = result.x.reshape(self.A.shape)


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
