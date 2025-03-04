import numpy as np
from scipy.optimize import minimize
from scipy.sparse import block_diag, coo_matrix, diags, spdiags


class SNMFOptimizer:
    def __init__(self, MM, Y0, X0=None, A=None, rho=1e18, eta=1, maxiter=300):
        print("Initializing SNMF Optimizer")
        self.MM = MM
        self.X0 = X0
        self.Y0 = Y0
        self.A = A
        self.rho = rho
        self.eta = eta
        self.maxiter = maxiter
        # Capture matrix dimensions
        self.N, self.M = MM.shape
        self.K = Y0.shape[0]

        # Initialize A, X0 if not provided
        if self.A is None:
            self.A = np.ones((self.K, self.M)) + np.random.randn(self.K, self.M) * 1e-3  # Small perturbation
        if self.X0 is None:
            self.X0 = np.random.rand(self.N, self.K)  # Ensures values in [0,1], no need for clipping
            print(self.X0)
        # Initialize solution matrices to be iterated on
        self.X = np.maximum(0, self.X0)
        self.Y = np.maximum(0, self.Y0)

        # Second-order spline: Tridiagonal (-2 on diagonal, 1 on sub/superdiagonals)
        # TODO re-add the option to have a first-order spline
        self.P = 0.25 * diags([1, -2, 1], offsets=[-1, 0, 1], shape=(self.M - 2, self.M))
        self.PP = self.P.T @ self.P
        PPPP = block_diag([self.PP] * self.K, format="csr")
        # Generate interleaved index sequence
        seq = np.arange(self.M * self.K).reshape(self.K, self.M).T.ravel()
        # Reorder rows and columns of PPPP (blocks interleaved instead of stacked)
        self.PPPP = PPPP[seq, :][:, seq]

        # Set up residual matrix, objective function, and history
        self.R = self.get_residual_matrix()
        self.objective_function = self.get_objective_function()
        self.objective_difference = None
        self.objective_history = [self.objective_function]

        # Set up tracking variables for the loop
        self.preX = np.zeros_like(self.X)  # Previously stored X (zeros for now)
        self.curX = self.X.copy()  # Current X, initialized to the given X
        self.preGraX = np.zeros_like(self.X)  # Previously stored gradient (zeros for now)
        self.curGraX = None  # Current gradient (None for now)

        for outiter in range(self.maxiter):
            self.outiter = outiter
            self.outer_loop()
            # Convergence check: Stop if diffun is small and at least 20 iterations have passed
            if self.objective_difference < self.objective_function * 1e-6 and outiter >= 20:
                break

    def outer_loop(self):
        # This inner loop runs up to four times per outer loop, making updates to X, Y
        for iter in range(4):
            self.iter = iter
            self.updateX()
            self.R = self.get_residual_matrix()
            self.objective_function = self.get_objective_function()
            self.objective_history.append(self.objective_function)
            if self.outiter == 0 and self.iter == 0:
                self.objective_difference = self.objective_history[-1] - self.objective_function
            self.preX = self.curX.copy()
            self.curX = self.X.copy()
            self.preGraX = self.curGraX.copy()

            # Now we update Y
            self.updateY2()
            self.R = self.get_residual_matrix()
            self.objective_function = self.get_objective_function()
            self.objective_history.append(self.objective_function)

            # Check whether to break out early
            if len(self.objective_history) >= 3:  # Ensure at least 3 values exist
                if self.objective_history[-3] - self.objective_function < self.objective_difference * 1e-3:
                    break  # Stop if improvement is too small

        # Then we update A
        self.updateA2()
        self.R = self.get_residual_matrix()
        self.objective_function = self.get_objective_function()
        self.objective_history.append(self.objective_function)
        self.objective_difference = self.objective_history[-1] - self.objective_function

    def apply_interpolation(self, a, x):
        """
        Applies an interpolation-based transformation to `x` based on scaling `a`.
        Also computes first (`Tx`) and second (`Hx`) derivatives.
        This replicates MATLAB-style behavior without explicit reshaping.
        """
        N = len(x)

        # Ensure `a` is an array and reshape for broadcasting
        a = np.atleast_1d(np.asarray(a))  # Ensures a is at least 1D
        # a = np.asarray(a)[None, :]  # Shape (1, M) to allow broadcasting

        # Compute fractional indices, broadcasting over `a`
        ii = np.arange(N)[:, None] / a  # Shape (N, M)

        II = np.floor(ii).astype(int)  # Integer part (still (N, M))
        valid_mask = II < (N - 1)  # Ensure indices are within bounds

        # Apply valid_mask to keep correct indices
        I = np.where(valid_mask, II, N - 2)  # Prevent out-of-bounds indexing
        i = np.where(valid_mask, ii, II)  # Keep i aligned

        # Ensure x is a 1D array
        x = np.asarray(x).ravel()

        # Compute Ax (linear interpolation)
        Ax = x[I] * (1 - i + I) + x[np.minimum(I + 1, N - 1)] * (i - I)

        # Fill the tail with the last valid value
        Ax_tail = np.full((N - len(I), Ax.shape[1]), Ax[-1, :])
        Ax = np.vstack([Ax, Ax_tail])

        # Compute first derivative (Tx)
        di = -i / a
        Tx = x[I] * (-di) + x[np.minimum(I + 1, N - 1)] * di
        Tx = np.vstack([Tx, np.zeros((N - len(I), Tx.shape[1]))])

        # Compute second derivative (Hx)
        ddi = -di / a + i * a**-2
        Hx = x[I] * (-ddi) + x[np.minimum(I + 1, N - 1)] * ddi
        Hx = np.vstack([Hx, np.zeros((N - len(I), Hx.shape[1]))])

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
        for k in range(self.Y.shape[0]):  # K
            Ax, _, _ = self.apply_interpolation(self.A[k, :], self.X[:, k])  # Only use Ax
            R += self.Y[k, :] * Ax  # Element-wise scaling and sum
        return R

    def get_objective_function(self, R=None, A=None):
        if R is None:
            R = self.R
        if A is None:
            A = self.A
        residual_term = 0.5 * np.linalg.norm(R, "fro") ** 2
        # original code selected indices, but for now we'll compute the norm over the whole matrix
        # residual_term = 0.5 * np.linalg.norm(self.R[index, :], 'fro') ** 2
        regularization_term = 0.5 * self.rho * np.linalg.norm(self.P @ A.T, "fro") ** 2
        sparsity_term = self.eta * np.sum(np.sqrt(self.X))  # Square root penalty
        # Final objective function value
        function = residual_term + regularization_term + sparsity_term
        return function

    def apply_interpolation_matrix(self, A=None):
        """
        Applies an interpolation-based transformation to the matrix `X` using `A`,
        weighted by `Y`. Optionally computes first (`Tx`) and second (`Hx`) derivatives.
        Equivalent to getAfun_matrix in MATLAB.
        """
        N, K = self.X.shape
        M = self.Y.shape[1]

        if A is None:
            A = self.A

        # Compute scaled indices (MATLAB: AA = repmat(reshape(A',1,M*K).^-1, N,1))
        AA = np.tile(self.A.T.reshape(1, M * K) ** -1, (N, 1))  # Corrects broadcasting

        # Compute `ii` (MATLAB: ii = repmat((0:N-1)',1,K*M).*AA)
        ii = np.tile(np.arange(N)[:, None], (1, M * K)) * AA

        # Weighting matrix (MATLAB: YY = repmat(reshape(Y',1,M*K), N,1))
        YY = np.tile(self.Y.T.reshape(1, M * K), (N, 1))

        # Bias for indexing into reshaped X (MATLAB: bias = kron((0:K-1)*(N+1),ones(N,M)))
        bias = np.kron(np.arange(K) * (N + 1), np.ones((N, M), dtype=int)).reshape(N, K * M)

        # Handle boundary conditions for interpolation (MATLAB: X1=[X;X(end,:)])
        X1 = np.vstack([self.X, self.X[-1, :]])  # Duplicate last row (like MATLAB)

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
        XI1 = X1.flatten()[II1_.ravel()].reshape(N, K * M)
        XI2 = X1.flatten()[II2_.ravel()].reshape(N, K * M)

        # Interpolation (MATLAB: Ax2=XI1.*(1-iI)+XI2.*(iI); Ax=Ax2.*YY)
        Ax2 = XI1 * (1 - iI) + XI2 * iI
        Ax = Ax2 * YY  # Apply weighting

        # Compute first derivative (MATLAB: Tx2=XI1.*(-di)+XI2.*di; Tx=Tx2.*YY)
        di = -ii * AA
        Tx2 = XI1 * (-di) + XI2 * di
        Tx = Tx2 * YY

        # Compute second derivative (MATLAB: Hx2=XI1.*(-ddi)+XI2.*ddi; Hx=Hx2.*YY)
        ddi = -di * AA * 2
        Hx2 = XI1 * (-ddi) + XI2 * ddi
        Hx = Hx2 * YY

        return Ax, Tx, Hx

    def apply_transformation_matrix(self, R=None):
        """
        Computes the transformation matrix `AT` for residual `R`,
        using scaling matrix `A` and weight coefficients `Y`.
        """
        if R is None:
            R = self.R

        # Compute scaling matrix (MATLAB: AA = repmat(reshape(A,1,M*K).^-1,Nindex,1))
        AA = np.tile(self.A.reshape(1, self.M * self.K) ** -1, (self.N, 1))

        # Compute indices (MATLAB: ii = repmat((index-1)',1,K*M).*AA)
        ii = np.arange(self.N)[:, None] * AA  # Shape (N, M*K), replacing `index`

        # Weighting coefficients (MATLAB: YY = repmat(reshape(Y,1,M*K),Nindex,1))
        YY = np.tile(self.Y.reshape(1, self.M * self.K), (self.N, 1))  # Shape (N, M*K)

        # Compute floor indices (MATLAB: II = floor(ii); II1 = min(II+1,N+1); II2 = min(II1+1,N+1))
        II = np.floor(ii).astype(int)
        II1 = np.minimum(II + 1, self.N)
        II2 = np.minimum(II1 + 1, self.N)

        # Assign directly (MATLAB: II1_ = II1; II2_ = II2)
        II1_ = II1
        II2_ = II2

        # Compute fractional part (MATLAB: iI = ii - II)
        iI = ii - II

        # Expand row indices (MATLAB: repm = repmat(1:K, Nindex, M))
        repm = np.tile(np.arange(self.K), (self.N, self.M))  # indexed to zero here

        # Compute transformations (MATLAB: kro = kron(R(index,:), ones(1, K)))
        kro = np.kron(R, np.ones((1, self.K)))  # Use full `R`

        # (MATLAB: kroiI = kro .* (iI); iIYY = (iI-1) .* YY)
        kroiI = kro * iI
        iIYY = (iI - 1) * YY

        # Construct sparse matrices (MATLAB: sparse(II1_,repm,kro.*-iIYY,(N+1),K))
        x2 = coo_matrix(((-kro * iIYY).flatten(), (II1_.flatten(), repm.flatten())), shape=(self.N + 1, self.K))
        x3 = coo_matrix(((kroiI * YY).flatten(), (II2_.flatten(), repm.flatten())), shape=(self.N + 1, self.K))

        # Convert to LIL format
        x2 = x2.tolil()
        x3 = x3.tolil()
        x2[self.N - 1, :] += x2[self.N, :]
        x3[self.N - 1, :] += x3[self.N, :]
        x2 = x2[:-1, :].tocsc()  # Remove last row
        x3 = x3[:-1, :].tocsc()

        # Final transformation matrix
        AT = (x2 + x3).tocsc()

        return AT

    def solve_mkr_box(self, T, m):
        """
        Solves the quadratic program for updating y in stretched NMF:

            min J(y) = 0.5 * y^T Q y + d^T y
            subject to: 0 <= y <= 1

        where:
            Q = T(index,:)' * T(index,:)
            d = -T(index,:)' * MM(index,m)

        Parameters:
        - T: (N, K) matrix computed from getAfun(A(k,m), X(:,k))
        - MM: (N, M) matrix in the stretched NMF formulation
        - index: Indices used to select a subset of rows
        - m: Column index for MM

        Returns:
        - y: (K,) optimal solution
        """

        # Compute Q and d
        Q = T.T @ T  # Gram matrix (K x K) using all rows
        d = -T.T @ self.MM[:, m]  # Linear term (K,) using all rows
        K = Q.shape[0]  # Number of variables

        # Objective function
        def objective(y):
            return 0.5 * np.dot(y, Q @ y) + np.dot(d, y)

        # Bounds (0 ≤ y ≤ 1)
        bounds = [(0, 1)] * K

        # Initial guess (per cA=1, start at upper bound y=1)
        y0 = np.ones(K)

        # Solve QP
        result = minimize(objective, y0, bounds=bounds, method="SLSQP")

        return result.x  # Optimal solution

    def updateX(self):
        """
        Updates `X` using gradient-based optimization with adaptive step size L.
        """
        # Compute `AX` using the interpolation function
        AX, _, _ = self.apply_interpolation_matrix()  # Discard the other two outputs
        # Compute RA and RR
        RA = AX.reshape(self.N * self.M, self.K).sum(axis=1).reshape(self.N, self.M)
        RR = RA - self.MM
        # Compute gradient `GraX`
        self.GraX = self.apply_transformation_matrix(RR).toarray()  # toarray equivalent of full, make non-sparse
        self.curGraX = self.GraX.copy()
        # Compute initial step size `L0`
        L0 = np.linalg.eigvalsh(self.Y.T @ self.Y).max() * np.max([self.A.max(), 1 / self.A.min()])
        # Compute adaptive step size `L`
        if self.outiter == 1 and self.iter == 1:
            L = L0
        else:
            num = np.sum((self.GraX - self.preGraX) * (self.curX - self.preX))  # Element-wise multiplication
            denom = np.linalg.norm(self.curX - self.preX, "fro") ** 2  # Frobenius norm squared
            L = num / denom if denom > 0 else L0  # Ensure L0 fallback

        L = np.maximum(L, L0)  # ensure L is positive
        while True:  # iterate updating X
            X_ = self.curX - self.GraX / L
            # Solve x^3 + p*x + q = 0 for the largest real root
            # off the shelf solver did not work element-wise for matrices
            X = np.square(rooth(-X_, self.eta / (2 * L)))
            # Mask values that should be set to zero
            mask = (X**2 * L / 2 - L * X * X_ + self.eta * np.sqrt(X)) < 0
            X[mask] = 0
            # Check if objective function improves
            if (
                self.objective_history[-1]
                - self.get_objective_function(self.get_residual_matrix(np.maximum(0, X), self.Y, self.A), self.A)
                > 0
            ):
                break
            # Increase L
            L *= 2
            if np.isinf(L):
                break

        # Update `self.curX`
        self.curX = X

    def updateY2(self):
        """
        Updates Y using matrix operations, solving a quadratic program via `solve_mkr_box`.
        """
        K, M = self.Y.shape
        N = self.X.shape[0]

        for m in range(M):
            T = np.zeros((N, K))  # Initialize T as an (N, K) zero matrix

            # Populate T using apply_interpolation
            for k in range(K):
                T[:, k] = self.apply_interpolation(self.A[k, m], self.X[:, k])[0].flatten()

            # Solve quadratic problem for y using solve_mkr_box
            y = self.solve_mkr_box(T, m)

            # Update Y
            self.Y[:, m] = y

    def regularize_function(self, A=None):
        """
        Computes the regularization function, gradient, and Hessian for optimization.
        Returns:
        - fun: Objective function value (scalar)
        - gra: Gradient (same shape as A)
        - hess: Hessian matrix (M*K, M*K)
        """

        # Use provided A or default to self.A
        if A is None:
            A = self.A
        K, M = A.shape
        N = self.X.shape[0]

        # Compute interpolated matrices
        AX, TX, HX = self.apply_interpolation_matrix(A)

        # Compute residual
        RA = np.reshape(np.sum(np.reshape(AX, (N * M, K)), axis=1), (N, M)) - self.MM

        # Compute objective function
        fun = self.get_objective_function(RA, A)

        # Compute gradient (removed index filtering)
        gra = (
            np.reshape(np.sum(TX * np.tile(RA, (1, K)), axis=0), (M, K)).T + self.rho * A @ self.P.T @ self.P
        )  # Gradient matrix

        # Compute Hessian (removed index filtering)
        hess = np.zeros((M * K, M * K))

        for m in range(M):
            Tx = TX[:, m + M * np.arange(K)]  # Now using all rows
            hess[m * K : (m + 1) * K, m * K : (m + 1) * K] = Tx.T @ Tx

        hess = (
            hess
            + spdiags(
                np.reshape(
                    np.reshape(np.sum(HX * np.tile(RA, (1, K)), axis=0), (M, K)).T,
                    (M * K,),  # ✅ Ensure 1D instead of (M*K,1)
                ),
                0,  # Diagonal index
                M * K,  # Number of rows
                M * K,  # Number of columns
            ).toarray()
            + self.rho * self.PPPP
        )

        return fun, gra, hess

    def updateA2(self):
        """
        Updates matrix A using constrained optimization (equivalent to fmincon in MATLAB).
        """

        # Flatten A for compatibility with the optimizer (since SciPy expects 1D input)
        A_initial = self.A.flatten()

        # Define the optimization function
        def objective(A_vec):
            A_matrix = A_vec.reshape(self.A.shape)  # Reshape back to matrix form
            return self.regularize_function(A_matrix)

        # Optimization constraints: lower bound 0.1, no upper bound
        bounds = [(0.1, None)] * A_initial.size  # Equivalent to 0.1 * ones(K, M)

        # Solve optimization problem (equivalent to fmincon)
        result = minimize(
            fun=lambda A_vec: objective(A_vec)[0],  # Objective function
            x0=A_initial,  # Initial guess
            method="trust-constr",  # Equivalent to 'trust-region-reflective'
            jac=lambda A_vec: objective(A_vec)[1].flatten(),  # Gradient
            hess=lambda A_vec: objective(A_vec)[2],  # Hessian
            bounds=bounds,  # Lower bounds on A
        )

        # Update A with the optimized values
        self.A = result.x.reshape(self.A.shape)


def rooth(p, q):
    """
    Solves x^3 + p*x + q = 0 element-wise for matrices, returning the largest real root.
    """
    # Handle special case where q == 0
    y = np.where(q == 0, np.maximum(0, -p) ** 0.5, np.zeros_like(p))  # q=0 case

    # Compute discriminant
    delta = (q / 2) ** 2 + (p / 3) ** 3
    d = np.sqrt(delta)

    # Compute cube roots
    a1 = (-q / 2 + d) ** (1 / 3)
    a2 = (-q / 2 - d) ** (1 / 3)

    # Compute cube roots of unity
    w = (np.sqrt(3) * 1j - 1) / 2

    # Compute the three possible roots (element-wise)
    y1 = a1 + a2
    y2 = w * a1 + w**2 * a2
    y3 = w**2 * a1 + w * a2

    # Take the largest real root element-wise
    real_roots = np.stack([np.real(y1), np.real(y2), np.real(y3)], axis=0)
    y = np.max(real_roots, axis=0) * (delta < 0)  # Keep only real roots when delta < 0

    return y
