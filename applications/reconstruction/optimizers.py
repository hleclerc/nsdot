from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize, line_search


class Optimizer(ABC):
    """Base class for optimization algorithms."""

    @abstractmethod
    def minimize(self, scalar_loss, x0, callback=None):
        """Minimize scalar_loss starting from x0.

        Args:
            scalar_loss: function(x) -> float
            x0: initial point
            callback: optional function(step, x) called at each iteration

        Returns:
            optimized x
        """
        pass


class GradientDescent(Optimizer):
    """Simple gradient descent with fixed step size."""

    def __init__(self, lr: float = 0.2, nb_steps: int = 100):
        self.lr = lr
        self.nb_steps = nb_steps

    def minimize(self, scalar_loss, x0, callback=None):
        from sdot import driver

        x = x0.copy() if isinstance(x0, np.ndarray) else np.array(x0)
        # jit ONCE and reuse across steps: without it each step re-traces + re-lowers the whole
        # graph (unbounded RSS growth, ~4x slower). The C++ `driver.call` survives the trace as an
        # FFI primitive; only the surrounding tensor algebra is fused by XLA. See driver.jit.
        grad = driver.jit(driver.grad(scalar_loss))

        for step in range(self.nb_steps):
            x = x - self.lr * grad(x)
            if callback is not None:
                callback(step, x)

        return x


class LBFGS(Optimizer):
    """L-BFGS optimization via scipy."""

    def __init__(self, max_iter: int = 100, ftol: float = 1e-8):
        self.max_iter = max_iter
        self.ftol = ftol

    def minimize(self, scalar_loss, x0, callback=None):
        from sdot import driver

        x0 = x0.copy() if isinstance(x0, np.ndarray) else np.array(x0)
        shape_orig = x0.shape

        # Flatten to 1D for scipy
        x0_flat = x0.reshape(-1)
        # jit ONCE (compiled on the flat 1D signature scipy calls with) and reuse across iterations.
        loss_j = driver.jit(lambda xf: scalar_loss(xf.reshape(shape_orig)))
        grad_j = driver.jit(lambda xf: driver.grad(scalar_loss)(xf.reshape(shape_orig)).reshape(-1))

        step_counter = [0]  # mutable counter for callback

        def loss_flat(x_flat):
            return loss_j(x_flat)

        def grad_flat(x_flat):
            return grad_j(x_flat)

        def scipy_callback(x_flat):
            if callback is not None:
                callback(step_counter[0], x_flat.reshape(shape_orig))
            step_counter[0] += 1

        result = minimize(
            loss_flat,
            x0_flat,
            jac=grad_flat,
            method='L-BFGS-B',
            callback=scipy_callback,
            options={'maxiter': self.max_iter, 'ftol': self.ftol}
        )

        return result.x.reshape(shape_orig)


class GradientDescentLineSearch(Optimizer):
    """Gradient descent with backtracking line search."""

    def __init__(self, lr: float = 1.0, nb_steps: int = 100, c1: float = 1e-4, rho: float = 0.5):
        self.lr = lr
        self.nb_steps = nb_steps
        self.c1 = c1
        self.rho = rho

    def minimize(self, scalar_loss, x0, callback=None):
        from sdot import driver

        x = x0.copy() if isinstance(x0, np.ndarray) else np.array(x0)
        grad = driver.jit(driver.grad(scalar_loss))
        loss_j = driver.jit(scalar_loss)

        for step in range(self.nb_steps):
            g = grad(x)
            loss_curr = loss_j(x)

            # Backtracking line search: find step size that decreases loss
            alpha = self.lr
            max_backtracks = 20
            for _ in range(max_backtracks):
                x_new = x - alpha * g
                loss_new = loss_j(x_new)
                if loss_new <= loss_curr - self.c1 * alpha * np.dot(g.reshape(-1), g.reshape(-1)):
                    break
                alpha *= self.rho
            else:
                x_new = x - alpha * g

            x = x_new
            if callback is not None:
                callback(step, x)

        return x


class Adam(Optimizer):
    """Adam optimizer with gradient clipping and adaptive learning rate."""

    def __init__(self, lr: float = 0.1, nb_steps: int = 200, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, grad_clip: float = None):
        self.lr = lr
        self.nb_steps = nb_steps
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.grad_clip = grad_clip

    def minimize(self, scalar_loss, x0, callback=None):
        from sdot import driver

        x = x0.copy() if isinstance(x0, np.ndarray) else np.array(x0)
        grad = driver.jit(driver.grad(scalar_loss))

        m = np.zeros_like(x)  # first moment
        v = np.zeros_like(x)  # second moment

        for step in range(self.nb_steps):
            g = grad(x)

            # Optional gradient clipping
            if self.grad_clip is not None:
                grad_norm = np.linalg.norm(g)
                if grad_norm > self.grad_clip:
                    g = g * (self.grad_clip / grad_norm)

            # Update biased first moment estimate
            m = self.beta1 * m + (1 - self.beta1) * g
            # Update biased second raw moment estimate
            v = self.beta2 * v + (1 - self.beta2) * (g ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - self.beta1 ** (step + 1))
            # Compute bias-corrected second raw moment estimate
            v_hat = v / (1 - self.beta2 ** (step + 1))

            # Update parameters
            x = x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            if callback is not None:
                callback(step, x)

        return x
