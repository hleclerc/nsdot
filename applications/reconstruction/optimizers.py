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
    """L-BFGS optimization via scipy.

    `min_iter` : nombre de pas IMPOSÉ avant que scipy ait le droit de conclure à convergence.
    Sans ça, L-BFGS-B peut s'arrêter dès le pas 0/1 dès que `ftol`/`gtol` (défaut scipy) sont
    satisfaits -- observé sur le modèle DISQUES, où la perte est quasi stationnaire le long de
    directions pourtant loin d'être optimales (un disque peut glisser sans changer le résidu
    tant qu'il ne chevauche personne). Mis en oeuvre en DEUX appels `scipy.optimize.minimize` :
    un premier à `maxiter=min_iter` avec `ftol=gtol=0` (aucun arrêt anticipé possible), puis un
    second qui reprend le `ftol` demandé pour les pas restants.

    `disp_tol` : pendant ce second appel, arrêt anticipé (levée de `StopIteration` depuis le
    callback -- supporté nativement par `scipy.optimize.minimize` depuis la 1.11) dès que le
    déplacement max d'un point entre deux pas passe sous ce seuil (mêmes unités que les points).
    Combine bien avec `min_iter` : "au moins `min_iter` pas, puis tant que ça bouge encore".
    `None` (défaut) désactive ce critère -- seul `ftol` scipy arrête alors la phase 2.
    """

    def __init__(self, max_iter: int = 100, ftol: float = 1e-8,
                 min_iter: int = 0, disp_tol: float | None = None):
        self.max_iter = max_iter
        self.ftol = ftol
        self.min_iter = min_iter
        self.disp_tol = disp_tol

    def minimize(self, scalar_loss, x0, callback=None):
        from sdot import driver

        x0 = x0.copy() if isinstance(x0, np.ndarray) else np.array(x0)
        shape_orig = x0.shape

        # Flatten to 1D for scipy
        x0_flat = x0.reshape(-1)
        # jit ONCE (compiled on the flat 1D signature scipy calls with) and reuse across iterations.
        loss_j = driver.jit(lambda xf: scalar_loss(xf.reshape(shape_orig)))
        grad_j = driver.jit(lambda xf: driver.grad(scalar_loss)(xf.reshape(shape_orig)).reshape(-1))

        step_counter = [0]  # mutable counter for callback, shared across the two minimize() calls
        prev_x = [x0_flat.copy()]

        def loss_flat(x_flat):
            return loss_j(x_flat)

        def grad_flat(x_flat):
            return grad_j(x_flat)

        def make_callback(check_disp: bool):
            def scipy_callback(x_flat):
                step = step_counter[0]
                if callback is not None:
                    callback(step, x_flat.reshape(shape_orig))
                stop = False
                if check_disp and self.disp_tol is not None:
                    disp = float(np.max(np.abs(x_flat - prev_x[0])))
                    stop = disp < self.disp_tol
                prev_x[0] = x_flat.copy()
                step_counter[0] += 1
                if stop:
                    raise StopIteration
            return scipy_callback

        x_flat = x0_flat
        if self.min_iter > 0:
            result = minimize(
                loss_flat, x_flat, jac=grad_flat, method='L-BFGS-B',
                callback=make_callback(check_disp=False),
                options={'maxiter': self.min_iter, 'ftol': 0.0, 'gtol': 0.0},
            )
            x_flat = result.x

        remaining = max(0, self.max_iter - step_counter[0])
        if remaining > 0:
            result = minimize(
                loss_flat, x_flat, jac=grad_flat, method='L-BFGS-B',
                callback=make_callback(check_disp=True),
                options={'maxiter': remaining, 'ftol': self.ftol},
            )
            x_flat = result.x

        return x_flat.reshape(shape_orig)


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
