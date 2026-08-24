"""Progress reporting for `reconstruction_jax.optimize`."""
import time

import numpy as np


class Tracker:
    """Prints the loss every step and, optionally, records a point-cloud
    snapshot per step for `export_html` (see `viz.export_points_html`)."""

    def __init__(self, print_every=1, record_frames=False):
        self.print_every = print_every
        self.record_frames = record_frames
        self.frames = []
        self.losses = []
        self._t0 = None

    def start(self):
        self._t0 = time.time()

    def step(self, i, loss_value, points):
        # Mathematically loss >= 0 always (it's a squared distance); the raw
        # float can read slightly negative near convergence from computing it
        # as a difference of two close, large-ish quantities (worse on GPU,
        # whose reduction order differs from CPU's). Only the DISPLAYED/
        # recorded value is clamped here, after value_and_grad/optax already
        # used the real one -- can't perturb the optimization trajectory the
        # way changing the loss formula itself did (see conversation).
        loss_value = max(0.0, float(loss_value))
        self.losses.append(loss_value)
        if self.record_frames:
            self.frames.append(np.asarray(points))
        if self.print_every and i % self.print_every == 0:
            print(f"[{i:5d}] loss={loss_value:.6e}  ({time.time() - self._t0:.3e}s)")

    def export_html(self, out_path, extent, **kwargs):
        from .viz import export_points_html
        export_points_html(self.frames, extent, out_path, **kwargs)
