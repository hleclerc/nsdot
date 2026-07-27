"""Test warm-start: verify positions are correctly passed between optimizers."""

import numpy as np
from ..Sinogram import Sinogram
from ..reconstruction import random_positions, loss, reconstruct
from ..optimizers import LBFGS, Adam

# Create test problem
sino = Sinogram(nb_angles=20, nb_bins=20, extent=6.0)
sino.add_disk(center=[0.3, -0.2], radius=1.0)

positions_init = random_positions(100, extent=6.0, seed=42)
loss_init = float(loss(sino, positions_init).tensor)

print(f"Initial setup:")
print(f"  positions_init shape: {positions_init.tensor.shape}")
print(f"  positions_init dtype: {positions_init.tensor.dtype}")
print(f"  initial loss: {loss_init:.8f}")
print(f"  first position: {positions_init.tensor[0]}")

# Phase 1: LBFGS (just a few iterations to make positions change)
print(f"\n=== Phase 1: LBFGS ===")
positions_after_lbfgs = reconstruct(
    sino,
    positions_init,
    optimizer=LBFGS(max_iter=3, ftol=1e-8)
)
loss_after_lbfgs = float(loss(sino, positions_after_lbfgs).tensor)

print(f"After LBFGS:")
print(f"  positions_after_lbfgs shape: {positions_after_lbfgs.tensor.shape}")
print(f"  positions_after_lbfgs dtype: {positions_after_lbfgs.tensor.dtype}")
print(f"  loss: {loss_after_lbfgs:.8f}")
print(f"  first position: {positions_after_lbfgs.tensor[0]}")
print(f"  position change: {np.linalg.norm(positions_after_lbfgs.tensor[0] - positions_init.tensor[0]):.8f}")

# Phase 2: Adam starting from LBFGS result
print(f"\n=== Phase 2: Adam (warm-start) ===")

# Manually check what Adam sees as initial loss
print(f"  Expected initial loss for Adam: {loss_after_lbfgs:.8f}")

callback_losses = []
def track_callback(step, pos):
    l = float(loss(sino, pos).tensor)
    callback_losses.append((step, l))
    if step == -1:
        print(f"  Callback initial loss (step=-1): {l:.8f}")
    elif step == 0:
        print(f"  Callback step 0 loss: {l:.8f}")

positions_after_adam = reconstruct(
    sino,
    positions_after_lbfgs,
    optimizer=Adam(lr=0.1, nb_steps=3, grad_clip=10.0),
    callback=track_callback
)
loss_after_adam = float(loss(sino, positions_after_adam).tensor)

print(f"After Adam:")
print(f"  positions_after_adam shape: {positions_after_adam.tensor.shape}")
print(f"  loss: {loss_after_adam:.8f}")
print(f"  first position: {positions_after_adam.tensor[0]}")

# Diagnostics
print(f"\n=== Diagnostics ===")
print(f"Loss trajectory:")
print(f"  Initial:      {loss_init:.8f}")
print(f"  After LBFGS:  {loss_after_lbfgs:.8f} (reduction: {(loss_init - loss_after_lbfgs):.8f})")
print(f"  After Adam:   {loss_after_adam:.8f} (reduction: {(loss_after_lbfgs - loss_after_adam):.8f})")

if callback_losses:
    print(f"\nCallback losses:")
    for step, l in callback_losses[:5]:
        print(f"  Step {step}: {l:.8f}")

# Check if positions actually changed
pos_diff = np.linalg.norm(positions_after_adam.tensor - positions_after_lbfgs.tensor)
print(f"\nPosition norm difference (Adam vs LBFGS): {pos_diff:.8e}")

if pos_diff < 1e-6 and loss_after_adam >= loss_after_lbfgs:
    print("❌ WARNING: Positions did NOT change in Adam phase!")
else:
    print("✓ Positions changed correctly")
