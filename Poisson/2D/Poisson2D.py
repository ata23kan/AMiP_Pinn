import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt

from model import PINN

# -----------------------------
# Exact solution and forcing for 2D Poisson
# -----------------------------
# u(x,y) = sin(pi x) * sin(pi y)
# Delta u = -2*pi^2 * sin(pi x)*sin(pi y)

def u_exact(xy: jnp.ndarray) -> jnp.ndarray:
    # xy: (N,2)
    return jnp.sin(jnp.pi * xy[:,0]) * jnp.sin(jnp.pi * xy[:,1])

def f_rhs(xy: jnp.ndarray) -> jnp.ndarray:
    return -2.0 * (jnp.pi**2) * jnp.sin(jnp.pi * xy[:,0]) * jnp.sin(jnp.pi * xy[:,1])

# -----------------------------
# Net configuration
# -----------------------------
net_config = {
    'layers':     [2, 128, 128, 1],
    'activation': 'tanh',
    'n_iter':     10000,
    'f_rhs':      f_rhs
}
learning_rate = 1e-3
momentum = 0.99
key = random.PRNGKey(0)

# -----------------------------
# Instantiate 2D PINN
# -----------------------------
pinn = PINN(net_config, learning_rate, momentum, key)

# -----------------------------
# Generate training batch
# -----------------------------
# Boundary: grid on edges of [0,1]^2
n_b = 50
x = jnp.linspace(0,1,n_b)
# four edges
edges = jnp.vstack([
    jnp.stack([x, jnp.zeros_like(x)], axis=1),
    jnp.stack([x, jnp.ones_like(x)], axis=1),
    jnp.stack([jnp.zeros_like(x), x], axis=1),
    jnp.stack([jnp.ones_like(x), x], axis=1),
])

x_u = edges  # shape (4*n_b,2)
u_b = u_exact(x_u)

# Collocation interior points
n_f = 5000
key, subkey = random.split(key)
x_f = random.uniform(subkey, (n_f,2), minval=0.0, maxval=1.0)

batch = {'x_u': x_u, 'u': u_b, 'x_f': x_f}

# -----------------------------
# Train
# -----------------------------
print("Starting training 2D Poisson PINN...")
pinn.train(batch)

# -----------------------------
# Evaluate on grid
# -----------------------------
n_test = 50
xt = np.linspace(0,1,n_test)
yt = np.linspace(0,1,n_test)
X, Y = np.meshgrid(xt, yt)
xy_test = jnp.stack([X.ravel(), Y.ravel()], axis=1)

u_pred = pinn.u_net(xy_test).reshape(n_test, n_test)
u_true = u_exact(xy_test).reshape(n_test,n_test)

# Relative L2 error
rel_l2 = jnp.linalg.norm(u_pred - u_true) / jnp.linalg.norm(u_true)
print(f"Relative L2 error: {rel_l2:.3e}")

# -----------------------------
# Plot full fields side by side
# -----------------------------
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
# Exact Solution
im0 = axs[0].imshow(
    np.array(u_true), extent=[0,1,0,1], origin='lower', aspect='auto'
)
axs[0].set_title('Exact Solution')
axs[0].set_xlabel('x'); axs[0].set_ylabel('y')
plt.colorbar(im0, ax=axs[0])

# PINN Prediction
im1 = axs[1].imshow(
    np.array(u_pred), extent=[0,1,0,1], origin='lower', aspect='auto'
)
axs[1].set_title('PINN Prediction')
axs[1].set_xlabel('x'); axs[1].set_ylabel('y')
plt.colorbar(im1, ax=axs[1])

# Pointwise Absolute Error
error = np.abs(np.array(u_pred) - np.array(u_true))
im2 = axs[2].imshow(
    error, extent=[0,1,0,1], origin='lower', aspect='auto'
)
axs[2].set_title('Absolute Error')
axs[2].set_xlabel('x'); axs[2].set_ylabel('y')
plt.colorbar(im2, ax=axs[2])

plt.suptitle(f"2D Poisson PINN Fields (Rel L2 error: {rel_l2:.2e})")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
