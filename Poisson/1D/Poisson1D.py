# driver.py

import jax
import jax.numpy as jnp
from jax import random
import numpy as onp
import matplotlib.pyplot as plt

from model import PINN  # your model.py created above

# 1) Define exact solution and forcing
def u_exact(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sin(jnp.pi * x)

def f_rhs(x: jnp.ndarray) -> jnp.ndarray:
    # u'' = -π² sin(π x)
    return - (jnp.pi**2) * jnp.sin(jnp.pi * x)

# 2) Subclass to override r_net → u''(x) - f(x)
class Poisson1D_PINN(PINN):
    def r_net(self, x: jnp.ndarray) -> jnp.ndarray:
        # original r_net returns u''(x)
        u_dd = super().r_net(x)
        return u_dd - f_rhs(x)

# 3) Problem setup
key = random.PRNGKey(0)
net_config = {
    "layers":     [1, 64, 64, 1],   # 1D input → two 64-wide hiddens → 1 output
    "activation": "tanh",
    "n_iter": 5000
}
pinn = Poisson1D_PINN(
    net_config=net_config,
    learning_rate=1e-3,
    momentum=0.99,
    key=key
)

# Boundary (Dirichlet) data
x_u = jnp.array([[0.0], [1.0]])
u_b = jnp.array([[0.0], [0.0]])

# Collocation points in interior
key, subkey = random.split(key)
x_f = random.uniform(subkey, (200,1), minval=0.0, maxval=1.0)

batch = {
    "x_u": x_u,
    "u":   u_b,
    "x_f": x_f
}

# 4) Training loop
pinn.train(batch)


# 5) Evaluation on a dense grid
x_test = jnp.linspace(0, 1, 200).reshape(-1,1)
u_pred = pinn.u_net(x_test)
u_true = u_exact(x_test)

# compute relative L2 error
rel_l2 = jnp.linalg.norm(u_pred - u_true) / jnp.linalg.norm(u_true)
print(f"\nRelative L2 error on [0,1]: {rel_l2:.3e}")

# plot
import matplotlib.pyplot as plt
x_np    = onp.squeeze(onp.array(x_test))
u_p_np  = onp.squeeze(onp.array(u_pred))
u_t_np  = onp.squeeze(onp.array(u_true))

plt.figure(figsize=(6,4))
plt.plot(x_np, u_t_np, label="Exact", lw=2)
plt.plot(x_np, u_p_np, "--", label="PINN")
plt.legend()
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title(f"1D Poisson PINN (L2 error: {rel_l2:.2e})")
plt.tight_layout()
plt.show()
