import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map
import flax.linen as nn
from flax.linen.initializers import xavier_uniform, zeros
from flax.training import train_state
import optax
from functools import partial
from typing import Any, Dict, Callable

# ----------------------------------------------------------------------------
# Activation mapping and helper
# ----------------------------------------------------------------------------
activation_map: Dict[str, Callable[[jnp.ndarray], jnp.ndarray]] = {
    "relu":    nn.relu,
    "gelu":    nn.gelu,
    "sigmoid": nn.sigmoid,
    "swish":   nn.swish,
    "tanh":    jnp.tanh,
    "sin":     jnp.sin,
}

def _get_activation(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    key = name.lower()
    if key in activation_map:
        return activation_map[key]
    else:
        raise NotImplementedError(f"Activation '{name}' not supported yet!")

# ----------------------------------------------------------------------------
# Configurable MLP backbone using flax.linen
# ----------------------------------------------------------------------------
class MLP(nn.Module):
    net_config: Dict[str, Any]
    kernel_init: Any = xavier_uniform()
    bias_init:   Any = zeros

    def setup(self):
        act_name = self.net_config.get("activation", "tanh")
        self.activation_fn = _get_activation(act_name)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        layers = self.net_config["layers"]
        for i, out_dim in enumerate(layers[1:]):
            x = nn.Dense(
                features=out_dim,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init
            )(x)
            if i < len(layers) - 2:
                x = self.activation_fn(x)
        return x

# ----------------------------------------------------------------------------
# Custom TrainState with exponential moving-average weights
# ----------------------------------------------------------------------------
class TrainState(train_state.TrainState):
    weights: Any      # EMA of model parameters
    momentum: float

    def apply_weights(self, new_weights: Any, **kwargs) -> "TrainState":
        ra = lambda old, new: old * self.momentum + (1.0 - self.momentum) * new
        averaged = tree_map(ra, self.weights, new_weights)
        averaged = lax.stop_gradient(averaged)
        return self.replace(weights=averaged, **kwargs)

# ----------------------------------------------------------------------------
# PINN wrapper class for Poisson problems
# ----------------------------------------------------------------------------
class PINN:
    def __init__(
        self,
        net_config: Dict[str, Any],
        learning_rate: float,
        momentum: float,
        key: jax.random.PRNGKey
    ):
        """
        net_config: {
          'layers': [in_dim, ..., out_dim],
          'activation': 'tanh',
          'n_iter': int,
          optionally 'f_rhs': Callable[[jnp.ndarray], jnp.ndarray]
        }
        """
        self.net_config = net_config
        self.model = MLP(net_config=net_config)
        self.f_rhs = net_config.get('f_rhs', lambda x: 0.0)

        in_dim = net_config['layers'][0]
        dummy_x = jnp.zeros((1, in_dim))
        params = self.model.init(key, dummy_x)

        tx = optax.adam(learning_rate)
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=tx,
            weights=params,
            momentum=momentum
        )

    def u_net(self, x: jnp.ndarray) -> jnp.ndarray:
        """Network prediction."""
        return self.state.apply_fn(self.state.params, x)

    def r_net_1d(self, x: jnp.ndarray) -> jnp.ndarray:
        """1D Poisson residual u''(x) - f(x)."""
        grad1 = jax.grad(lambda xx: jnp.sum(self.u_net(xx)))
        grad2 = jax.grad(lambda xx: jnp.sum(grad1(xx)))
        return grad2(x) - self.f_rhs(x)

    def r_net_2d(self, x: jnp.ndarray) -> jnp.ndarray:
        """2D Poisson residual (u_xx + u_yy) - f(x,y)."""
        # u(x) expects shape (N,2)
        def u_single(xi):
            # xi shape (2,), return scalar
            return self.u_net(xi[None, :])[0, 0]

        # Hessian of u_single: [[u_xx, u_xy], [u_yx, u_yy]]
        hess_fn = jax.hessian(u_single)
        # compute for each point
        hessians = jax.vmap(hess_fn)(x)
        laplace = hessians[:, 0, 0] + hessians[:, 1, 1]
        return laplace - self.f_rhs(x)

    def losses(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """Compute data & physics losses based on input dim."""
        x_u = batch['x_u']; u_true = batch['u']; x_f = batch['x_f']
        u_pred = self.u_net(x_u)
        # choose residual based on input dimension
        if x_f.shape[1] == 1:
            r_pred = self.r_net_1d(x_f)
        else:
            r_pred = self.r_net_2d(x_f)
        data_loss = jnp.mean((u_pred - u_true) ** 2)
        phys_loss = jnp.mean(r_pred ** 2)
        return {'data': data_loss, 'phys': phys_loss}

    @partial(jax.jit, static_argnums=0)
    def train_step(self, state: TrainState, batch: Dict[str, jnp.ndarray]) -> TrainState:
        """One training step."""
        def loss_fn(params):
            # temporarily swap params
            orig = state.params
            s = state.replace(params=params)
            self.state = s
            losses = self.losses(batch)
            # restore
            self.state = state
            return losses['data'] + losses['phys']

        grads = jax.grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)
        new_state = new_state.apply_weights(new_state.params)
        return new_state

    def step(self, batch: Dict[str, jnp.ndarray]) -> None:
        """Update internal state by one training iteration."""
        self.state = self.train_step(self.state, batch)

    def train(self, batch: Dict[str, jnp.ndarray]) -> None:
        """Run training for `n_iter` iterations."""
        n_iter = self.net_config.get('n_iter')
        if n_iter is None:
            raise ValueError("`n_iter` must be in net_config to train")
        for i in range(n_iter):
            self.step(batch)
            if i % 1000 == 0:
                # compute total loss
                losses = self.losses(batch)
                print(f"Iter {i}: data_loss={losses['data']:.3e}, phys_loss={losses['phys']:.3e}")
