import jax
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal, normal, zeros, constant, xavier_normal
from flax import linen as nn
from jax.tree_util import tree_map
from jax import lax
import optax
from flax.training import train_state
from typing import Callable, Sequence
from functools import partial
import math
from typing import Dict, Any

activation_map = {
    "relu": nn.relu,
    "gelu": nn.gelu,
    "sigmoid": nn.sigmoid,
    "swish": nn.swish,
    "tanh": jnp.tanh,
    "sin": jnp.sin,
}

def _get_activation(key):
    if key in activation_map:
        return activation_map[key]

    else:
        raise NotImplementedError(f"Activation {key} not supported yet!")


class MLP(nn.Module):
    net_config: dict
    kernel_init: Any = glorot_normal()
    bias_init:   Any = zeros

    def setup(self):
        act_name = self.net_config.get("activation", "tanh")
        self.activation_fn = _get_activation(act_name)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        layers = self.net_config["layers"]
        for i, out_dim in enumerate(layers[1:]):
            x = nn.Dense(features=out_dim,
                         kernel_init=self.kernel_init,
                         bias_init=self.bias_init)(x)
            if i < len(layers) - 2:
                x = self.activation_fn(x)
        return x


class TrainState(train_state.TrainState):
    weights: Any
    momentum: float

    def apply_weights(self, new_weights: Any, **kwargs) -> "TrainState":
        """
        Update `self.weights` by exponential moving average against `new_weights`.
        """
        ra = lambda old, new: old * self.momentum + (1.0 - self.momentum) * new
        averaged = tree_map(ra, self.weights, new_weights)
        averaged = lax.stop_gradient(averaged)

        return self.replace(weights=averaged, **kwargs)

class PINN:
    def __init__(self,
                 net_config: dict,
                 learning_rate: float,
                 momentum: float,
                 key: jax.random.PRNGKey):
        # save config
        self.net_config = net_config

        # instantiate the MLP
        self.model = MLP(net_config=net_config)

        # dummy input to infer shape
        in_dim = net_config["layers"][0]
        dummy_x = jnp.zeros((1, in_dim))

        # initialize network parameters
        params = self.model.init(key, dummy_x)

        # set up optimizer and TrainState
        tx = optax.adam(learning_rate)
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=tx,
            weights=params,
            momentum=momentum
        )

    def u_net(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass for predicted field(s)."""
        return self.state.apply_fn(self.state.params, x)

    def r_net(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute PDE residual; here as example u''(x)."""
        grad1 = jax.grad(lambda xx: jnp.sum(self.u_net(xx)))
        grad2 = jax.grad(lambda xx: jnp.sum(grad1(xx)))
        return grad2(x)

    def losses(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """Separate data and physics losses."""
        u_pred = self.u_net(batch["x_u"])
        r_pred = self.r_net(batch["x_f"])
        data_loss = jnp.mean((u_pred - batch["u"])**2)
        phys_loss = jnp.mean(r_pred**2)
        return {"data": data_loss, "phys": phys_loss}

    @partial(jax.jit, static_argnums=0)
    def loss(self, params: Any, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Total loss combining data and physics components."""
        # temporarily swap in `params` for computing loss
        orig_params = self.state.params
        self.state = self.state.replace(params=params)
        loss_dict = self.losses(batch)
        # restore
        self.state = self.state.replace(params=orig_params)
        return loss_dict["data"] + loss_dict["phys"]

    # @jax.jit
    @partial(jax.jit, static_argnums=0)
    def train_step(self, state: TrainState, batch: Dict[str, jnp.ndarray]) -> TrainState:
        """One training step: update params and EMA weights."""
        grads = jax.grad(self.loss)(state.params, batch)
        new_state = state.apply_gradients(grads=grads)
        new_state = new_state.apply_weights(new_state.params)
        return new_state

    def step(self, batch: Dict[str, jnp.ndarray]) -> None:
        """Public method to run a training step and update internal state."""
        self.state = self.train_step(self.state, batch)

    def train(self, batch: Dict[str, jnp.ndarray]) -> None:
        """
        Run training for `n_iter` iterations as specified in net_config.
        """
        n_iter = self.net_config.get("n_iter")
        if n_iter is None:
            raise ValueError("`n_iter` must be specified in net_config for training.")
        for i in range(n_iter):
            self.step(batch)
            if i % 1000 == 0:
                # compute total loss
                total_loss = self.loss(self.state.params, batch)
                print(f"Iter {i:5d} → Loss = {total_loss:.3e}")
