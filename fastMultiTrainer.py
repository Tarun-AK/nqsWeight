from functools import partial
from typing import Any, Callable, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import lax


def _normalize(log_psi: jnp.ndarray, machine_pow: int) -> jnp.ndarray:
    return log_psi - 1 / machine_pow * jax.scipy.special.logsumexp(
        machine_pow * log_psi.real, axis=-1, keepdims=True
    )


class GroupedMaskedConv2D(nn.Module):
    num_models: int
    features: int
    kernel_size: Tuple[int, int]
    exclusive: bool = False
    use_bias: bool = True
    param_dtype: Any = jnp.float32
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        in_channels = x.shape[-1] // self.num_models
        kernel_shape = (self.num_models, *self.kernel_size, in_channels, self.features)
        kernel = self.param("kernel", self.kernel_init, kernel_shape, self.param_dtype)

        mask = jnp.ones(self.kernel_size)
        ky, kx = self.kernel_size
        mid_y, mid_x = ky // 2, kx // 2
        mask = mask.at[mid_y, mid_x + (1 if self.exclusive else 0) :].set(0)
        mask = mask.at[mid_y + 1 :, :].set(0)
        masked_kernel = kernel * mask[None, :, :, None, None]

        tkernel = jnp.transpose(masked_kernel, (1, 2, 3, 0, 4))
        tkernel = tkernel.reshape(
            (*self.kernel_size, in_channels, self.num_models * self.features)
        )

        dn = lax.conv_dimension_numbers(
            x.shape, tkernel.shape, ("NHWC", "HWIO", "NHWC")
        )

        out = lax.conv_general_dilated(
            x,
            tkernel,
            window_strides=(1, 1),
            padding="SAME",
            feature_group_count=self.num_models,
            dimension_numbers=dn,
        )

        if self.use_bias:
            bias = self.param(
                "bias",
                self.bias_init,
                (self.num_models, self.features),
                self.param_dtype,
            )
            out = out + bias.reshape((1, 1, 1, -1))
        return out


class BatchedARNNConv2D(nn.Module):
    num_models: int
    hilbert: Any
    layers: int
    features: int
    kernel_size: Tuple[int, int]
    machine_pow: int = 2
    param_dtype: Any = jnp.float64
    activation: Callable = jax.nn.selu

    def setup(self):
        self.L = 8
        local_size = self.hilbert.local_size
        self._layers = [
            GroupedMaskedConv2D(
                num_models=self.num_models,
                features=self.features if i < self.layers - 1 else local_size,
                kernel_size=self.kernel_size,
                exclusive=(i == 0),
                param_dtype=self.param_dtype,
                name=f"layer_{i}",
            )
            for i in range(self.layers)
        ]

    def __call__(self, inputs):
        batch_size = inputs.shape[0]
        x = inputs.astype(self.param_dtype)
        x = x.reshape(batch_size, self.num_models, self.L, self.L)
        x = jnp.transpose(x, (0, 2, 3, 1))

        for i, layer in enumerate(self._layers):
            if i > 0:
                x = self.activation(x)
            x = layer(x)

        x = x.reshape(batch_size, self.L * self.L, self.num_models, -1)
        x = jnp.transpose(x, (0, 2, 1, 3))
        log_psi_conditionals = _normalize(x, self.machine_pow)
        local_indices = ((inputs + 3) // 2).astype(jnp.int32)
        log_psi_actual = jnp.take_along_axis(
            log_psi_conditionals, local_indices[:, :, :, None], axis=-1
        ).squeeze(-1)
        return log_psi_actual.sum(axis=-1)


class FastMultiTrainer:
    def __init__(self, model, optimizer, num_models, batch_size):
        self.model = model
        self.optimizer = optimizer
        self.num_models = num_models
        self.batch_size = batch_size
        self.params = None
        self.opt_state = None

    def init(self, sample_input_shape, base_key=0):
        key = jax.random.PRNGKey(base_key)
        dummy = jnp.ones((1, self.num_models, *sample_input_shape))
        variables = self.model.init(key, dummy)
        self.params = variables["params"]
        self.opt_state = self.optimizer.init(self.params)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, params, opt_state, multi_batch):
        def loss_fn(p):
            logits = self.model.apply({"params": p}, multi_batch)
            return -jnp.mean(2 * logits.real)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
        return new_params, new_opt_state, loss

    # --- OPTIMIZED RUN_STEP ---
    def run_step(self, device_datasets, step_i):
        """
        Takes step_i instead of rng to perform high-speed sequential slicing.
        Expects device_datasets shape: [num_models, num_samples, hilbert_size]
        """
        num_samples = device_datasets.shape[1]

        # Calculate start point for slicing
        start = (step_i * self.batch_size) % (num_samples - self.batch_size)

        # Slicing is nearly instantaneous on GPU compared to random.choice
        # Shape: [num_models, batch_size, hilbert_size]
        batch = device_datasets[:, start : start + self.batch_size, :]

        # Transpose to [batch_size, num_models, hilbert_size] for the model
        multi_batch = jnp.transpose(batch, (1, 0, 2))

        self.params, self.opt_state, loss = self.step(
            self.params, self.opt_state, multi_batch
        )
        return loss
