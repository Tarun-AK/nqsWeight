from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
from flax.linen.dtypes import promote_dtype
from jax import lax
from jax.nn.initializers import lecun_normal, zeros
from netket.models.autoreg import ARNNSequential
from netket.utils.types import Array, DType, NNInitFunc

default_kernel_init = lecun_normal()


class MaskedConv2D(nn.Module):
    """2D convolution module with mask for autoregressive NN."""

    features: int
    kernel_size: tuple[int, int]
    kernel_dilation: tuple[int, int]
    exclusive: bool
    feature_group_count: int = 1
    use_bias: bool = True
    param_dtype: DType = jnp.float64
    precision: Any = None
    kernel_init: NNInitFunc = default_kernel_init
    bias_init: NNInitFunc = zeros

    def setup(self):
        kernel_h, kernel_w = self.kernel_size
        mask = np.ones((kernel_h, kernel_w, 1, 1), dtype=self.param_dtype)
        mask[-1, kernel_w // 2 + (not self.exclusive) :] = 0
        self.mask = jnp.asarray(mask)

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        kernel_h, kernel_w = self.kernel_size
        dilation_h, dilation_w = self.kernel_dilation
        ones = (1, 1)

        if inputs.ndim == 3:
            is_single_input = True
            inputs = jnp.expand_dims(inputs, axis=0)
        else:
            is_single_input = False

        in_features = inputs.shape[-1]
        assert in_features % self.feature_group_count == 0
        kernel_shape = self.kernel_size + (
            in_features // self.feature_group_count,
            self.features,
        )

        mask = self.mask
        kernel = self.param(
            "kernel",
            wrap_kernel_init(self.kernel_init, mask),
            kernel_shape,
            self.param_dtype,
        )

        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None

        inputs, mask, kernel, bias = promote_dtype(
            inputs, mask, kernel, bias, dtype=None
        )

        y = jnp.pad(
            inputs,
            (
                (0, 0),
                ((kernel_h - 1) * dilation_h, 0),
                (kernel_w // 2 * dilation_w, (kernel_w - 1) // 2 * dilation_w),
                (0, 0),
            ),
        )

        dimension_numbers = _conv_dimension_numbers(inputs.shape)
        y = lax.conv_general_dilated(
            y,
            mask * kernel,
            window_strides=ones,
            padding="VALID",
            lhs_dilation=ones,
            rhs_dilation=self.kernel_dilation,
            dimension_numbers=dimension_numbers,
            feature_group_count=self.feature_group_count,
            precision=self.precision,
        )

        if is_single_input:
            y = y.squeeze(axis=0)

        if self.use_bias:
            y = y + bias

        return y


def wrap_kernel_init(kernel_init, mask):
    """Correction to LeCun normal init."""
    corr = jnp.sqrt(mask.size / mask.sum())

    def wrapped_kernel_init(*args):
        return corr * mask * kernel_init(*args)

    return wrapped_kernel_init


def _conv_dimension_numbers(input_shape):
    """Computes the dimension numbers based on the input shape."""
    ndim = len(input_shape)
    lhs_spec = (0, ndim - 1, *tuple(range(1, ndim - 1)))
    rhs_spec = (ndim - 1, ndim - 2, *tuple(range(0, ndim - 2)))
    out_spec = lhs_spec
    return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


def _normalize(log_psi, machine_pow: int):
    """Normalizes log_psi to have L2-norm 1 along the last axis."""
    return log_psi - 1 / machine_pow * jax.scipy.special.logsumexp(
        machine_pow * log_psi.real, axis=-1, keepdims=True
    )


class ResidualBlock(nn.Module):
    """Residual block for masked autoregressive CNN."""

    features: int
    kernel_size: tuple[int, int]
    dropout_rate: float = 0.0
    use_dropout: bool = False
    param_dtype: DType = jnp.float64

    @nn.compact
    def __call__(self, x, training: bool = False):
        residual = x

        # First conv + activation
        y = MaskedConv2D(
            features=self.features,
            kernel_size=self.kernel_size,
            kernel_dilation=(1, 1),
            exclusive=False,
            param_dtype=self.param_dtype,
        )(x)
        y = nn.relu(y)

        # Optional dropout on residual path
        if self.use_dropout and self.dropout_rate > 0:
            y = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(y)

        # Second conv (no activation yet)
        y = MaskedConv2D(
            features=self.features,
            kernel_size=self.kernel_size,
            kernel_dilation=(1, 1),
            exclusive=False,
            param_dtype=self.param_dtype,
        )(y)

        # Projection if channel dimensions don't match
        if residual.shape[-1] != self.features:
            residual = MaskedConv2D(
                features=self.features,
                kernel_size=(1, 1),
                kernel_dilation=(1, 1),
                exclusive=False,
                param_dtype=self.param_dtype,
            )(residual)

        # Add residual and apply activation
        return nn.relu(residual + y)


class ResidualConv2D(ARNNSequential):
    """State-of-the-art Residual Masked CNN for autoregressive modeling.

    Recommended hyperparameters for spin lattices:
    - layers: 6-10 (use 8 as default)
    - features: 64-128 (use 64 for ~50k params, 96 for ~100k, 128 for ~200k)
    - kernel_size: (3, 5) for better receptive field
    - dropout_rate: 0.5 if use_dropout=True

    Example for spin-3/2 on 8x8 lattice with ~100k params:
        model = ResidualConv2D(
            layers=8,
            features=96,
            kernel_size=(3, 5),
            hilbert=hilbert,
            L=8,
            W=8,
            channels=1,
            use_dropout=False,  # Set to True if overfitting
            dropout_rate=0.5
        )
    """

    layers: int
    features: int
    kernel_size: tuple[int, int]
    hilbert: any
    L: int
    W: int
    channels: int
    use_dropout: bool = False
    dropout_rate: float = 0.5
    param_dtype: DType = jnp.float64
    machine_pow: int = 2

    @nn.compact
    def conditionals_log_psi(self, x, training: bool = True):
        x = self.reshape_inputs(x)

        # Initial projection to feature dimension
        x = MaskedConv2D(
            features=self.features,
            kernel_size=self.kernel_size,
            kernel_dilation=(1, 1),
            exclusive=False,
            param_dtype=self.param_dtype,
        )(x)
        x = nn.relu(x)

        # Stack of residual blocks
        for _ in range(self.layers):
            x = ResidualBlock(
                features=self.features,
                kernel_size=self.kernel_size,
                dropout_rate=self.dropout_rate,
                use_dropout=self.use_dropout,
                param_dtype=self.param_dtype,
            )(x, training=training)

        # Final output projection (exclusive mask for autoregressive property)
        x = MaskedConv2D(
            features=self.hilbert.local_size,
            kernel_size=self.kernel_size,
            kernel_dilation=(1, 1),
            exclusive=True,
            param_dtype=self.param_dtype,
        )(x)

        x = x.reshape((x.shape[0], -1, x.shape[-1]))
        log_psi = _normalize(x, 2)
        return log_psi

    def reshape_inputs(self, inputs):
        return inputs.reshape((inputs.shape[0], self.L, self.W, self.channels))


if __name__ == "__main__":
    from tqdm import trange

    L = 3
    hilbert = nk.hilbert.Spin(s=3 / 2, N=L**2)  # spin-3/2 system

    # Recommended configuration for ~100k parameters
    model = ResidualConv2D(
        layers=4,  # 8 residual blocks
        features=4,  # 64 feature maps (adjust to 96 or 128 for more capacity)
        kernel_size=(3, 5),  # (height, width) - asymmetric for better receptive field
        hilbert=hilbert,
        L=L,
        W=L,
        channels=1,
        use_dropout=False,  # Enable if you see overfitting
        dropout_rate=0.5,  # Standard rate when dropout is enabled
    )

    key = jax.random.PRNGKey(0)
    x = jnp.ones((1, L * L))
    params = model.init(key, x)

    # Count parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Total parameters: {param_count:,}")

    # Test forward pass
    for i in trange(100):
        y = model.apply(params, hilbert.all_states())
    print(f"Normalization check: {jnp.sum(jnp.exp(y) ** 2)}")
