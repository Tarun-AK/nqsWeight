from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import netket as nk
from flax.linen.dtypes import promote_dtype
from jax import lax
from jax.nn.initializers import lecun_normal, zeros
from netket.models.autoreg import ARNNSequential
from netket.utils.types import Array, DType, NNInitFunc

default_kernel_init = lecun_normal()


def wrap_kernel_init(kernel_init, mask):
    """Correction to LeCun normal init."""

    corr = jnp.sqrt(mask.size / mask.sum())

    def wrapped_kernel_init(*args):
        return corr * mask * kernel_init(*args)

    return wrapped_kernel_init


# This is copy-pasted from flax.linen.linear in order to vendor it
def _conv_dimension_numbers(input_shape):
    """Computes the dimension numbers based on the input shape."""
    ndim = len(input_shape)
    lhs_spec = (0, ndim - 1, *tuple(range(1, ndim - 1)))
    rhs_spec = (ndim - 1, ndim - 2, *tuple(range(0, ndim - 2)))
    out_spec = lhs_spec
    return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


def _normalize(log_psi, machine_pow: int):
    """
    Normalizes log_psi to have L2-norm 1 along the last axis.
    """
    return log_psi - 1 / machine_pow * jax.scipy.special.logsumexp(
        machine_pow * log_psi.real, axis=-1, keepdims=True
    )


class PeriodicMaskedConv2D(nn.Module):
    """
    2D Masked Convolution with Horizontal Periodicity (Cylindrical Topology).

    - Vertical: Causal Zero Padding (Top-padded only).
    - Horizontal (All Rows): Periodic Padding (Wrap) with Causal Masking.
    """

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
        kh, kw = self.kernel_size

        # Create AR mask for cylindrical topology
        mask = jnp.ones((kh, kw, 1, 1), dtype=self.param_dtype)

        # For the current row (last row of kernel):
        # - If exclusive=True: mask center pixel and everything to the right
        # - If exclusive=False: mask only pixels to the right of center
        center_idx = kw // 2
        if self.exclusive:
            mask = mask.at[-1, center_idx:].set(0)
        else:
            mask = mask.at[-1, center_idx + 1 :].set(0)

        self.mask = mask

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        kernel_h, kernel_w = self.kernel_size
        dilation_h, dilation_w = self.kernel_dilation
        ones = (1, 1)

        if inputs.ndim == 3:
            inputs = inputs[None]
            single = True
        else:
            single = False

        in_ch = inputs.shape[-1]
        ks = self.kernel_size + (in_ch // self.feature_group_count, self.features)

        # Initialize parameters
        kernel = self.param(
            "kernel",
            wrap_kernel_init(self.kernel_init, self.mask),
            ks,
            self.param_dtype,
        )
        bias = (
            self.param("bias", self.bias_init, (self.features,), self.param_dtype)
            if self.use_bias
            else None
        )

        inputs, mask, kernel, bias = promote_dtype(
            inputs, self.mask, kernel, bias, dtype=None
        )

        # --- Padding Calculation ---
        # Vertical: Causal (top only)
        pad_top = (kernel_h - 1) * dilation_h
        pad_bottom = 0

        # Horizontal: Periodic wrapping
        pad_left = (kernel_w // 2) * dilation_w
        pad_right = ((kernel_w - 1) // 2) * dilation_w

        # --- Apply Periodic Padding Horizontally ---
        # Wrap around horizontally (cylindrical)
        left_wrap = (
            inputs[:, :, -pad_left:, :]
            if pad_left > 0
            else jnp.zeros(
                (inputs.shape[0], inputs.shape[1], 0, inputs.shape[3]),
                dtype=inputs.dtype,
            )
        )
        right_wrap = (
            inputs[:, :, :pad_right, :]
            if pad_right > 0
            else jnp.zeros(
                (inputs.shape[0], inputs.shape[1], 0, inputs.shape[3]),
                dtype=inputs.dtype,
            )
        )

        inputs_wrapped = jnp.concatenate([left_wrap, inputs, right_wrap], axis=2)

        # --- Apply Causal Padding Vertically ---
        inputs_padded = jnp.pad(
            inputs_wrapped, ((0, 0), (pad_top, pad_bottom), (0, 0), (0, 0))
        )

        # --- Convolve with Masked Kernel ---
        dn = _conv_dimension_numbers(inputs.shape)

        y = lax.conv_general_dilated(
            inputs_padded,
            kernel * mask,
            window_strides=ones,
            padding="VALID",
            lhs_dilation=ones,
            rhs_dilation=self.kernel_dilation,
            dimension_numbers=dn,
            feature_group_count=self.feature_group_count,
            precision=self.precision,
        )

        if single:
            y = y[0]
        if bias is not None:
            y = y + bias
        return y


class Conv2D(ARNNSequential):
    layers: int
    features: int
    kernel_size: tuple[int, int]
    hilbert: any
    L: int
    W: int
    channels: int

    @nn.compact
    def conditionals_log_psi(self, x):
        x = self.reshape_inputs(x)
        for _ in range(self.layers - 1):
            x = PeriodicMaskedConv2D(
                features=self.features,
                kernel_size=self.kernel_size,
                kernel_dilation=(1, 1),
                exclusive=False,
            )(x)
            x = nn.relu(x)
        x = PeriodicMaskedConv2D(
            features=self.hilbert.local_size,
            kernel_size=self.kernel_size,
            kernel_dilation=(1, 1),
            exclusive=True,
        )(x)
        x = x.reshape((x.shape[0], -1, x.shape[-1]))
        log_psi = _normalize(x, 2)
        return log_psi

    def reshape_inputs(self, inputs):
        return inputs.reshape((inputs.shape[0], self.L, self.W, self.channels))


if __name__ == "__main__":
    from tqdm import trange

    L = 4
    hilbert = nk.hilbert.Spin(s=1 / 2, N=L**2)
    for i in trange(1000):
        model = Conv2D(
            layers=3,
            features=10,
            kernel_size=(2, 2),
            hilbert=hilbert,
            L=L,
            W=L,
            channels=1,
        )

        key = jax.random.PRNGKey(0)
        x = jnp.ones((1, L * L))
        params = model.init(key, x)
        y = model.apply(params, hilbert.all_states())
        print(jnp.sum(jnp.exp(y) ** 2))
