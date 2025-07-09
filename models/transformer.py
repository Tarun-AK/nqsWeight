# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Any, Callable, Tuple

import jax
from flax import linen as nn
from flax.linen.initializers import zeros
from jax import numpy as jnp
from netket.hilbert.homogeneous import HomogeneousHilbert
from netket.models import ARNNSequential
from netket.nn.masked_linear import default_kernel_init
from netket.utils.types import NNInitFunc

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any
DotGeneralT = Callable[..., Array]
ConvGeneralDilatedT = Callable[..., Array]


def get_angles(pos: int, i: int, d_model: int) -> float:
    angle_rates = 1 / jnp.power(10000, (2 * (i // 2)) / jnp.float64(d_model))
    return pos * angle_rates


def positional_encoding(position: int, d_model: int) -> Array:
    """
    Positional encoding according to cosine and sine.

    Args:
        position: Position
        d_model: Model depth

    Returns:

    """
    angle_rads = get_angles(
        jnp.arange(position)[:, jnp.newaxis],
        jnp.arange(d_model)[jnp.newaxis, :],
        d_model,
    )

    # apply sin to even indices in the array; 2i
    sines = jnp.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    cosines = jnp.cos(angle_rads[:, 1::2])

    pos_encoding = jnp.concatenate([sines, cosines], axis=-1)

    # pos_encoding = pos_encoding[jnp.newaxis, ...]

    return pos_encoding.astype(dtype=jnp.float64)


def scaled_dot_product_attention(q: Array, k: Array, v: Array, mask: Array):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = jnp.matmul(
        q, jnp.transpose(k, axes=[0, 2, 1])
    )  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = jnp.float64(k.shape[-1])

    scaled_attention_logits = matmul_qk / jnp.sqrt(dk)

    # add the mask to the scaled tensor.

    if mask is not None:
        # print(scaled_attention_logits.shape,mask.shape)
        scaled_attention_logits += mask * -1e9

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = nn.softmax(
        scaled_attention_logits, axis=-1
    )  # (..., seq_len_q, seq_len_k)

    output = jnp.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output


def split_heads(x, num_heads, depth):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = jnp.reshape(x, (-1, num_heads, depth))
    return jnp.transpose(x, axes=[1, 0, 2])


class MultiHeadAttention(nn.Module):
    """Multi-head attention module"""

    d_model: int
    """hidden layer size"""
    num_heads: int
    """number of attention heads"""

    def setup(self):
        assert (
            self.d_model % self.num_heads == 0
        ), "model dimension must be divisble by the number of attention heads"

        self.depth = self.d_model // self.num_heads

        self.wq = nn.Dense(self.d_model)
        self.wk = nn.Dense(self.d_model)
        self.wv = nn.Dense(self.d_model)

        self.dense = nn.Dense(self.d_model)

    @nn.compact
    def __call__(self, v, k, q, mask):
        q = self.wq(q)  # (seq_len, d_model)
        k = self.wk(k)  # (seq_len, d_model)
        v = self.wv(v)  # (seq_len, d_model)

        q = split_heads(q, self.num_heads, self.depth)  # (num_heads, seq_len_q, depth)
        k = split_heads(k, self.num_heads, self.depth)  # (num_heads, seq_len_k, depth)
        v = split_heads(v, self.num_heads, self.depth)  # (num_heads, seq_len_v, depth)

        scaled_attention = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = jnp.transpose(
            scaled_attention, axes=[1, 0, 2]
        )  # (seq_len_q, num_heads, depth)

        concat_attention = jnp.reshape(
            scaled_attention, (-1, self.d_model)
        )  # (seq_len_q, d_model)

        output = self.dense(concat_attention)  # (seq_len_q, d_model)

        return output


class PointWiseFeedForwardNetwork(nn.Module):
    """Point-wise feed forward neural network"""

    d_model: int
    """hidden layer size"""
    dff: int
    """DFF"""
    activation_fn: None = nn.activation.relu
    """activation function"""

    def setup(self):
        self.dense_1 = nn.Dense(features=self.dff)
        self.dense_2 = nn.Dense(features=self.d_model)

    @nn.compact
    def __call__(self, x):
        x_1 = self.dense_1(x)
        h_1 = self.activation_fn(x_1)
        x_2 = self.dense_2(h_1)
        return x_2


class DecoderLayer(nn.Module):
    d_model: int
    """hidden layer size"""
    dff: int
    """DFF"""
    num_heads: int
    """number of attention heads"""

    def setup(self):
        self.mha = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads)

        self.ffn = PointWiseFeedForwardNetwork(d_model=self.d_model, dff=self.dff)

        self.layernorm1 = nn.LayerNorm(epsilon=1e-6)
        self.layernorm2 = nn.LayerNorm(epsilon=1e-6)

    @nn.compact
    def __call__(self, x, look_ahead_mask):
        attn1 = self.mha(x, x, x, look_ahead_mask)  # (target_seq_len, d_model)

        out1 = attn1 + x
        out1 = self.layernorm1(out1)

        ffn_output = self.ffn(out1)  # (target_seq_len, d_model)

        out2 = ffn_output + out1
        out2 = self.layernorm2(out2)

        return out2


class Decoder(nn.Module):
    """Decoder consisting of the positional embedding and Multi-head attention layers"""

    hilbert: HomogeneousHilbert
    """the Hilbert space. Only homogeneous unconstrained Hilbert spaces are supported."""
    num_layers: int
    """number of attention layers"""
    d_model: int
    """hidden layer size"""
    dff: int
    """DFF"""
    num_heads: int
    """number of attention heads"""

    def setup(self):
        self.embedding = nn.Embed(self.hilbert.local_size, self.d_model)
        self.pos_encoding = positional_encoding(self.hilbert.size, self.d_model)

        self.dec_layers = [
            DecoderLayer(d_model=self.d_model, num_heads=self.num_heads, dff=self.dff)
            for _ in range(self.num_layers)
        ]

    @nn.compact
    def __call__(self, x, look_ahead_mask):
        seq_len = x.shape[0]

        x = self.embedding(x)  # ( target_seq_len, d_model)

        x *= jnp.sqrt(self.d_model)
        x += self.pos_encoding[:seq_len, :]

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, look_ahead_mask)

        return x


class Transformer(ARNNSequential):
    """Transformer Wavefunction (Adapted from Juan Carrasquilla's Tensorflow implementation)"""

    hilbert: HomogeneousHilbert
    """the Hilbert space. Only homogeneous unconstrained Hilbert spaces are supported."""
    autoreg: int
    """Whether the model is autoregressive or not"""
    num_layers: int
    """number of attention layers"""
    d_model: int
    """hidden layer size"""
    dff: int
    """DFF"""
    num_heads: int
    """number of attention heads"""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weights."""
    bias_init: NNInitFunc = zeros
    """initializer for the biases."""

    def setup(self):
        self.L = self.hilbert.size
        self.decoder = Decoder(
            hilbert=self.hilbert,
            num_layers=self.num_layers,
            d_model=self.d_model,
            dff=self.dff,
            num_heads=self.num_heads,
        )
        self.outputdense = nn.Dense(
            features=(self.hilbert.local_size - 1) * 2,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        if self.autoreg:
            self.mask = 1 - jnp.tril(jnp.ones((self.hilbert.size, self.hilbert.size)))
        else:
            self.mask = None

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.hilbert, HomogeneousHilbert):
            raise ValueError(
                f"Only homogeneous Hilbert spaces are supported by ARNN, but hilbert is a {type(self.hilbert)}."
            )

        if self.hilbert.constrained:
            raise ValueError("Only unconstrained Hilbert spaces are supported by ARNN.")

    def conditionals(self, inputs: Array) -> Array:
        """
        Computes the conditional probabilities for each site to take each value.

        Args:
          inputs: configurations with dimensions (batch, Hilbert.size).

        Returns:
          The probabilities with dimensions (batch, Hilbert.size, Hilbert.local_size).

        """
        inputs_dim = inputs.ndim
        # if there is only one batch dimension, expand with a one
        if inputs_dim == 2:
            inputs = jnp.expand_dims(inputs, axis=0)

        inputs = jnp.reshape(inputs, (-1, *inputs.shape[2:]))

        inputs = (inputs + 1) / 2
        inputs = inputs.astype(jnp.int32)

        log_psi = conditionals_log_psi(
            inputs,
            self.mask,
            self.hilbert.local_size,
            self.hilbert.size,
            self.decoder,
            self.outputdense,
        )

        p = jnp.exp(log_psi.real)
        return p

    def conditional(self, inputs: Array, index: int) -> Array:
        """
        Computes the conditional probabilities for one site to take each value.

        It should only be called successively with indices 0, 1, 2, ...,
        as in the autoregressive sampling procedure.

        Args:
          inputs: configurations of partially sampled sites with dimensions (batch, Hilbert.size),
            where the sites that `index` depends on must be already sampled.
          index: index of the site being queried.

        Returns:
          The probabilities with dimensions (batch, Hilbert.local_size).
        """
        return self.conditionals(inputs)[:, index, :]

    @nn.compact
    def __call__(self, inputs: Array) -> Array:

        inputs_dim = inputs.ndim
        # if there is only one batch dimension, expand with a one
        if inputs_dim == 2:
            inputs = jnp.expand_dims(inputs, axis=0)
        batch_shape = list(inputs.shape[:2])
        inputs = jnp.reshape(inputs, (-1, *inputs.shape[2:]))

        inputs = (inputs + 1) / 2
        inputs = inputs.astype(jnp.int32)

        log_psi = conditionals_log_psi(
            inputs,
            self.mask,
            self.hilbert.local_size,
            self.hilbert.size,
            self.decoder,
            self.outputdense,
        )
        one_hot_samples = nn.one_hot(inputs, self.hilbert.local_size, axis=-1)

        log_psi = (log_psi * one_hot_samples).sum(axis=(1, 2))

        if inputs_dim == 2:
            return log_psi
        else:
            return jnp.reshape(log_psi, batch_shape)


def log_coeffs_to_log_psi(logCoeffs: Array, size: int, local_size: int):
    # phase = 1j * jnp.concatenate(
    #     [jnp.zeros((size, 1)), logCoeffs[:, local_size - 1 :]], axis=-1
    # )
    amp = jnp.concatenate(
        [jnp.zeros((size, 1)), logCoeffs[:, : local_size - 1]], axis=-1
    )

    return 0.5 * jax.nn.log_softmax(amp)


@partial(jax.vmap, in_axes=(0, None, None, None, None, None))
def conditionals_log_psi(
    x: Array,
    mask: Array,
    local_size: int,
    size: int,
    decoder: Callable,
    outputdense: Callable,
) -> Array:
    """
    Computes the logarithmic wave function for each site to take each value.

    Args:
      x: configurations with dimensions (Hilbert.size).

    Returns:
      The logarithmic wavefunction with dimensions (Hilbert.size, Hilbert.local_size).

    """
    init = jnp.zeros(1, dtype=jnp.int32)
    output = jnp.concatenate([init, x], axis=0)
    output = output[0:size]
    dec_output = decoder(output, mask)
    output_ampl = outputdense(dec_output)  # (tar_seq_len, target_vocab_size)

    log_psi = log_coeffs_to_log_psi(
        output_ampl + 1e-14, size=size, local_size=local_size
    )

    return log_psi


#
# if __name__ == '__main__':
#     import netket as nk
#
#     L = 2
#     hi = nk.hilbert.Spin(0.5, L)
#     num_layers = 2  # 4
#     d_model = 16  # 128 #128
#     dff = 16  # 128 # 512
#     num_heads = 2  # 8
#     model = Transformer(hilbert=hi,
#                         num_layers=num_layers,
#                         num_heads=num_heads,
#                         d_model=d_model,
#                         dff=dff
#                         )
#     phi = nk.vqs.ExactState(hi, model=model)
#     log_probs = phi.to_array()
#     # print(sum(jnp.abs(log_probs) ** 2))
#
#     sampler = nk.sampler.ARDirectSampler(hi)
#     phi = nk.vqs.MCState(sampler=sampler, model=model, n_samples=10)
#     samples = phi.sample()
