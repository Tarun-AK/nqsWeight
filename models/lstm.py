import flax.linen as nn
import netket as nk
from flax import linen as nn
from flax.linen.dtypes import promote_dtype
from jax import numpy as jnp
from jax.nn.initializers import zeros
from netket.experimental.models.fast_rnn import FastRNN
from netket.experimental.nn.rnn import (
    FastRNNLayer,
    LSTMCell,
    RNNCell,
    default_kernel_init,
)
from netket.models.autoreg import ARNNSequential, _get_feature_list, _normalize
from netket.utils.types import NNInitFunc


class DenseCell(RNNCell):
    """A “fake” LSTMCell that applies a single Dense + ReLU to the inputs and ignores cell_mem."""

    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the weight matrix."""
    bias_init: NNInitFunc = zeros
    """Initializer for the bias."""

    @nn.compact
    def __call__(self, inputs, cell_mem, hidden):
        x = nn.Dense(
            self.features,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.param_dtype,
        )(inputs)

        output = nn.relu(x)

        return output, output


class FastLSTMNet(FastRNN):
    """
    Long short-term memory network with fast sampling.

    See :class:`netket.models.FastARNNSequential` for a brief explanation of fast autoregressive sampling.
    """

    def setup(self):
        features = _get_feature_list(self)
        assert self.layers > 1
        layers = [
            FastRNNLayer(
                cell=LSTMCell(
                    features=features[i],
                    param_dtype=self.param_dtype,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                ),
                size=self.hilbert.size,
                exclusive=(i == 0),
                reorder_idx=self.reorder_idx,
                inv_reorder_idx=self.inv_reorder_idx,
                prev_neighbors=self.prev_neighbors,
            )
            for i in range(self.layers - 1)
        ]
        layers.append(
            FastRNNLayer(
                cell=DenseCell(
                    features=self.hilbert.local_size,
                    param_dtype=self.param_dtype,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                ),
                size=self.hilbert.size,
                exclusive=False,
                reorder_idx=self.reorder_idx,
                inv_reorder_idx=self.inv_reorder_idx,
                prev_neighbors=self.prev_neighbors,
            )
        )
        self._layers = tuple(layers)


# class LSTMNet2D(nk.experimental.models.FastLSTMNet):
#     @nn.compact
#     def conditionals_log_psi(self, inputs):
#         inputs = self.reshape_inputs(inputs)
#
#         x = jnp.expand_dims(inputs, axis=-1)
#
#         for i in range(len(self._layers)):
#             if i > 0 and hasattr(self, "activation"):
#                 x = self.activation(x)
#             x = self._layers[i](x)
#
#         x = nn.Dense(self.)(x)
#
#         x = x.reshape((x.shape[0], -1, x.shape[-1]))
#         log_psi = _normalize(x, self.machine_pow)
#         return log_psi
