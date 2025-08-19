from flax import linen as nn
from jax.nn.initializers import zeros
from netket.experimental.models.fast_rnn import FastRNN
from netket.experimental.nn.rnn import (
    FastRNNLayer,
    LSTMCell,
    RNNCell,
    default_kernel_init,
)
from netket.models.autoreg import _get_feature_list
from netket.utils.types import NNInitFunc


class LSTMCell(RNNCell):
    """A “fake” LSTMCell that applies a single Dense + ReLU to the inputs and ignores cell_mem."""

    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the weight matrix."""
    bias_init: NNInitFunc = zeros
    """Initializer for the bias."""

    @nn.compact
    def __call__(self, inputs, cell_mem, hidden):
        # -- apply dense + relu only to inputs
        x = nn.Dense(
            self.features,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.param_dtype,
        )(inputs)

        output = nn.relu(x)

        # Return same output in both positions to match RNNCell interface
        return output, output


class VanillaRNN(FastRNN):
    """
    Long short-term memory network with fast sampling.

    See :class:`netket.models.FastARNNSequential` for a brief explanation of fast autoregressive sampling.
    """

    def setup(self):
        features = _get_feature_list(self)
        self._layers = [
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
            for i in range(self.layers)
        ]
