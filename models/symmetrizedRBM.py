import flax.linen as nn
import jax.numpy as jnp
from netket.models import RBM


class SymmetrizedRBM(RBM):
    @nn.compact
    def __call__(self, input):
        flippedInput = -1 * input
        p = jnp.exp(super().__call__(input))
        pFlipped = jnp.exp(super().__call__(flippedInput))
        pAvg = 0.5 * (p + pFlipped)
        return jnp.log(pAvg)
