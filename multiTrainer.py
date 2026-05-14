import os
import pickle
from functools import partial

import jax
import jax.numpy as jnp


class MultiTrainer:
    def __init__(
        self,
        datasets,
        models_per_dataset,
        model,
        sample_input,
        optimizer,
        batch_size,
        out=None,
        save_freq=1000,
        base_key=0,
        in_path=None,
    ):
        self.datasets = datasets  # jnp array of shape [num_datasets, num_samples, ...]
        self.models_per_dataset = models_per_dataset
        self.num_models = len(datasets) * models_per_dataset
        self.batch_size = batch_size
        self.log_path = out
        self.save_freq = save_freq
        self.step_count = 0

        # map model → dataset
        mdl = []
        for i in range(len(datasets)):
            for _ in range(models_per_dataset):
                mdl.append(i)
        self.model_to_dataset = jnp.array(mdl)

        # unique keys per model for initialization
        keys = jax.random.split(jax.random.PRNGKey(base_key), self.num_models)

        # init parameters
        self.apply_fn = model.apply

        if in_path is not None:
            # Load parameters from file
            with open(in_path, "rb") as f:
                self.params = pickle.load(f)
        else:
            # Initialize parameters from scratch
            self.params = jax.vmap(lambda k: model.init(k, sample_input))(keys)

        # optimizer
        self.optimizer = optimizer
        self.opt_state = jax.vmap(optimizer.init)(self.params)

    def _sample_batch_single(self, rng, dataset_id):
        d = self.datasets[dataset_id]
        n = d.shape[0]
        idx = jax.random.choice(rng, n, (self.batch_size,), replace=True)
        return d[idx]

    def sample_batch(self, rng):
        keys = jax.random.split(rng, self.num_models)
        return jax.vmap(self._sample_batch_single, in_axes=(0, 0))(
            keys, self.model_to_dataset
        )

    def loss_and_grad_single(self, params, batch, dropout_rng):
        def loss_fn(p):
            # Pass the dropout RNG stream and set training=True
            log_psi = self.apply_fn(p, batch, rngs={"dropout": dropout_rng})
            return -jnp.mean(2 * log_psi)

        loss, grad = jax.value_and_grad(loss_fn)(params)
        return loss, grad

    def loss_and_grad(self, params, batch, dropout_keys):
        # Vmap over params, batch, and unique dropout keys for each model
        return jax.vmap(self.loss_and_grad_single)(params, batch, dropout_keys)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, params, opt_state, rng):
        # Split main RNG for sampling and for dropout masks
        sample_rng, dropout_master_rng = jax.random.split(rng)

        # 1. Generate batch
        batch = self.sample_batch(sample_rng)

        # 2. Generate unique dropout keys for each model in the ensemble
        dropout_keys = jax.random.split(dropout_master_rng, self.num_models)

        # 3. Calculate loss and gradients
        loss, grads = self.loss_and_grad(params, batch, dropout_keys)

        # 4. Update via optimizer
        updates, new_opt = jax.vmap(self.optimizer.update)(grads, opt_state, params)
        new_params = jax.tree.map(lambda p, u: p + u, params, updates)

        return new_params, new_opt, loss

    def run_step(self, rng):
        self.params, self.opt_state, loss = self.step(self.params, self.opt_state, rng)
        self.step_count += 1

        if self.log_path is not None and self.step_count % self.save_freq == 0:
            dir_path = os.path.dirname(self.log_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(self.log_path, "wb") as f:
                pickle.dump(self.params, f)

        return loss
