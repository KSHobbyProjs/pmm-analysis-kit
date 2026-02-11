#!/usr/bin/env python3

import numpy as np
import jax.numpy as jnp
import jax
from jax import config
config.update("jax_enable_x64", True)

class PolyFit:
    def __init__(self, dim=10,
             eta=1.0e-2, beta1=0.9, beta2=0.999, eps=1.0e-8, absmaxgrad=1.0e3,
             l2=0.0, mag=1.0e-1, seed=0):
        # PMM state
        self._dim = dim
        
        self._sample_data = {}
        self._losses = []
        self._epochs = 0

        # ADAM state
        self._eta = eta
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._absmaxgrad = absmaxgrad
        self._l2 = l2
        
        self._mag = mag   # these two are only recorded for metadata, never used internally past `_init_params()`
        self._seed = seed

        # Initialize learnable Hermitian parameters
        key = jax.random.PRNGKey(seed)
        self._params = mag * jnp.random.normal(key, shape=(self._dim,), dtype=jnp.float64) 
        self._vt = jnp.zeros_like(self._params)
        self._mt = jnp.zeros_like(self._params)

    def sample_energies(self, Ls, energies):
        Ls = jnp.atleast_1d(Ls)
        energies = jnp.atleast_1d(energies)
        if Ls.shape[0] != energies.shape[0]:
            raise RuntimeError("Sample parameters (`Ls`) and sample eigenvalues (`energies`) need to have the same length in `sample(Ls, energies)`") 
        if energies.ndim == 1:
            energies = energies[:, None]
       
        self._sample_data["Ls"], self._sample_data["energies"] = Ls, energies
        return Ls, energies

    def predict_energies(self, Ls_predict, k_num=None):
        Ls_predict = jnp.atleast_1d(Ls_predict)
        Ms = PMM._M(self._params, Ls_predict)
        eigvals, _ = PMM._get_eigenvalues(Ms)
        if k_num is None: 
            return eigvals
        else:
            return eigvals[:, :k_num] # report only the k_num lowest eigenvalues

    def train_pmm(self, epochs, store_loss=100):
        if not self._sample_data:
            raise RuntimeError("No data loaded. Run `sample_energies()` or `load()` before `train_pmm()`.")

        # construct vt and mt moments (tree.map allows us to move over the whole dictionary at once)
        params = self._params
        vt, mt = self._vt, self._mt
        Ls, energies = self._sample_data["Ls"], self._sample_data["energies"]

        # create array to store loss at epoch t
        losses = np.zeros(epochs // store_loss)

        # jit the loss function so that it's significantly quicker to call
        jit_loss = jax.jit(self.loss)
        grad_loss = jax.jit(jax.grad(jit_loss))

        for t in range(epochs):
            # update epoch counter
            self._epochs += 1
            # calculate the gradient (automatically applies through leafs (dictionary keys))
            # update the parameters with jax.tree.map (automatically aligns and moves through
            # dictionary keys so the whole dictionary can be moved through at once)
            gt = grad_loss(params, Ls, energies, self._l2)
            update = jax.tree.map(lambda p, v, m, g: PMM._adam_update(p, v, m, t, g, 
                                                                             self._eta, self._beta1, self._beta2,
                                                                             self._eps, self._absmaxgrad),
                                          params, vt, mt, gt
                                          )

            # jax.tree.map returns updates like update["primary_diags"] = (params, vt, mt), so re-split them
            # PyTrees are recursive, so the tuples inside the values will be looped over if we do another tree.map;
            # the is_leaf call prevents jax from applying the function recursively past the tuples. it stops at the values
            # of the dictionary
            params = jax.tree.map(lambda x: x[0], update, is_leaf=lambda x: isinstance(x, tuple))
            vt = jax.tree.map(lambda x: x[1], update, is_leaf=lambda x: isinstance(x, tuple))
            mt = jax.tree.map(lambda x: x[2], update, is_leaf=lambda x: isinstance(x, tuple))

            # store loss
            if t % store_loss == 0:
                losses_at_t = jit_loss(params, Ls, energies, self._l2)
                losses[t // store_loss] = losses_at_t
        
        self._losses.extend(losses)
        self._params = params
        self._vt, self._mt = vt, mt
        return params, losses 

    @staticmethod
    def loss(params, Ls, energies, l2):
        k_num = energies.shape[1]
        predict_energies = params


