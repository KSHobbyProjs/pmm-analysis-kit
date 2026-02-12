import numpy as np
import jax.numpy as jnp
import jax
from jax import config
config.update("jax_enable_x64", True)
from typing import Sequence

from . import matutils as matutils

import logging
logger = logging.getLogger(__name__)



class PMM:
    # -------------------------- Initialization ---------------------------------------------
    def __init__(
            self,
            dim : int = 5,
            matrix_specs : Sequence[matutils.MatrixSpec] = [
                matutils.MatrixSpec(name="A", mat_type="herm"),
                matutils.MatrixSpec(name="B", mat_type="herm", basis_fn=lambda L: L)
                ],
            eta : float = 1.0e-2,
            beta1 : float = 0.9,
            beta2 : float = 0.999,
            eps : float = 1.0e-8,
            absmaxgrad : float = 1.0e3,
            mag : float = 0.5e-1,
            seed : int = 0
        ):
        """
        Initialize a Parametric Matrix Model (PMM).
        
        Parameters
        ----------
        dim : int
            Dimension of the square matrices.
        matrix_specs : Sequence[matutils.MatrixSpec]
            Specifications for the primary matrices (must be at least one).
        eta : float
            Learning rate for ADAM optimizer.
        beta1 : float
            ADAM beta1 parameter.
        beta2 : float
            ADAM beta2 parameter.
        eps : float
            ADAM epsilon parameter for numerical stability.
        absmaxgrad : float
            Maximum allowed gradient value for clipping in ADAM optimizer.
        mag : float
            Factor to multiply initial matrices by.
        seed : int
            Random seed for parameter initialization.
        """

        # define dictionary storing intial PMM configuration in case of hot start later.
        self._init_kwargs = {"dim" : dim,
                             "eta" : eta,
                             "beta1" : beta1,
                             "beta2" : beta2,
                             "eps" : eps,
                             "absmaxgrad" : absmaxgrad,
                             "mag" : mag,
                             "seed" : seed}
       
        # PMM state
        self._dim = dim
        self._matrix_specs = tuple(matrix_specs)
        self._primary_specs = tuple(
                spec for spec in self._matrix_specs
                if not spec.is_secondary
            )
        self._secondary_specs = tuple(
                spec for spec in self._matrix_specs
                if spec.is_secondary
            )
        # raise error if user attempts to train less than one primary matrices
        if len(self._matrix_specs) < 1: 
            raise ValueError(f"Parametric matrix models require at least one primary matrices, got {len(self._matrix_specs)}")
        
        self._sample_data = {}
        self._losses = []
        self._epochs = 0

        # ADAM state
        self._eta = eta
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._absmaxgrad = absmaxgrad
        
        self._mag = mag   # these two are only recorded for metadata, never used internally past `_init_params()`
        self._seed = seed

        # Initialize learnable parameters and first / second moments for ADAM
        self._params = self._init_params() 
        self._mt = jax.tree.map(jnp.zeros_like, self._params)
        self._vt = jax.tree.map(jnp.zeros_like, self._params)

    def _init_params(self):
        """
        Initialize a dictionary storing the learnable parameters for each matrix in the model.

        Each matrix is stored as a real vector of the appropriate length for reconstruction:
        - Hermitian or general real: n^2 real elements.
        - Real symmetric or PSD: n*(n+1)/2 real elements.

        Returns
        -------
        params : dict[str, jnp.ndarray]
            Dictionary mapping matrix names to their parameter vectors.
            Example: `params["A"]` contains the vector representation of matrix 'A'.
        """
        n = self._dim
        specs = self._matrix_specs
        mag = self._mag
        seed = self._seed
        
        params = {}
        # initialize a vector to store parameters for each matrix in spec
        key = jax.random.PRNGKey(seed)
        for spec in specs:
            key, subkey = jax.random.split(key)
            if spec.mat_type == "herm" or spec.mat_type == "real":
                vec_len = n * n
                params[spec.name] = mag * jax.random.normal(subkey, (vec_len,), dtype=jnp.float64)
            elif spec.mat_type == "psd" or spec.mat_type == "real_sym":
                vec_len = n * (n + 1) // 2
                params[spec.name] = mag * jax.random.normal(subkey, (vec_len,), dtype=jnp.float64)
            else:
                raise TypeError(f"Matrix type must be 'herm', 'real', 'psd' or 'realsym', got {spec.type}.")
        
        return params

    # ------------------------------------------ Sampling ------------------------------------------------------
    def sample_energies(self, Ls, energies):
        """
        Populate the PMM with sample data.

        Parameters
        ----------
        Ls : float or array-like
            Sample parameters.
        energies : float or array-like
            Sample energies.

        Returns
        -------
        Ls : jnp.ndarray
            Sample parameter array of shape (len(Ls),).
        energies : jnp.ndarray
            Sample energy array of shape (len(Ls), k).
        """
        Ls = jnp.atleast_1d(Ls)
        energies = jnp.atleast_1d(energies)
        if Ls.shape[0] != energies.shape[0]:
            raise RuntimeError("Sample parameters (`Ls`) and sample eigenvalues (`energies`) need to have the same length in `sample(Ls, energies)`") 
        if energies.ndim == 1:
            energies = energies[:, None]
       
        self._sample_data["Ls"], self._sample_data["energies"] = Ls, energies
        return Ls, energies

    # -------------------------------------------- Training ----------------------------------------------------
    def train_pmm(self, epochs, store_loss=100):
        if not self._sample_data:
            raise RuntimeError("No data loaded. Run `sample_energies()` or `load()` before `train_pmm()`.")

        # construct vt and mt moments (tree.map allows us to move over the whole dictionary at once)
        primary_specs = self._primary_specs
        secondary_specs = self._secondary_specs
        params = self._params
        mt, vt = self._mt, self._vt
        sample_data = self._sample_data
        n = self._dim

        # create array to store loss at epoch t
        losses = np.zeros(epochs // store_loss)

        # wrap the loss to avoid issues with non-hashable spec variable
        def loss_wrapped(params, sample_data):
            return self.loss(n, primary_specs, secondary_specs, params, sample_data)
        # jit the loss function so that it's significantly quicker to call
        jit_loss = jax.jit(loss_wrapped)
        grad_loss = jax.jit(jax.grad(loss_wrapped))

        for t in range(epochs):
            # update epoch counter
            self._epochs += 1
            # calculate the gradient (automatically applies through leafs (dictionary keys))
            # update the parameters with jax.tree.map (automatically aligns and moves through
            # dictionary keys so the whole dictionary can be moved through at once)
            gt = grad_loss(params, sample_data)
            update = jax.tree.map(lambda p, m, v, g: PMM._adam_update(p, m, v, t, g, 
                                                                             self._eta, self._beta1, self._beta2,
                                                                             self._eps, self._absmaxgrad),
                                          params, mt, vt, gt
                                          )

            # jax.tree.map returns updates like update["A"] = (params, mt, vt), so re-split them
            # PyTrees are recursive, so the tuples inside the values will be looped over if we do another tree.map;
            # the is_leaf call prevents jax from applying the function recursively past the tuples. it stops at the values
            # of the dictionary
            params = jax.tree.map(lambda x: x[0], update, is_leaf=lambda x: isinstance(x, tuple))
            mt = jax.tree.map(lambda x: x[1], update, is_leaf=lambda x: isinstance(x, tuple))
            vt = jax.tree.map(lambda x: x[2], update, is_leaf=lambda x: isinstance(x, tuple))

            # store loss
            if t % store_loss == 0:
                losses_at_t = jit_loss(params, sample_data)
                losses[t // store_loss] = losses_at_t
        
        self._losses.extend(losses)
        self._params = params
        self._mt, self._vt = mt, vt
        return params, losses 

    # -------------------------------------------- Prediction -------------------------------------------------
    def predict_energies(self, predict_Ls, k_num=None):
        predict_Ls = jnp.atleast_1d(predict_Ls)
        Ms = PMM._M(self._dim, self._matrix_specs, self._params, predict_Ls)
        eigvals, _ = PMM._get_eigenpairs(Ms)
        if k_num is None: 
            return eigvals
        else:
            return eigvals[:, :k_num] # report only the k_num lowest eigenvalues

    # add function here that wraps all pmm mechanics: sampling, training, predicting, saving, and loading
    # keep saving and loading separate in a pipeline code (like if load: PMM.load, etc.)
    def run_pmm(self, sample_Ls, energies, epochs, predict_Ls, k_num=None, store_loss=100):
        self.sample_energies(sample_Ls, energies)
        _, losses = self.train_pmm(epochs, store_loss=store_loss)
        eigvals = self.predict_energies(predict_Ls, k_num=k_num)
        return losses, eigvals

    # ------------------------------------------- Saving / Loading State ---------------------------------------
    def get_metadata(self):
        # flag error if no data has been sampled
        if not self._sample_data: raise RuntimeError("Can't get metadata because `sample()` hasn't been run. Sample data needs to be run to store metadata.")
        # if data has been sampled but pmm hasn't been run
        final_loss = self._losses[-1] if len(self._losses) > 0 else 'not-run'
        metadata = {
                "type" : self.__class__.__name__,
                "dim" : self._dim,
                "num_primary" : self._num_primary,
                "k_num_sample" : self._sample_data["energies"].shape[1],
                "epochs" : self._epochs,
                "final_loss" : final_loss,
                "num_secondary" : self._num_secondary,
                "eta" : self._eta,
                "beta1" : self._beta1,
                "beta2" : self._beta2,
                "eps" : self._eps,
                "absmaxgrad" : self._absmaxgrad,
                "l2" : self._l2,
                "mag" : self._mag,
                "seed" : self._seed
                }
        return metadata

    def get_state(self):
        state = {
                # training info
                "data" : self._sample_data,
                "losses" : self._losses,
                "params" : self._params,
                "vt" : self._vt,
                "mt" : self._mt,
                # adam info
                "eta" : self._eta,
                "beta1" : self._beta1,
                "beta2" : self._beta2,
                "eps" : self._eps,
                "absmaxgrad" : self._absmaxgrad,
                "l2" : self._l2,
                # model info
                "dim" : self._dim,
                "num_primary" : self._num_primary,
                "num_secondary" : self._num_secondary,
                "mag" : self._mag,
                "seed" : self._seed,
                "epochs" : self._epochs
                }
        return state

    def set_state(self, state):
        # define function to re-jax-ify arrays
        def _to_jax(x):
            if isinstance(x, np.ndarray):
                return jnp.array(x)
            elif isinstance(x, dict):
                return {k : _to_jax(v) for k, v in x.items()}
            else:
                return x

        # training info
        self._sample_data = _to_jax(state["data"])
        self._losses = state["losses"]
        self._params = _to_jax(state["params"])
        self._vt = _to_jax(state["vt"])
        self._mt = _to_jax(state["mt"])
        # adam info
        self._eta = state["eta"]
        self._beta1 = state["beta1"]
        self._beta2 = state["beta2"]
        self._eps = state["eps"]
        self._absmaxgrad = state["absmaxgrad"]
        self._l2 = state["l2"]
        # model info
        self._dim = state["dim"]
        self._num_primary = state["num_primary"]
        self._num_secondary = state["num_secondary"]

        self._mag = state["mag"]
        self._seed = state["seed"]
        self._epochs = state["epochs"]

    # ------------------------------------------- Loss and Basis for M ---------------------------------------
    # loss function
    # mean squared error of the predicted eigenvalues to the true eigenvalues
    @staticmethod
    def loss(n, primary_specs, secondary_specs, params, sample_data):
        """
        Calculate the loss given a specific PMM configuration.

        Parameters
        ----------
        n : int
            Dimension of PMM.
        specs : Sequence[matutils.MatrixSpec]
            Specification for the model matrices.
        params : dict[str, jnp.ndarray]
            Parameters storing matrix information.
        Ls : jnp.ndarray
            Sample parameter array of shape (len(Ls),).
        energies : jnp.ndarray
            Sample energy array of shape (len(Ls), k).

        Returns
        -------
        loss : float
            Loss value for given PMM configuration.
        """
        Ls, energies = sample_data["Ls"], sample_data["energies"]
        k_num = energies.shape[1]
        Ms = PMM._M(n, primary_specs, params, Ls)
        eigvals, _ = PMM._get_eigenpairs(Ms)
        eigvals = eigvals[:, :k_num] # truncate to the k_num_sample lowest eigenvalues
        loss = jnp.mean(jnp.abs(eigvals - energies)**2)
        return loss

    # -------------------------------------------- Utility Methods --------------------------------------------

    # get all eigenvalues of M (or Ms if M is given as a batch of matrices)
    @staticmethod
    def _get_eigenpairs(M):
        """
        Calculate the eigenpairs of PMM model M.

        Parameters
        ----------
        M : jnp.ndarray
            Array of PMM matrices of shape (num_primary, n, n).
        
        Returns
        -------
        eigvals : jnparray
            shape (len(M), k_num,)
        eigvecs : jnparray
            shape (len(M), k_num, n)
        """

        # compute eigenpairs
        eigvals, eigvecs = jnp.linalg.eigh(M)

        # sort eigenpairs
        idx = jnp.argsort(eigvals, axis=1)
        eigvals = jnp.take_along_axis(eigvals, idx, axis=1)
        eigvecs = jnp.take_along_axis(eigvecs, idx[:, None, :], axis=2)

        # transpose eigvecs to (len(M), k_num, :)
        eigvecs = eigvecs.swapaxes(1, 2)

        return eigvals, eigvecs

    @staticmethod
    def _M(n, primary_specs, params, Ls):
        """
        Construct the PMM matrix M given matrix specs and the updated parameters.
        
        Parameters
        ----------
        n : int
            PMM dimension.
        specs : Sequence[matutils.MatrixSpec]
            Specification for the model matrices. 
        params : dict[str, jnp.ndarray]
            Parameters storing matrix information.
        Ls : jnp.ndarray
            Sample parameter array of shape shape (len(Ls),).
        
        Returns
        -------
        M : jnp.ndarray
            List of PMM matrices. Shape (num_primary, n, n).
        """
        terms = [
            spec.basis_fn(Ls)[:, None, None]
            * matutils.build_matrix(n, spec, params[spec.name])
            for spec in primary_specs
        ]

        M = jnp.sum(jnp.stack(terms), axis=0)
        """
        M = sum(
                spec.basis_fn(Ls)[:, None, None] * matutils.build_matrix(n, spec, params[spec.name]) for spec in specs
                if not spec.is_secondary
                )
        """
        return M
   
       
    # define general Adam-update for complex parameters and real loss functions (no longer using complex ADAM. see below.).
    # update: parameters are real even if M is complex, so we use the real ADAM update function.
    @staticmethod
    def _adam_update(parameter, mt, vt, t, grad, eta=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, absmaxgrad=1e3):
        # cap the gradient with absmaxgrad
        gt = jnp.clip(grad, -absmaxgrad, absmaxgrad) 
        # compute the moments (momentum and normalizing) step parameters
        mt = beta1 * mt + (1 - beta1) * gt
        vt = beta2 * vt + (1 - beta2) * jnp.abs(gt)**2

        # bias correction
        mt_hat = mt / (1 - beta1 ** (t + 1))
        vt_hat = vt / (1 - beta2 ** (t + 1))

        # step parameter
        parameter = parameter - eta * mt_hat / (jnp.sqrt(vt_hat) + eps)
        return parameter, mt, vt

class PMMSecondary(PMM):
    def _make_RR(n, secondary_spec, params):
        RR = matutils.build_matrix(n, secondary_spec, params[secondary_spec.name])
        return RR
            
    def sample_secondarydata(self, secondarydata):
        secondarydata = jnp.atleast_1d(secondarydata)

        if secondarydata.ndim == 1:
            secondarydata = secondarydata[:, None]

        self._sample_data["secondarydata"] = secondarydata

    def _get_expected_rrs(RR, eigvecs):
        return jnp.einsum('abi,ij,abj->ab', eigvecs, RR, eigvecs)

    @staticmethod
    def loss(n, primary_specs, secondary_specs, params, sample_data):
        Ls, energies, secondarydata = sample_data["Ls"], sample_data["energies"], sample_data["secondarydata"]
        k_num, k_num_secondary = energies.shape[1], secondarydata.shape[1]
        
        # mean-squared error energy loss term
        Ms = PMMSecondary._M(n, specs, params, Ls)
        eigvals, eigvecs = PMMSecondary._get_eigenpairs(Ms)
        eigvals = eigvals[:, :k_num] # truncate to the k_num_sample lowest eigenvalues
        energyloss = jnp.mean(jnp.abs(eigvals - energies)**2)

        # secondary-data loss term
        RR = PMMSecondary._make_RR(n, secondary_specs[0], params)
        expected_rrs = PMMSecondary._get_expected_rrs(RR, eigvecs)
        rrloss = jnp.mean(jnp.abs((rrs - expected_rrs) / Ls[:, None]**2)**2)
        
        loss = energyloss + rrloss
        return loss

    def predict_secondarydata(self, predict_Ls, k_num=None):
        predict_Ls = jnp.atleast_1d(predict_Ls)
        _, eigvecs = self._get_eigenpairs(self._M(self._dim, self._primary_specs, self._params, predict_Ls))
        RR = self._make_RR(self._dim, self._secondary_specs, self._params)
        rrs = self._get_expected_rrs(RR, eigvecs)
        if k_num is None:
            return rrs
        else:
            return rrs[:, :k_num]



