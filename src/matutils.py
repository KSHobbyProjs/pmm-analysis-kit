import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from dataclasses import dataclass
from typing import Literal, Callable

MatrixType = Literal["real_sym", "herm", "psd", "real"]
@dataclass
class MatrixSpec:
    """
    Define the specifications for how a specific PMM matrix behaves.

    Parameters
    ----------
    name : str
        Name of the PMM. E.g. 'A'.
    is_secondary : bool, optional
        Determines if the matrix is treated as a secondary matrix in the PMM.
        Default is False.
    mat_type : MatrixType
        Type of matrix. E.g. "herm" corresponds to a Hermitian matrix.
    basis_fn : Callable, optional
    """
    name : str
    mat_type : MatrixType
    is_secondary : bool = False
    basis_fn : Callable[[jnp.ndarray], jnp.ndarray] = lambda L: jnp.ones_like(L)

def vec_to_real(vec, n):
    A = vec.reshape((n, n))
    return A

# unpack vector into a real, symmetric matrix
def vec_to_realsym(vec, n):
  A = jnp.zeros((n, n), dtype=jnp.float64)
  idx = jnp.triu_indices(n)
  A = A.at[idx].set(vec)
  A = A.at[(idx[1], idx[0])].set(vec)
  return A

# unpack vector into a positive, semi-definite matrix
def vec_to_psd(vec, n):
  F = jnp.zeros((n, n), dtype=jnp.float64)
  idx = jnp.tril_indices(n)
  F = F.at[idx].set(vec)

  # enforce positive diagonal
  diag_idx = jnp.diag_indices(n)
  F = F.at[diag_idx].set(jax.nn.softplus(F[diag_idx]) + 1e-6)
  return F @ F.T

# unpack vector into Hermitian matrix
def vec_to_herm(vec, n):
  F = jnp.zeros((n, n), dtype=jnp.complex128)

  # set real diagonal
  diag = vec[:n]
  diag_idx = jnp.diag_indices(n)
  F = F.at[diag_idx].set(diag)

  # set off-diagonal
  off = vec[n:]
  assert off.shape[0] == 2 * (n * (n-1) // 2)
  idx = jnp.triu_indices(n, k=1)
  real, imag = off[::2], off[1::2]
  offelems = real + 1j * imag
  F = F.at[idx].set(offelems)
  F = F.at[(idx[1], idx[0])].set(jnp.conj(offelems))
  return F

def build_matrix(n, spec, vec):
    if spec.mat_type == "real_sym":
        A = vec_to_realsym(vec, n)
    elif spec.mat_type == "herm":
        A = vec_to_herm(vec, n)
    elif spec.mat_type == "psd":
        A = vec_to_psd(vec, n)
    elif spec.mat_type == "real":
        A = vec_to_real(vec, n)
    else:
        raise TypeError(f"Got an unsupported matrix type {spec.mat_type}.")
    return A
    
