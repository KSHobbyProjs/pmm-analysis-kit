# GLOBAL PMM ANSATZ PRESETS
import jax

from src import matutils

affine = [
        matutils.MatrixSpec(name="A", mat_type="herm"),
        matutils.MatrixSpec(name="B", mat_type="herm", basis_fn=lambda L: L)
        ]

affine_sec = [
        matutils.MatrixSpec(name="A", mat_type="real_sym"),
        matutils.MatrixSpec(name="B", mat_type="real_sym", basis_fn=lambda L: L),
        matutils.MatrixSpec(name="R", mat_type="psd", is_secondary=True)
        ]

affine_realsym = [
        matutils.MatrixSpec(name="A", mat_type="real_sym"),
        matutils.MatrixSpec(name="B", mat_type="real_sym", basis_fn=lambda L: L)
        ]

luscher = [
        matutils.MatrixSpec(name="A", mat_type="real_sym"),
        matutils.MatrixSpec(name="B", mat_type="real_sym", basis_fn=lambda L: jnp.exp(L) / L)
        ]

PRESETS = {
        "affine" : affine,
        "affine_realsym" : affine_realsym,
        "luscher" : luscher, 
        "affine_sec" : affine_sec
        }


