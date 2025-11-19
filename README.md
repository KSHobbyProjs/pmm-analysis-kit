# pmm-utils
A repository for a set of tools for computing exact eigenpair data of a physical model, running a parametric matrix model on that data, or running
eigenvector continuation on that data.

## Project Structure

```
└── pmm-analysis-kit/
    ├── get_eigenpairs.py
    ├── run_pmm.py
    ├── run_ec.py
    ├── src/
    │   ├── parse.py
    │   ├── utils.py
    │   ├── pmm.py
    │   ├── io.py
    │   ├── ec.py
    │   └── physics_models/
    │       ├── noninteracting_spins.py
    │       ├── ising.py
    │       ├── base_model.py
    │       └── gaussian.py
    ├── README.md
    └── requirements.txt
```

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/KSHobbyProjs/pmm-analysis-kit.git
cd pmm-analysis-kit
pip install -r requirements.txt
```

Typical dependencies include:
- `numpy`
- `scipy`
- `matplotlib`
- `jax`

---

## Usage

Compute exact eigenpairs of a model using the `get_eigenpairs.py` script, and use `run_ec.py` or `run_pmm.py` to predict energies using that data.

```bash
./get_eigenpairs.py --m gaussian.Gaussian1d:N=128,V0=-4.0,R=2.0 -L 5.0,6.0 -k 4 --vectors -o sample.h5
./run_ec.py sample.h5 -L 5.0,20.0:150 -o ec_results.dat
./pmm_ec.py sample.h5 -L 5.0,20.0:150 -o pmm_results.dat
```

More information on the arguments used for each script can be found in their individual git repos:
`https://github.com/KSHobbyProjs/pmm.git`
`https://github.com/KSHobbyProjs/ec.git`
`https://github.com/KSHobbyProjs/exact-eigenpairs.git`

## Extending the Framework

To add a new physics model:
1. Create a new file under `src/physics_models/` (e.g., `double_well.py`).
2. Create a class within this file that subclasses `BaseModel`.
3. Change
   ```python
   def construct_H(self, L): ...
   ```
   so that it constructs the Hamiltonian for your model as it depends on the parameter $L$.

To add a new PMM variant:
1. Add a new class under `src/pmm.py` that subclasses `PMM`.
2. Modify
   ```python
   def loss(params, Ls, energies, l2): ...
   def get_basis(Ls, num_primary): ...
   ```
   to change how the loss is computed (default is mean squared error between predicted eigenvalues and sample eigenvalues,
   and how the basis is constructed (default is affine $H_\theta=A_\theta+\lambda B_\theta$).
---

