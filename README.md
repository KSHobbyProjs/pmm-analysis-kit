# pmm-utils
A toolkit for computing exact eigenpair data of physical models and running a parametric matrix model / eigenvectors continuation using that data.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/KSHobbyProjs/pmm-analysis-kit.git
cd pmm-analysis-kit
pip install -r requirements.txt
```

Dependencies inclue `numpy`, `scipy`, `matplotlib`, `jax`, `h5py`

---

## Usage

Compute exact eigenpairs of a model using the `get_eigenpairs.py` script, use `run_ec.py` or `run_pmm.py` to predict energies using that sample data with EC or PMMs, respectively, and plot the data with `plot_energies.py`.

```bash
python get_eigenpairs.py --m gaussian.Gaussian1d:N=128,V0=-4.0,R=2.0 -L 5.0,6.0 -k 4 --vectors -o sample.h5
python run_ec.py sample.h5 -L 5.0,20.0:150 -o ec_results.dat
python pmm_ec.py sample.h5 -L 5.0,20.0:150 -o pmm_results.dat
python plot_energies.py -f sample.h5 *.dat -l="o,-,--"
```

More information on the arguments used for each script can be found in their individual git repos:
- `https://github.com/KSHobbyProjs/pmm.git`
- `https://github.com/KSHobbyProjs/ec.git`
- `https://github.com/KSHobbyProjs/exact-eigenpairs.git`

---

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

