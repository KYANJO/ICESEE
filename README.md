# **ICESEE**

**Ice Sheet State and Parameter Estimator (ICESEE)** is a state-of-the-art data assimilation software package designed for ice sheet models. This advanced software facilitates the creation of an adaptive intelligent wrapper with robust protocols and APIs to seamlessly couple and integrate with various ice sheet models. The primary objective is to simplify the interaction between different models, enabling the adoption of complex data assimilation techniques across multiple frameworks.

This design is being extended to integrate with cloud computing services such as **AWS**, ensuring scalability and efficiency for larger simulations. Eventually, the software will be incorporated into the **GHUB online ice sheet platform**, significantly enhancing its capabilities by including the new features currently under development.

---

## Installation

ICESEE can be installed as a Python package for general use or set up for development. Choose the appropriate method below based on your needs.

### Option 1: Install as a Python Package (General Use)

To use ICESEE as a Python module, install it via pip:

```bash
pip install ICESEE
```

This installs the latest released version of ICESEE from PyPI 

### Option 2: Virtual Environment + Package Installation (Recommended for Development)

Clone the repository and set up a virtual environment with the package installed:

```bash
git clone https://github.com/your-repo/ICESEE.git 
cd ICESEE
chmod +x setup_venv.sh
./setup_venv.sh
source venv/bin/activate
make install
```

- `setup_venv.sh` creates a virtual environment named `icesee-env` with the project directory in `PYTHONPATH`.
- `make install` installs ICESEE in editable mode (`pip install -e .`), making `PYTHONPATH` unnecessary.

### Option 3: Virtual Environment + PYTHONPATH (Development)

Clone the repository and set up a virtual environment with `PYTHONPATH` configured:

```bash
git clone https://github.com/your-repo/ICESEE.git 
cd ICESEE
chmod +x setup_venv.sh
./setup_venv.sh
source venv/bin/activate
```

The virtual environment automatically includes the project directory in `PYTHONPATH` via `sitecustomize.py`.

### Option 4: Manual PYTHONPATH Setup (Development)

Clone the repository and manually configure `PYTHONPATH`:

```bash
git clone https://github.com/your-repo/ICESEE.git 
cd ICESEE
make setup
source ~/.bashrc  # or source ~/.zshrc
```

Alternatively, for Windows:

```cmd
git clone https://github.com/your-repo/ICESEE.git 
cd ICESEE
setup_env.bat
```

- `make setup` or `setup_env.bat` adds the project directory to `PYTHONPATH`.
- Run `source ~/.bashrc` (or equivalent) to apply changes.

### Windows Setup (Development)

For Windows users, after cloning:

1. Set up a virtual environment:
```cmd
setup_venv.bat
venv\Scripts\activate
```

2. Install ICESEE:
```cmd
make install
```

Or use `PYTHONPATH`:
```cmd
setup_env.bat
```

---

## Usage

After setup, you can run ICESEE applications. Example:

```bash
python -m ICESEE.applications.icepack.examples.synthetic_flow.synthetic_flow
```

This runs the `synthetic_flow.py` script, which uses functions from `ICESEE.core.utilities` (e.g., `some_function`).

### Supported Applications

The supported applications are located in the [applications](./applications) directory and currently include:
- **[Flowline](./applications/flowline/)**: Documentation and full implementation forthcoming.
- **[Icepack](./applications/icepack_model/)**: Ice sheet modeling application (fully developed).
- **[ISSM](./applications/issm_model/)**: Ice sheet modeling application (development underway)
- **[Lorenz-96](./applications/lorenz-96/)**: Simplified model for testing data assimilation.

Each application includes either a Python script or a Jupyter notebook for execution. Detailed documentation is included in the README files in each application folder.

### Running Icepack in Containers

Icepack applications can be run in containers using **Apptainer** and **Docker**, making them suitable for high-performance computing (HPC) clusters. For details, see [/src/container/apptainer](./src/container/apptainer/).

### Executing Applications with Data Assimilation

ICESEE incorporates four variants of the Ensemble Kalman Filter (EnKF) for data assimilation:

1. **EnKF**: Stochastic Ensemble Kalman Filter, available in two forms (traditional and a robust variant).
2. **DEnKF**: Deterministic Ensemble Kalman Filter.
3. **EnTKF**: Ensemble Transform Kalman Filter.
4. **EnRSKF**: Ensemble Square Root Kalman Filter.

These variants provide flexible and scalable data assimilation methods optimized for ice sheet modeling. For detailed usage instructions, consult the README files in each application’s directory.

Currently, only the **EnKF** variant supports advanced parallelization and cutting-edge data assimilation techniques, such as new implementations of robust random fields and pseudorandom field generators, eliminating the need for localization. Both MPI-based and non-MPI-based applications are supported for data assimilation.

The other variants (**DEnKF**, **EnTKF**, **EnRSKF**) are available in the serial version of ICESEE, with their implementations located in the [/src/EnKF](./src/EnKF) directory. At present, only **EnKF** and **DEnKF** support localization, while all variants support inflation.

For non-MPI applications, the traditional implementations of **EnKF** and **DEnKF** are fully parallelized.

## Build the Package

To build and distribute ICESEE to PyPI (required for `pip install ICESEE`), ensure you have **setuptools**, **wheel**, and **twine** installed:

```bash
pip install setuptools wheel twine
```

1. Build the package:
```bash
python setup.py sdist bdist_wheel
```

2. Upload to PyPI (requires PyPI credentials):
```bash
twine upload dist/*
```

This makes ICESEE available for installation via `pip install ICESEE`.

---

## Development Notes

- **Namespace Package**: ICESEE is structured as a Python namespace package (no `__init__.py` in `ICESEE/`), allowing modularity and clean imports (e.g., `from ICESEE.src.run_model_da`).
- **Virtual Environment**: Use `setup_venv.sh` (or `setup_venv.bat` for Windows) to create an isolated environment, ensuring no conflicts with system Python.
- **Makefile**: Simplifies setup with `make setup` (configures `PYTHONPATH`) or `make install` (installs the package, recommended).
- **Dependencies**: Add required libraries (e.g., numPy, sciPy, mpi4py,gstools) to `setup.py` under `install_requires`.
- **Testing**: Test the setup by cloning the repository in a clean environment (e.g., Docker container).

---

## Project Structure

```

ICESEE/
├── applications/
│   ├── icepack_model/
│   │   ├── examples/
│   │   │   ├── synthetic_ice_stream/
│   │   │   ├── shallowIce/
│   │   │──icepack_utils 
│   ├── issm_model/
│   │   ├── examples/
│   │   │   ├── ISMIP/
│   │   ├──issm_utils 
│   ├── flowline_model/
│   ├── lorenz-96/
├── src/
│   ├── EnKF
│   ├── Container
│   ├── parallelization
│   ├── run_model_da
│   ├── tests
│   ├── utils
├── config/
├── setup.py
├── pyproject.toml
├── Makefile
├── setup_venv.sh
├── setup_venv.bat
├── setup_env.bat
├── README.md
```

---

## Future Plans

- Integration with **AWS** for scalable cloud computing.
- Incorporation into the **GHUB online ice sheet platform** with enhanced features.

For questions or contributions, please open an issue or pull request on the [GitHub repository](https://github.com/your-repo/ICESEE) or contact me at bkyanjo3@gatech.edu