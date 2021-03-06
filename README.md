# avalanche_ml
Applying machine learning in the Norwegian Avalanche warning Service

## Cloning
The repo uses `NVE/varsomdata` as a submodule. To clone it, use

    git clone --recurse-submodules git@github.com:NVE/avalanche_ml.git
or

    git clone --recurse-submodules https://github.com/NVE/avalanche_ml.git


## Conda Environment:
This repository includes an `environment.yml` file that will install all of the packages you need using Anaconda to run the Python scripts and Jupyter Notebooks. To preserve the conda environment across sessions, please add this line of code into your `~/.condarc` file, or create that file if it does not currently exist:
```bash
envs_dirs:
 - /home/user/conda-envs/
```

Once you are in the project directory, run:
```bash
conda env create -f environment.yml -n avalanche_ml
```

And to activate the environment:
```bash
conda activate avalanche_ml
```

## Example program
See [`demo_sk-classifier.py`](demo_sk-classifier.py) and [`demo_sk-cluster.py`](demo_sk-cluster.py).
