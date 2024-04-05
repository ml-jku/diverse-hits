# diverse-hits
![Tests](https://github.com/ml-jku/diverse-hits/actions/workflows/test_main.yml/badge.svg?branch=main)
<a target="_blank" href="https://colab.research.google.com/github/ml-jku/diverse-hits/blob/main/example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


![graphic_abstract](https://github.com/ml-jku/diverse-hits/blob/main/notebooks/figures/graphic_abstract.png?raw=true)

This repository contains the code associated to the paper
**Diverse Hits in de Novo Molecule Design: A Diversity-based Comparison of Goal-directed Generators**. In this study we compared 14 goal-directed generators in their ability to generate diverse, high-scoring molecules under different compute constraints.

This repository contains the code to:
- Benchmark your own optimizer using the setup described in the paper. 
- Easily create your own diverse optimization setup using the provided scoring function.
- Reproduce the results of the paper.

The central part of the code is the `divopt` package which contains a scoring function class that comes with a diversity filter, tracks the generated molecules and takes care of stopping the optimization after a specified time limit or after the scoring function has evaluated a specified number of unique molecules. `example.ipynb` gives a quick overview of the functionality.

Feel free to raise an issue if you have any questions or problems with the code.

## Setup
Clone the git repository:
```
git clone https://github.com/renzph/diverse-efficiency.git
cd diverse-efficiency
```

### Setup to test your own optimizer
If you want to test your own optimizer, you need install the `divopt` package and make use of the scoring function including the diversity filter. To do so run:

```
echo export DIVOPTPATH=$(pwd) >> ~/.bashrc
pip install -e .
```
The notebook `example.ipynb` shows how to use the scoring function and how to evaluate the results.

### Setup to test all optimizers in the paper
This will create two conda environments that will be able to run all the provided optimizers.
All optimizers except for GFlowNet run in the `divopt` environment.
The `gflownet` environment requires `g++` to be installed. 

To install the dependencies and download the data run:
```
bash setup.sh
```

You can test the installation by running the tests:
```
python -m pytest test -s
```

## Reproduce the benchmark results
The results can be reproduced following a few steps:
1. Run hyperparameter search
2. Select best models for time/sample limit
3. Re-run those models with different random seeds
4. Run the virtual screening baselines. 

The steps are detailed below and can be achieved by running the following
```
python scripts/define_search_space.py --runs_base ./runs --num_trials 15 --seed 0
python scripts/run_directory.py --base_dir runs/hyperparameter_search
python scripts/create_repeat_runs.py --runs_base runs --num_repeats 5
python scripts/run_directory.py --base_dir runs/best_variance_samples
python scripts/run_directory.py --base_dir runs/best_variance_time

python scripts/create_virtual_screening_runs.py --runs_base ./runs --num_repeats 5
python scripts/run_directory.py --base_dir runs/virtual_screening
```

**Note:** All algorithms should be run in environment `divopt` except for the gflownet variants which need to be executed in the `gflownet` environment. `scripts/run_directory.py` selects the correct environment automatically for each algorithm.

### Hyperparameter tuning 
The run dirs for hyperparameter search are created using 
```
python scripts/define_search_space.py --runs_base ./runs --num_trials 15 --seed 0
```

This creates run directories of form `runs/hyperparameter_search/{task_name}_{optimizer_name}_{idx}`.
Each such directory has a config.json file specifying the run settings. 
These runs serve for the time/sample limit simultaneously.

Those runs can be started using:
```
python scripts/run_directory.py --base_dir runs/hyperparameter_search
```
This script goes through rundirs without results and locks them during a run. 
This means one can start multiple instances (if the used machine has enough CPUs/GPUs)
or on different machines if they share a network drive.

The results of a run are in general given by multiple files:
- `results.csv` gives the whole generation history
- `results_diverse_[all|novel]_[samples|time].csv` gives the generation history of only the diverse hits 
- `metrics.json` gives all relevant scalar metrics for the runs.

The latter will be used alongside `config.json` to create a dataframe of performance and 
model parameters.

### Repeated runs
For both the time/sample setting best hyperparameters are determined according to 
the most diverse hits and new run_dirs are created for repetitions of the same model setting.
The following script 

```
python scripts/create_repeat_runs.py --runs_base runs --num_repeats 5 
```

reads the search results in `runs/hyperparameter_search` and creates 
new directories `runs/best_variance_samples` and `runs/best_variance_time` for the respective compute limits. This will give us error bars on the performance values.

The new run directories will have names of the shape `{task_name}_{optimizer_name}_{search_idx}_{repeat_idx}`. 

The runs are executed using 
```
python scripts/run_directory.py --base_dir runs/best_variance_samples
python scripts/run_directory.py --base_dir runs/best_variance_time
```
### Virtual screening baselines
This creates the configs and runs virtual screening baselines
```
python scripts/create_virtual_screening_runs.py --runs_base ./runs --num_repeats 5
python scripts/run_directory.py --base_dir runs/virtual_screening
```

### Optionally: Repeat runs without diversity filter
```
python scripts/create_nodf_runs.py --runs_base=runs
python scripts/run_directory.py --base_dir runs/best_variance_samples_nodf  
python scripts/run_directory.py --base_dir runs/best_variance_time_nodf  
```


## Visualize results
All the plots and tables are created in jupyter notebooks in the `notebooks` folder.

- `barplots.ipynb`: Main results as barplots + variants not in the paper.
- `hyperparameter_table.ipynb`: Hyperparameter search spaces and selected parameters.
- `tables_all_metrics.ipynb`: More metrics including diverse hits, novel diverse hits and internal diversity.
- `property_distributions.ipynb`: Distributions for molecular properties for constraint settings and optimizers
- `optimization_curves.ipynb`: Optimization curves for the optimizers.

All resulting figures/tables are stored in the paths  `notebooks/figures` and `notebooks/tables` respectively.

## Reproduction of data and scoring functions

### Download scoring function data
```bash
mkdir -p data/scoring_functions/gsk3
wget -O data/scoring_functions/gsk3/all.txt https://raw.githubusercontent.com/wengong-jin/multiobj-rationale/master/data/gsk3/all.txt

mkdir -p data/scoring_functions/jnk3
wget -O data/scoring_functions/jnk3/all.txt https://raw.githubusercontent.com/wengong-jin/multiobj-rationale/master/data/jnk3/all.txt

python scripts/prepare_reinvent_data.py
```

### Train the scoring functions
The scoring functions can be trained using:
```
jupyter nbconvert --execute train_scoring_functions.ipynb
``` 
The results will be written to `data/scoring/functions/{task_name}`:
- `classifier.pkl`: Trained random forest classifier 
- `stats.json`: Information about the data set and predictive performance
- `splits.csv`: The original data set including information about the train/test splits

### Property thresholds
The following will compute the scoring function thresholds:
```
jupyter nbconvert --execute property_thresholds.ipynb
```
The results are stored in 
- `data/guacamol_known_bits.json`: All unique ECFP4 hash values in the ChEMBL calibration set.
- `data/guacamol_thresholds.json`: Threshold values to be used during optimization


### Prepare MaxMin screening library
Re-order the guacamol screening library using the MaxMin algorithm. 
This took a long time for me so I recommend not doing it. The results are stored in `data/guacamol_v1_all_maxmin_order.smiles`.
The command is
```
python scripts/create_maxmin_library.py
```





