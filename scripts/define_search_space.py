import json
import os
from itertools import product
from typing import Any, Dict

import numpy as np
from scipy.stats import loguniform, randint, uniform

divoptpath = os.environ.get("DIVOPTPATH")
if divoptpath is None:
    raise ValueError("DIVOPTPATH environment variable is not set")

arg_path = os.path.join(divoptpath, "data/optimizer_default_args.json")
with open(arg_path, "r") as f:
    optimizer_args_defaults = json.load(f)


def uniform_range(a, b):
    return uniform(loc=a, scale=b - a)


str_to_dist = {
    "randint": randint,
    "loguniform": loguniform,
    "uniform": uniform_range,
}


def parse_search_spaces(search_spaces_str):
    """
    Parse the search spaces from a string representation.

    Args:
        search_spaces_str (dict): A dictionary containing the search spaces as strings.

    Returns:
        dict: A dictionary containing the parsed search spaces.

    """
    search_spaces_from_str = {}
    for optimizer_name, ss in search_spaces_str.items():
        search_spaces_from_str[optimizer_name] = {}
        for par, (dist_name, (lower, upper)) in ss.items():
            dist = str_to_dist[dist_name]
            search_spaces_from_str[optimizer_name][par] = dist(lower, upper)
    return search_spaces_from_str


def sample_search_space(search_space: dict) -> dict:
    """
    Samples parameters from the given search space.

    Parameters:
    - search_space (dict): A dictionary containing parameter names as keys and probability distributions as values.

    Returns:
    - sampled_parameters (dict): A dictionary containing sampled parameter values.
    """
    sampled_parameters = {param: dist.rvs(size=1)[0] for param, dist in search_space.items()}
    return sampled_parameters


def get_complete_parameters(optimizer_name: str) -> Dict[str, Any]:
    sampled_parameters = sample_search_space(search_spaces[optimizer_name])
    complete_parameters = optimizer_args_defaults[optimizer_name].copy()
    complete_parameters.update(sampled_parameters)

    if optimizer_name == "LSTM-HC":
        complete_parameters["keep_top"] = int(complete_parameters["mols_to_sample"] * 0.5)
    # double check that njobs is set to 8
    if "n_jobs" in complete_parameters:
        complete_parameters["n_jobs"] = 8

    for k, v in complete_parameters.items():
        if isinstance(v, np.int64):
            complete_parameters[k] = int(v)
        if isinstance(v, np.float64):
            complete_parameters[k] = float(v)
    return complete_parameters


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--runs_base",
        type=str,
        help="Base directory for runs. Usually ./runs",
    )
    argparser.add_argument(
        "--num_trials",
        type=int,
        default=15,
        help="Number of trials to create",
    )
    argparser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    # example invocation:
    # python scripts/define_search_space.py --runs_base ./runs --num_trials 15 --seed 0

    args = argparser.parse_args()
    hyperpar_dir = os.path.join(args.runs_base, "hyperparameter_search")

    tasks = ["GSK3", "JNK3", "DRD2"]
    # load search spaces
    with open("data/search_spaces.json", "r") as f:
        search_spaces_str = json.load(f)

    search_spaces = parse_search_spaces(search_spaces_str)
    optimizers = search_spaces.keys()

    with open("data/global_settings.json", "r") as json_file:
        global_settings = json.load(json_file)

    # print(global_settings)
    print("Settings:")
    for k, v in global_settings.items():
        print(f"{k}: {v}")

    # Base configuration for all runs
    run_args_base: dict[str, Any] = dict(
        scoring_function_name=None,
        optimizer_name=None,
        optimizer_args=None,
        memory_score_threshold=global_settings[
            "score_threshold"
        ],  # score threshold is fixed at 0.5. That corresponds to DF on.
        memory_distance_threshold=global_settings["distance_threshold"],  # distance threshold is fixed at 0.7
        time_budget=global_settings["time_budget"],
        sample_budget=global_settings["sample_budget"],
        use_property_constraints=True,
        memory_known_active_init=False,  # no usage of known actives for DF
        n_jobs_scoring_function=global_settings["n_jobs"],
    )

    # create the run directories
    np.random.seed(seed=args.seed)
    for i, (task, optimizer_name) in enumerate(product(tasks, optimizers)):
        for j in range(args.num_trials):
            optimizer_args = get_complete_parameters(optimizer_name)

            run_args = run_args_base.copy()
            run_args["optimizer_name"] = optimizer_name
            run_args["optimizer_args"] = optimizer_args
            run_args["scoring_function_name"] = task

            # disable DF for gflownet as it should create diverse molecules by itself
            if optimizer_name == "Gflownet":
                run_args["memory_score_threshold"] = 5.0

            # gflownet is uses multiprocessing itself
            if optimizer_name.startswith("Gflownet"):
                run_args["n_jobs_scoring_function"] = 1

            run_dir = os.path.join(hyperpar_dir, f"{task}_{optimizer_name}_{j}")
            if os.path.exists(run_dir):
                print(f"{run_dir} already exists")
                continue

            for v in run_args.values():
                assert v is not None, f"None value in {run_args}"

            if os.path.exists(run_dir):
                print(f"{run_dir} already exists")
                continue

            print("Creating run dir:", run_dir)
            os.makedirs(run_dir)

            # dump the run_args as config.json
            with open(os.path.join(run_dir, "config.json"), "w") as json_file:
                json.dump(run_args, json_file, indent=4)
