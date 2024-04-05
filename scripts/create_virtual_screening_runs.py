import argparse
import json
import os
from copy import deepcopy

from divopt.utils import get_run_args_base

divoptpath = os.environ.get("DIVOPTPATH")
if divoptpath is None:
    raise ValueError("DIVOPTPATH environment variable is not set")

arg_path = os.path.join(divoptpath, "data/optimizer_default_args.json")
with open(arg_path, "r") as f:
    optimizer_args_defaults = json.load(f)

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--runs_base",
    type=str,
    help="Base directory for runs. Usually ./runs",
)
argparser.add_argument(
    "--num_repeats",
    type=int,
    default=5,
    help="Number of trials to create",
)
# sample invocation:
# python scripts/create_virtual_screening_runs.py --runs_base ./runs --num_repeats 5


args = argparser.parse_args()
# virtual_screening_dir = "../runs/virtual_screening"
virtual_screening_dir = os.path.join(args.runs_base, "virtual_screening")
tasks = ["GSK3", "JNK3", "DRD2"]


with open("data/global_settings.json", "r") as json_file:
    global_settings = json.load(json_file)

print(global_settings)
run_args_base = get_run_args_base(global_settings)

# random order virtual screening
optimizer_args = deepcopy(optimizer_args_defaults["VS"])
optimizer_args["smiles_file"] = "./data/guacamol_v1_all.smiles"
optimizer_args["shuffle"] = True

run_args = run_args_base.copy()
run_args["optimizer_name"] = "VS"
run_args["optimizer_args"] = optimizer_args

for task in tasks:
    run_args["scoring_function_name"] = task
    for i in range(args.num_repeats):
        config_dir_name_new = os.path.join(virtual_screening_dir, f"{task}_VSRandom_guacamol_{i}")
        os.makedirs(config_dir_name_new, exist_ok=True)

        with open(os.path.join(config_dir_name_new, "config.json"), "w") as f:
            json.dump(run_args, f, indent=4)


# maxmin order virtual screening
optimizer_args["smiles_file"] = "./data/guacamol_v1_all_maxmin_order.smiles"
optimizer_args["shuffle"] = False
run_args["optimizer_args"] = optimizer_args

for task in tasks:
    run_args["scoring_function_name"] = task
    config_dir_name_new = os.path.join(virtual_screening_dir, f"{task}_VSMaxMin_guacamol_0")
    os.makedirs(config_dir_name_new, exist_ok=True)
    with open(os.path.join(config_dir_name_new, "config.json"), "w") as f:
        json.dump(run_args, f, indent=4)
