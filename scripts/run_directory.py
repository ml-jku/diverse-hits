import argparse
import glob
import json
import os
import subprocess
import sys
import traceback

parser = argparse.ArgumentParser(description="Run optimization for all directories in a given directory")

# Add the settings argument
parser.add_argument(
    "--base_dir",
    type=str,
    required=True,  # Make it mandatory
    help="Path to folder containing run definitions for hyperparameter optimization",
)

# Parse the arguments
args = parser.parse_args()
base_dir = args.base_dir

if not os.path.isdir(base_dir):
    print(f"'{base_dir}' is not a valid directory.")
    sys.exit(1)


def get_run_dirs(base_dir):
    config_files = glob.glob(os.path.join(base_dir, "*", "config.json"))
    run_dirs = sorted([os.path.dirname(config_file) for config_file in config_files])
    run_dirs = [run_dir for run_dir in run_dirs if not os.path.exists(os.path.join(run_dir, "results.csv"))]
    run_dirs = [run_dir for run_dir in run_dirs if not os.path.exists(os.path.join(run_dir, "lock"))]
    # sort runs by last character
    run_dirs = sorted(run_dirs, key=lambda x: int(x.split("_")[-1]))
    return run_dirs


def lock_run_dir(run_dir) -> None:
    lock_file = os.path.join(run_dir, "lock")
    with open(lock_file, "w") as _:
        pass


def remove_lock(run_dir):
    lock_file = os.path.join(run_dir, "lock")
    if os.path.exists(lock_file):
        os.remove(lock_file)


def is_locked(run_dir) -> bool:
    lock_file = os.path.join(run_dir, "lock")
    return os.path.exists(lock_file)


run_dirs = get_run_dirs(base_dir)
print("Found {} runs to do".format(len(run_dirs)))
for run_dir in run_dirs[:10]:
    print(run_dir)

for run_dir in run_dirs:
    # create lock file and skip if it already exists
    if is_locked(run_dir):
        continue

    lock_run_dir(run_dir)

    print('Starting run "{}"'.format(run_dir))
    config_path = os.path.join(run_dir, "config.json")

    with open(config_path) as f:
        config = json.load(f)

    optimizer_name = config["optimizer_name"]
    env_map = {"Gflownet": "gflownet", "GflownetDF": "gflownet"}
    env_name = env_map.get(optimizer_name, "divopt")

    cmd = [
        "conda",
        "run",
        "-n",
        env_name,
        "python",
        "scripts/run_single.py",
        "--config",
        config_path,
    ]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Cleaning up and exiting...")
        sys.exit()
    except Exception:
        traceback.print_exc(file=sys.stdout)
    finally:
        remove_lock(run_dir)
