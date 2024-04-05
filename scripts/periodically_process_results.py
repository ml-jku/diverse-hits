import argparse
import glob
import os
from time import sleep

from divopt.evaluation.process_results import (
    get_unprocessed_result_fnames,
    process_results_file,
)

parser = argparse.ArgumentParser(
    description="Run optimization for all directories in a given directory"
)

# Add the settings argument
parser.add_argument(
    "--base_dir",
    type=str,
    required=True,  # Make it mandatory
    help="Path to folder containing model runs",
)
parser.add_argument(
    "--root_dir",
    type=str,
    required=True,  # Make it mandatory
    help="root dir of the divopt repository",
)
args = parser.parse_args()
base_dir = args.base_dir

while True:
    result_fnames = glob.glob(os.path.join(base_dir, "*/results.csv"))
    unprocessed = get_unprocessed_result_fnames(result_fnames)
    print(f"Found {len(unprocessed)}/{len(result_fnames)} unprocessed results files")
    if len(unprocessed) == 0:
        sleep(15)
        continue
    process_results_file(unprocessed[0], args.root_dir)
