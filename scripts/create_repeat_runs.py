import argparse
import os
import shutil

from divopt.evaluation.process_results import load_results_dir


def create_repeat_runs(runs_base: str, num_repeats, dry_run=False) -> None:
    hyperparameter_dir = os.path.join(runs_base, "hyperparameter_search")
    print(f"Loading results from {hyperparameter_dir}")
    results = load_results_dir(hyperparameter_dir)

    limit_names = ["time", "samples"]

    for constraint_name in limit_names:
        # identify the best settings for each task and optimizer
        idx = results.groupby(["Task", "Optimizer"])[
            f"n_diverse_all_{constraint_name}"
        ].idxmax()
        best_settings = results.loc[idx]

        best_var_dir = os.path.join(runs_base, f"best_variance_{constraint_name}/")
        for config_fname in best_settings["config_fname"]:
            config_dir_name = os.path.dirname(config_fname)
            run_name = os.path.basename(config_dir_name)
            for i in range(num_repeats):
                config_dir_name_new = os.path.join(best_var_dir, f"{run_name}_{i}")
                if os.path.exists(config_dir_name_new):
                    print(f"Directory {config_dir_name_new} already exists, skipping")
                    continue
                if i == 0:
                    print(
                        f"Copying whole directory: {config_dir_name} -> {config_dir_name_new}"
                    )
                    if not dry_run:
                        shutil.copytree(config_dir_name, config_dir_name_new)
                else:
                    print(
                        f"Copying config file: {config_dir_name} -> {config_dir_name_new}"
                    )
                    if not dry_run:
                        os.makedirs(config_dir_name_new, exist_ok=True)
                        shutil.copy(
                            os.path.join(config_dir_name, "config.json"),
                            config_dir_name_new,
                        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--runs_base",
        type=str,
        help="Base directory for runs. Usually ./runs",
    )
    argparser.add_argument(
        "--num_repeats",
        type=int,
        default=3,
        help="Number of repeats to create",
    )
    argparser.add_argument(
        "--dry_run",
        action="store_true",
        help="If true, don't actually copy anything",
    )

    args = argparser.parse_args()

    create_repeat_runs(
        runs_base=args.runs_base, num_repeats=args.num_repeats, dry_run=args.dry_run
    )
