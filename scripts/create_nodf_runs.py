import argparse
import json
import os


def create_nodf_runs(runs_base):
    best_dirs = [f"{runs_base}/best_variance_{limit_name}" for limit_name in ["samples", "time"]]

    for repeat_dir in best_dirs:
        nodf_dir = repeat_dir + "_nodf"
        for run_dir in os.listdir(repeat_dir):
            if not os.path.isdir(os.path.join(repeat_dir, run_dir)):
                continue
            
            run_dir_nodf = os.path.join(nodf_dir, run_dir)
            os.makedirs(run_dir_nodf, exist_ok=True)
            config_fname = os.path.join(repeat_dir, run_dir, "config.json")
            config_fname_nodf = os.path.join(run_dir_nodf, "config.json")
            
            with open(config_fname, "r") as f:
                config = json.load(f)
            
            # disable diversity filtering with score that is never reached
            config["memory_score_threshold"] = 5
            with open(config_fname_nodf, "w") as f:
                json.dump(config, f, indent=4)
                

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--runs_base",
        type=str,
        help="Base directory for runs. Usually ./runs",
    )

    args = argparser.parse_args()

    create_nodf_runs(args.runs_base)
