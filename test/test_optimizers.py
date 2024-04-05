import glob
import json
import os
import subprocess

import pytest

optimizer_names = [
    "Mimosa",
    "LSTM-HC",
    "LSTM-PPO",
    "GA",
    "GraphMCTS",
    "SmilesGA",
    "VS",
    "Stoned",
    "Reinvent",
    "Mars",
    "Gflownet",
    "GflownetDF",
    "AugHC",
    "BestAgentReminder",
    "AugMemory",
]


config_files = glob.glob("test/test_runs/*/config.json")


@pytest.mark.parametrize("config_path", config_files)
def test_optimizers(config_path):
    with open(config_path) as f:
        config = json.load(f)

    dirname = config_path.replace("config.json", "")
    result_files = [
        "results.csv",
        "results_diverse_all_time.csv",
        "results_diverse_all_samples.csv",
        "results_diverse_novel_samples.csv",
        "results_diverse_novel_time.csv",
        "metrics.json",
    ]
    result_files = [os.path.join(dirname, result_file) for result_file in result_files]

    for result_file in result_files:
        if os.path.exists(result_file):
            os.remove(result_file)

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

    subprocess.run(cmd)

    for result_file in result_files:
        print(result_file)
        assert os.path.exists(result_file)
