import glob
import json
import os
from copy import deepcopy
from functools import lru_cache
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from rdkit.DataStructs import ExplicitBitVect  # type: ignore

from divopt.chem import morgan_from_smiles
from divopt.evaluation.selection import se_algorithms
from divopt.evaluation.simgraph import compute_distance_matrix_from_fps, internal_diversity
from divopt.utils import filterdf


@lru_cache(maxsize=12)
def get_train_actives(scoring_function_name: str, basedir: str = "../data/scoring_functions"):
    """Get the train actives of a scoring function

    Args:
        scoring_function_name (str): The name of the scoring function
        basedir (str, optional): The base directory where the scoring functions are stored. 
            Defaults to "../data/scoring_functions".

    Returns:
        List[str]: The list of train actives
    """
    scoring_function_dir = os.path.join(basedir, scoring_function_name.lower())
    scoring_function_df = pd.read_csv(os.path.join(scoring_function_dir, "splits.csv"))
    train_actives = filterdf(scoring_function_df, {"Split": "train", "label": 1}).smiles.to_list()
    return train_actives


def calculate_novelty(
    reference_fps: List[ExplicitBitVect],
    query_fps: List[ExplicitBitVect],
    distance_threshold: float,
) -> List[int]:
    """
    Calculate the novelty of query fingerprints compared to reference fingerprints.

    Args:
        reference_fps (List[ExplicitBitVect]): List of reference fingerprints.
        query_fps (List[ExplicitBitVect]): List of query fingerprints.
        distance_threshold (float): Threshold for the distance matrix.

    Returns:
        List[int]: List of indices of query fingerprints that are considered novel.

    """
    distance_matrix = compute_distance_matrix_from_fps(reference_fps, query_fps, n_jobs=8)
    return (distance_matrix > distance_threshold).all(axis=0).nonzero()[0]


def compute_diverse_solutions(
    smiles: List[str],
    distance_threshold: float,
    radius: int = 2,
    nbits: int = 2048,
    algorithm: str = "maxmin_random",
    reference_smiles: List[str] = [],
) -> List[int]:
    """Compute diverse solutions from a list of smiles

    Args:
        smiles (List[str]): The list of smiles
        distance_threshold (float): The distance threshold
        radius (int, optional): The radius of the morgan fingerprint. Defaults to 2.
        nbits (int, optional): The number of bits of the morgan fingerprint. Defaults to 2048.
        algorithm (str, optional): The algorithm to use. Defaults to "maxmin_random".
        reference_smiles (List[str], optional): The reference smiles. Defaults to [].

    Returns:
        List[int]: The list of indices of the diverse solutions
    """
    if len(smiles) == 0:
        return []

    # gets train actives and computes which of the solutions are novel
    # the novel_idx is a list of indices of the novel solutions
    # we need them to map the indices of the diverse solutions to the original indices
    if len(reference_smiles) > 0:
        reference_fps = [morgan_from_smiles(smiles=s, radius=radius, nbits=nbits) for s in reference_smiles]
        fps = [morgan_from_smiles(smiles=s, radius=radius, nbits=nbits) for s in smiles]
        novel_idx = calculate_novelty(reference_fps, fps, distance_threshold)
        selection_input = np.array(smiles)[novel_idx].tolist()
    else:
        selection_input = smiles
        novel_idx = list(range(len(smiles)))

    if len(selection_input) == 0:
        return []

    # find the diverse solutions
    picks = se_algorithms[algorithm](
        selection_input,
        distance_threshold=distance_threshold,
        radius=radius,
        nbits=nbits,
    )
    return sorted([novel_idx[i] for i in picks])


def get_unprocessed_result_fnames(result_fnames):
    """
    Returns a list of unprocessed result filenames.

    Args:
        result_fnames (list): A list of result filenames.

    Returns:
        list: A list of unprocessed result filenames.

    """
    unprocessed = []
    for results_fname in result_fnames:
        results_dirname = os.path.dirname(results_fname)
        metrics_fname = os.path.join(results_dirname, "metrics.json")
        if not os.path.isfile(metrics_fname):
            unprocessed.append(results_fname)
    return unprocessed


def process_results_file(results_fname, root_dir: str = "./"):
    """
    Process the results file and calculate various metrics.

    Args:
        results_fname (str): The path to the results file.
        root_dir (str, optional): The root directory. Defaults to "./".

    Returns:
        dict: A dictionary containing the calculated metrics.

    Raises:
        NotImplementedError: If the limit_name is not "time" or "samples".
    """

    config_fname = os.path.join(os.path.dirname(results_fname), "config.json")
    with open(config_fname, "r") as f:
        config = json.load(f)
    task_name = config["scoring_function_name"]
    with open(os.path.join(root_dir, "data", "global_settings.json")) as f:
        global_settings = json.load(f)

    train_actives = get_train_actives(task_name, basedir=os.path.join(root_dir, "data", "scoring_functions"))
    reference_dict = {
        "all": [],
        "novel": train_actives,
    }

    metrics: Dict[str, Union[float, int]] = {}
    df = pd.read_csv(results_fname)
    df_valid = df[~pd.isna(df["CanSmiles"])]
    df_unique = deepcopy(df_valid.drop_duplicates(subset=["CanSmiles"], keep="first").reset_index(drop=True))
    df_unique["unique_idx"] = np.arange(len(df_unique))
    df_passing_score = df_unique[df_unique["Score"] >= global_settings["score_threshold"]]

    metrics["n_molecules_total"] = len(df)
    metrics["n_molecules_valid"] = len(df)

    if metrics["n_molecules_total"] == 0:
        metrics["valid_fraction"] = 0
    else:
        metrics["valid_fraction"] = metrics["n_molecules_valid"] / metrics["n_molecules_total"]
    metrics["n_molecules_unique"] = len(df_unique)

    for limit_name in ["time", "samples"]:
        # reduce by time or sample constraint
        if limit_name == "time":
            df_sub = df_passing_score[df_passing_score["Total time [s]"] < global_settings["time_budget"]]
        elif limit_name == "samples":
            df_sub = df_passing_score[df_passing_score["unique_idx"] <= global_settings["sample_budget"]]
        else:
            raise NotImplementedError

        # calculate the diverse sets and save them
        metrics[f"n_solutions_{limit_name}"] = len(df_passing_score)
        for init_name, reference_smiles in reference_dict.items():
            fname_diverse = results_fname.replace(".csv", f"_diverse_{init_name}_{limit_name}.csv")
            if os.path.isfile(fname_diverse):
                df_diverse = pd.read_csv(fname_diverse)
            else:
                picks = np.sort(
                    compute_diverse_solutions(
                        smiles=df_sub["CanSmiles"].values,
                        distance_threshold=global_settings["distance_threshold"],
                        algorithm="maxmin_random",
                        reference_smiles=reference_smiles,
                    )
                )
                df_diverse = df_sub.iloc[picks]
            metrics[f"n_diverse_{init_name}_{limit_name}"] = len(df_diverse)
            df_diverse.to_csv(fname_diverse, index=False)

        # calculate the internal diversity for both limits
        smiles_list = df_sub["CanSmiles"].values.tolist()
        # take at max 4000 samples for the internal diversity
        if len(smiles_list) > 4000:
            smiles_list = list(np.random.choice(smiles_list, size=4000, replace=False))

        if len(smiles_list) > 1:
            metrics[f"IntDiv_{limit_name}"] = internal_diversity(smiles_list, radius=2, nbits=2048, n_jobs=8)
        else:
            metrics[f"IntDiv_{limit_name}"] = 0

    metrics_fname = os.path.join(os.path.dirname(results_fname), "metrics.json")
    with open(metrics_fname, "w") as f:
        json.dump(metrics, f, indent=4)
    return metrics


def add_memory_names(results: pd.DataFrame) -> pd.DataFrame:
    """Add memory names to a dataframe. For example, if the memory score threshold is greater than or equal to 1e14,
    the memory name is "-". If the memory name is "Div" and the memory known active init is True,
    the memory name is "NovDiv".

    Args:
        results (pd.DataFrame): Dataframe with memory score threshold and memory known active init columns

    Returns:
        pd.DataFrame: Dataframe with memory names added
    """

    if len(results) == 0:
        raise ValueError("Results dataframe is empty")

    memory_string_1 = ["-" if mst >= 1e14 else "Div" for mst in results["memory_score_threshold"]]
    memory_string_2 = [
        "NovDiv" if (m == "Div" and init) else m
        for m, init in zip(memory_string_1, results["memory_known_active_init"])
    ]
    results["Memory"] = memory_string_2
    return results


def get_virtual_screening_name(smiles_file) -> str:
    """
    Returns the virtual screening name based on the given smiles file.

    Parameters:
        smiles_file (str): The path to the smiles file.

    Returns:
        str: The virtual screening name.
    """
    fname_to_label = {
        "guacamol_v1_all.smiles": "Random",
        "guacamol_v1_all_maxmin_order.smiles": "MaxMin",
    }
    fname = os.path.basename(smiles_file)
    return "VS" + fname_to_label[fname]


def add_virtual_screening_names(results: pd.DataFrame) -> pd.DataFrame:
    """
    Adds virtual screening names to the results DataFrame.

    Parameters:
    results (pd.DataFrame): The DataFrame containing the results.

    Returns:
    pd.DataFrame: The updated DataFrame with virtual screening names added.
    """
    augmented_names = []
    for _, row in results.iterrows():
        if row["optimizer_name"] == "VS":
            smiles_file = row["optimizer_args"]["smiles_file"]
            name = get_virtual_screening_name(smiles_file)
            augmented_names.append(name)
        else:
            augmented_names.append(row["optimizer_name"])

    results["Optimizer"] = augmented_names
    return results


def load_results_dir(dirname: str) -> pd.DataFrame:
    """
    Load results from a directory and return them as a pandas DataFrame.

    Args:
        dirname (str): The directory path containing the results.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded results.

    """
    # Find all config files in the directory and read them along with the metrics.
    config_pattern = os.path.join(dirname, "*/config.json")
    values_list = []
    for config_fname in glob.glob(config_pattern):
        with open(config_fname) as f:
            config = json.load(f)
        config["config_fname"] = config_fname
        config["results_fname"] = config_fname.replace("config.json", "results.csv")
        res_dir = os.path.dirname(config_fname)
        metrics_fname = os.path.join(res_dir, "metrics.json")
        if not os.path.isfile(metrics_fname):
            continue
        with open(metrics_fname) as f:
            metrics = json.load(f)

        config.update(metrics)
        values_list.append(config)

    # create a dataframe from the list of dictionaries
    results = pd.DataFrame(values_list)
    results = add_memory_names(results)
    results = add_virtual_screening_names(results)
    results = results.rename(columns={"scoring_function_name": "Task"})

    # Put some columns first
    first_columns = [
        "Task",
        "Optimizer",
        "Memory",
        "n_molecules_total",
        "valid_fraction",
        "n_molecules_unique",
        "n_solutions_time",
        "n_diverse_all_time",
        "n_diverse_novel_time",
        "n_solutions_samples",
        "n_diverse_all_samples",
        "n_diverse_novel_samples",
    ]

    results = results.sort_values(first_columns)
    results.reset_index(inplace=True, drop=True)
    old_columns = results.columns.tolist()
    new_columns = first_columns + [c for c in old_columns if c not in first_columns]
    results = results[new_columns]
    # Rename some optimizers for the paper
    new_names_dict = {
        "GA": "GraphGA",
        "BestAgentReminder": "BAR",
        "AugHC": "AugmentedHC",
    }
    results["Optimizer"] = [new_names_dict.get(opt_name, opt_name) for opt_name in results["Optimizer"]]
    new_task_names_dict = {"GSK3": "GSK3β"}
    results["Task"] = [new_task_names_dict.get(task_name, task_name) for task_name in results["Task"]]
    return results


def load_results(runs_base: str) -> Dict[str, pd.DataFrame]:
    """
    Load and process results from different directories.

    Args:
        runs_base (str): The base directory where the results are stored.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary containing the loaded and processed results.
            The keys of the dictionary represent the different limits ('samples' and 'time'),
            and the values are pandas DataFrames containing the results.

    """
    limit_to_dir = {"samples": "best_variance_samples", "time": "best_variance_time"}
    virtual_screening_dir = "virtual_screening"

    results_dict = {}
    for limit_name, limit_dir in limit_to_dir.items():
        res = load_results_dir(os.path.join(runs_base, limit_dir))
        vs_res = load_results_dir(os.path.join(runs_base, virtual_screening_dir))
        res = pd.concat([res, vs_res])
        results_dict[limit_name] = res

    return results_dict


def get_ranks(results, limit_name):
    """
    Calculate the ranks of the results based on the mean values of the specified limit.

    Parameters:
    results (DataFrame): The results DataFrame containing the data.
    limit_name (str): The name of the limit to calculate the ranks for.

    Returns:
    ranks (DataFrame): The DataFrame containing the ranks of the results.
    """
    means = results.pivot_table(index="Optimizer", columns="Task", values=f"n_diverse_all_{limit_name}", aggfunc="mean")
    ranks = means.rank(0, ascending=False)
    return ranks


def get_avg_rank(results, limit_name):
    """
    Calculate the average rank of the results based on the given limit name.

    Parameters:
    results (DataFrame): The results data.
    limit_name (str): The name of the limit.

    Returns:
    DataFrame: A DataFrame containing the average rank for each result.
    """
    return get_ranks(results, limit_name).mean(1).to_frame()


def get_sorted_optimizers(results, limit_name):
    """
    Returns a list of optimizers sorted based on their average rank.

    Parameters:
    results (DataFrame): The results of the optimization process.
    limit_name (str): The name of the limit to consider.

    Returns:
    list: A list of optimizers sorted based on their average rank.
    """
    avg_rank = get_avg_rank(results, limit_name)
    sorted_optimizers = avg_rank.sort_values(0).index
    return sorted_optimizers
