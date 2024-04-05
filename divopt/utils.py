import os
from multiprocessing import Pool
from typing import List

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors  # type: ignore
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.QED import qed


def filterdf(df: pd.DataFrame, d: dict) -> pd.DataFrame:
    """Filter a dataframe for rows that match a dictionary of column names and values.

    Args:
        df (pd.DataFrame): Dataframe to filter
        d (dict): Dictionary of column names and values to filter for

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    idx = np.ones(len(df), dtype=bool)
    for name, value in d.items():
        if isinstance(value, list):
            new_idx = df[name].isin(value)
        else:
            new_idx = df[name] == value
        idx = np.logical_and(idx, new_idx)
    return df[idx]


def list_cache(function):
    cache = {}

    def wrapped_function(input_list: List):
        not_in_cache_list = [x for x in input_list if x not in cache]
        if len(not_in_cache_list) > 0:
            new_results = function(not_in_cache_list)
            cache.update(dict(zip(not_in_cache_list, new_results)))
        return [cache[x] for x in input_list]

    return wrapped_function


def pool_wrapper(function, n_jobs=8):
    def wrapped_function(input_list: List):
        nonlocal n_jobs
        n_jobs = min(n_jobs, len(input_list))
        with Pool(n_jobs) as p:
            return p.map(function, input_list)

    return wrapped_function


def calculate_properties(smiles):
    props = {}
    mol = Chem.MolFromSmiles(smiles)
    props["QED"] = qed(mol)
    props["MW"] = ExactMolWt(mol)
    props["LogP"] = MolLogP(mol)
    # morgan fingerprint from rdkit
    fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
    props["ECFP4bits"] = set(fp.GetNonzeroElements().keys())
    return props


def add_memory_names(results: pd.DataFrame) -> pd.DataFrame:
    """Add memory names to a dataframe

    Args:
        results (pd.DataFrame): Dataframe with memory score threshold and memory known active init columns

    Returns:
        pd.DataFrame: Dataframe with memory names added
    """
    memory_string_1 = ["-" if mst >= 1e14 else "Div" for mst in results["memory_score_threshold"]]
    memory_string_2 = [
        "NovDiv" if (m == "Div" and init) else m
        for m, init in zip(memory_string_1, results["memory_known_active_init"])
    ]
    results["Memory"] = memory_string_2
    return results


def add_virtual_screening_names(results: pd.DataFrame) -> pd.DataFrame:
    fname_to_label = {
        "chembl_33_maxmin_order.smiles": "MaxMin",
        "guacamol_v1_all.smiles": "",
        "chembl_33_random_order.smiles": "",
        "guacamol_v1_all_maxmin_order.smiles": "MaxMin",
    }

    suffix_list = []
    for row in results.iterrows():
        if row[1]["optimizer_name"] == "VS":
            fname = os.path.basename(row[1]["optimizer_args"]["smiles_file"])
            suffix_list.append(fname_to_label[fname])
        else:
            suffix_list.append("")

    results["Optimizer"] = [" ".join([a, b]) for a, b in zip(results["optimizer_name"], suffix_list)]
    return results


def passes_threshold(a, b, distance_threshold):
    d = 1 - DataStructs.FingerprintSimilarity(a, b)
    return d > distance_threshold


def get_run_args_base(global_settings):
    run_args_base = dict(
        scoring_function_name=None,
        optimizer_name=None,
        optimizer_args=None,
        memory_score_threshold=global_settings[
            "score_threshold"
        ],  # score threshold is fixed at 0.5. That corresponds to DF on.
        memory_distance_threshold=global_settings["distance_threshold"],  # distance threshold is fixed at 0.7
        time_budget=global_settings["time_budget"],
        use_property_constraints=global_settings.get("use_property_constraints", True),
        memory_known_active_init=global_settings.get(
            "memory_known_active_init", False
        ),  # no usage of known actives for DF
        n_jobs_scoring_function=global_settings["n_jobs"],
    )
    return run_args_base
