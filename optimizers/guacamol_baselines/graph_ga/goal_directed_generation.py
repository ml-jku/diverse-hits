from __future__ import print_function

import argparse
import heapq
import json
import os
import random
from time import time
from typing import List, Optional

import joblib
import numpy as np

from joblib import delayed
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from optimizers.guacamol_baselines.graph_ga import crossover as co
from optimizers.guacamol_baselines.graph_ga import mutate as mu

def canonicalize(smiles: str, include_stereocenters=True) -> Optional[str]:
    """
    Canonicalize the SMILES strings with RDKit.

    The algorithm is detailed under https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00543

    Args:
        smiles: SMILES string to canonicalize
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES string

    Returns:
        Canonicalized SMILES string, None if the molecule is invalid.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)
    else:
        return None


def mols2smiles(mols):
    return [Chem.MolToSmiles(m) for m in mols]


def make_mating_pool(population_mol: List[Mol], population_scores, offspring_size: int):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights

    Args:
        population_mol: list of RDKit Mol
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return

    Returns: a list of RDKit Mol (probably not unique)

    """
    # scores -> probs
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    mating_pool = np.random.choice(
        population_mol, p=population_probs, size=offspring_size, replace=True
    )
    return mating_pool


def reproduce(mating_pool, mutation_rate):
    """

    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation

    Returns:

    """
    parent_a = random.choice(mating_pool)
    parent_b = random.choice(mating_pool)
    new_child = co.crossover(parent_a, parent_b)
    if new_child is not None:
        new_child = mu.mutate(new_child, mutation_rate)
    return new_child


def score_mol(mol, score_fn):
    return score_fn(Chem.MolToSmiles(mol))


def sanitize(population_mol):
    new_population = []
    smile_set = set()
    for mol in population_mol:
        if mol is not None:
            try:
                smile = Chem.MolToSmiles(mol)
                if smile is not None and smile not in smile_set:
                    smile_set.add(smile)
                    new_population.append(mol)
            except ValueError:
                print("bad smiles")
    return new_population


class GB_GA_Generator():
    def __init__(
        self,
        smi_file,
        population_size,
        offspring_size,
        generations,
        mutation_rate,
        n_jobs=-1,
        random_start=False,
        patience=5,
        canonicalize=True,
    ):
        self.pool = joblib.Parallel(n_jobs=n_jobs)
        self.smi_file = smi_file
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.random_start = random_start
        self.patience = patience
        self.canonicalize = canonicalize
        self.all_smiles = self.load_smiles_from_file(self.smi_file)

    def load_smiles_from_file(self, smi_file):
        with open(smi_file) as f:
            if self.canonicalize:
                return self.pool(delayed(canonicalize)(s.strip()) for s in f)
            else:
                return f.read().split()

    def top_k(self, smiles, scoring_function, k):
        scores = scoring_function.score_list(smiles)
        scored_smiles = list(zip(scores, smiles))
        scored_smiles = sorted(scored_smiles, key=lambda x: x[0], reverse=True)
        return [smile for score, smile in scored_smiles][:k]

    def generate_optimized_molecules(
        self,
        scoring_function,
        get_history=False,
    ) -> List[str]:

        if self.random_start:
            starting_population = np.random.choice(
                self.all_smiles, self.population_size
            )
        else:
            starting_population = self.top_k(
                self.all_smiles, scoring_function, self.population_size
            )

        # select initial population
        # this is also slow
        # population_smiles = heapq.nlargest(self.population_size, starting_population, key=scoring_function.score)
        starting_scores = scoring_function.score_list(starting_population)
        population_smiles = [
            x
            for _, x in sorted(
                zip(starting_scores, starting_population),
                key=lambda pair: pair[0],
                reverse=True,
            )
        ]

        population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]

        # this is slow. Don't know exactly why. maybe pickling classifiers is not too nice
        # population_scores_old = self.pool(delayed(score_mol)(m, scoring_function.score) for m in population_mol)
        population_scores = scoring_function.score_list(mols2smiles(population_mol))

        # evolution: go go go!!
        t0 = time()

        patience = 0

        for generation in range(self.generations):
            # new_population
            mating_pool = make_mating_pool(
                population_mol, population_scores, self.offspring_size
            )
            offspring_mol = self.pool(
                delayed(reproduce)(mating_pool, self.mutation_rate)
                for _ in range(self.population_size)
            )

            # add new_population
            population_mol += offspring_mol
            population_mol = sanitize(population_mol)

            # stats
            gen_time = time() - t0
            mol_sec = self.population_size / gen_time
            t0 = time()

            old_scores = population_scores
            # population_scores = self.pool(delayed(score_mol)(m, scoring_function.score) for m in population_mol)
            population_scores = scoring_function.score_list(
                [Chem.MolToSmiles(m) for m in population_mol]
            )
            population_tuples = list(zip(population_scores, population_mol))
            population_tuples = sorted(
                population_tuples, key=lambda x: x[0], reverse=True
            )[: self.population_size]
            population_mol = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]

            # early stopping
            if population_scores == old_scores:
                patience += 1
                print(f"Failed to progress: {patience}")
                if patience >= self.patience:
                    print(f"No more patience, bailing...")
                    break
            else:
                patience = 0

            res_time = time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_file", default="data/guacamol_v1_all.smiles")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--population_size", type=int, default=100)
    parser.add_argument("--offspring_size", type=int, default=200)
    parser.add_argument("--mutation_rate", type=float, default=0.01)
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--random_start", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--suite", default="v2")

    args = parser.parse_args()

    np.random.seed(args.seed)

    setup_default_logger()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.realpath(__file__))

    # save command line args
    with open(os.path.join(args.output_dir, "goal_directed_params.json"), "w") as jf:
        json.dump(vars(args), jf, sort_keys=True, indent=4)

    optimiser = GB_GA_Generator(
        smi_file=args.smiles_file,
        population_size=args.population_size,
        offspring_size=args.offspring_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        n_jobs=args.n_jobs,
        random_start=args.random_start,
        patience=args.patience,
    )

    json_file_path = os.path.join(args.output_dir, "goal_directed_results.json")
    # skips annoying tensorflow import
    # assess_goal_directed_generation(
    #     optimiser, json_output_file=json_file_path, benchmark_version=args.suite
    # )


if __name__ == "__main__":
    main()
