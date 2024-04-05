import argparse
import os
from pprint import pprint


def print_args(**kwargs):
    pprint(kwargs)


parser = argparse.ArgumentParser()
parser.add_argument("--smiles_file", default="data/guacamol_v1_all.smiles")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--population_size", type=int, default=100)
parser.add_argument("--n_mutations", type=int, default=200)
parser.add_argument("--gene_size", type=int, default=300)
parser.add_argument("--generations", type=int, default=1000)
parser.add_argument("--n_jobs", type=int, default=-1)
parser.add_argument("--random_start", action="store_true")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--suite", default="v2")

args = parser.parse_args()

optimiser = print_args(
    smi_file=args.smiles_file,
    population_size=args.population_size,
    n_mutations=args.n_mutations,
    gene_size=args.gene_size,
    generations=args.generations,
    n_jobs=args.n_jobs,
    random_start=args.random_start,
    patience=args.patience,
)

args = parser.parse_args()
