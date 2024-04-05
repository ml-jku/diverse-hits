# copied and adapted from
# https://github.com/wenhao-gao/mol_opt/blob/main/main/stoned/run.py

import numpy as np
import selfies

# from main.optimizer import BaseOptimizer
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from selfies import decoder, encoder
from typing import List, Optional
from guacamol.scoring_function import ScoringFunction
from guacamol.goal_directed_generator import GoalDirectedGenerator

RDLogger.DisableLog("rdApp.*")


def get_ECFP4(mol):
    """Return rdkit ECFP4 fingerprint object for mol

    Parameters:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object

    Returns:
    rdkit ECFP4 fingerprint object for mol
    """
    return AllChem.GetMorganFingerprint(mol, 2)


def sanitize_smiles(smi):
    """Return a canonical smile representation of smi

    Parameters:
    smi (string) : smile string to be canonicalized

    Returns:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object                          (None if invalid smile string smi)
    smi_canon (string)          : Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): True/False to indicate if conversion was  successful
    """
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)


def mutate_selfie(selfie, max_molecules_len, write_fail_cases=False):
    """Return a mutated selfie string (only one mutation on slefie is performed)

    Mutations are done until a valid molecule is obtained
    Rules of mutation: With a 50% propbabily, either:
        1. Add a random SELFIE character in the string
        2. Replace a random SELFIE character with another

    Parameters:
    selfie            (string)  : SELFIE string to be mutated
    max_molecules_len (int)     : Mutations of SELFIE string are allowed up to this length
    write_fail_cases  (bool)    : If true, failed mutations are recorded in "selfie_failure_cases.txt"

    Returns:
    selfie_mutated    (string)  : Mutated SELFIE string
    smiles_canon      (string)  : canonical smile of mutated SELFIE string
    """
    valid = False
    fail_counter = 0
    chars_selfie = get_selfie_chars(selfie)

    while not valid:
        fail_counter += 1

        alphabet = list(selfies.get_semantic_robust_alphabet())  # 34 SELFIE characters

        choice_ls = [1, 2]  # 1=Insert; 2=Replace; 3=Delete
        random_choice = np.random.choice(choice_ls, 1)[0]

        # Insert a character in a Random Location
        if random_choice == 1:
            random_index = np.random.randint(len(chars_selfie) + 1)
            random_character = np.random.choice(alphabet, size=1)[0]

            selfie_mutated_chars = (
                chars_selfie[:random_index]
                + [random_character]
                + chars_selfie[random_index:]
            )

        # Replace a random character
        elif random_choice == 2:
            random_index = np.random.randint(len(chars_selfie))
            random_character = np.random.choice(alphabet, size=1)[0]
            if random_index == 0:
                selfie_mutated_chars = [random_character] + chars_selfie[
                    random_index + 1 :
                ]
            else:
                selfie_mutated_chars = (
                    chars_selfie[:random_index]
                    + [random_character]
                    + chars_selfie[random_index + 1 :]
                )

        # Delete a random character
        elif random_choice == 3:
            random_index = np.random.randint(len(chars_selfie))
            if random_index == 0:
                selfie_mutated_chars = chars_selfie[random_index + 1 :]
            else:
                selfie_mutated_chars = (
                    chars_selfie[:random_index] + chars_selfie[random_index + 1 :]
                )

        else:
            raise Exception("Invalid Operation trying to be performed")

        selfie_mutated = "".join(x for x in selfie_mutated_chars)
        sf = "".join(x for x in chars_selfie)

        try:
            smiles = decoder(selfie_mutated)
            mol, smiles_canon, done = sanitize_smiles(smiles)
            if len(selfie_mutated_chars) > max_molecules_len or smiles_canon == "":
                done = False
            if done:
                valid = True
            else:
                valid = False
        except:
            valid = False
            if fail_counter > 1 and write_fail_cases == True:
                f = open("selfie_failure_cases.txt", "a+")
                f.write(
                    "Tried to mutate SELFIE: "
                    + str(sf)
                    + " To Obtain: "
                    + str(selfie_mutated)
                    + "\n"
                )
                f.close()

    return (selfie_mutated, smiles_canon)


def get_selfie_chars(selfie):
    """Obtain a list of all selfie characters in string selfie

    Parameters:
    selfie (string) : A selfie string - representing a molecule

    Example:
    >>> get_selfie_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
    ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']

    Returns:
    chars_selfie: list of selfie characters present in molecule selfie
    """
    chars_selfie = []  # A list of all SELFIE sybols from string selfie
    while selfie != "":
        chars_selfie.append(selfie[selfie.find("[") : selfie.find("]") + 1])
        selfie = selfie[selfie.find("]") + 1 :]
    return chars_selfie


class StonedOptimizer(GoalDirectedGenerator):
    def __init__(
        self,
        generation_size: int = 500,
        smi_file: str = "../data/guacamol_v1_all.smiles",
        iterations: int = int(1e15),
    ):
        self.model_name = "stoned"
        self.generation_size = generation_size
        with open(smi_file, "r") as f:
            self.all_smiles = f.read().split("\n")
        self.iterations = iterations

    def generate_optimized_molecules(
        self,
        scoring_function: ScoringFunction,
        number_molecules: int,
        starting_population: Optional[List[str]] = None,
    ) -> List[str]:
        population = np.random.choice(
            self.all_smiles, size=self.generation_size
        ).tolist()
        population = [encoder(smi) for smi in population]
        len_random_struct = max([len(get_selfie_chars(s)) for s in population])

        for _ in range(self.iterations):
            fitness = scoring_function.score_list([decoder(i) for i in population])

            #    Step 1: Keep the best molecule:  Keep the best member & mutate the rest
            print("Best fitness: ", np.max(fitness))
            best_idx = np.argmax(fitness)
            best_selfie = population[best_idx]

            #    Step 2: Get mutated selfies
            new_population = []
            for _ in range(self.generation_size - 1):
                selfie_mutated, _ = mutate_selfie(
                    best_selfie, len_random_struct, write_fail_cases=True
                )
                new_population.append(selfie_mutated)
            new_population.append(best_selfie)

            # Define new population for the next generation
            population = new_population[:]

        # sort population by fitness and return the top number_molecules
        fitness = scoring_function.score_list([decoder(i) for i in population])
        population_sort = [x for _, x in sorted(zip(fitness, population), reverse=True)]
        return population_sort[:number_molecules]
