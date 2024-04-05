import json
from functools import lru_cache, partial
from multiprocessing import Pool
from typing import List, Optional, Set, Tuple

import numpy as np
from rdkit import Chem  # type: ignore
from rdkit.Chem import rdMolDescriptors  # type: ignore
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import ExactMolWt  # type: ignore
from rdkit.DataStructs.cDataStructs import ExplicitBitVect


def ebv2np(ebv: Optional[ExplicitBitVect]) -> Optional[np.ndarray]:
    """Explicit bit vector returned by rdkit to numpy array. Faster than just calling np.array(ebv)"""
    if ebv is None:
        return None
    return (np.frombuffer(bytes(ebv.ToBitString(), "utf-8"), "u1") - ord("0")).astype(bool)  # type: ignore


def ebv2int(ebv: ExplicitBitVect) -> int:
    """
    Converts an ExplicitBitVect object to an integer.

    Parameters:
        ebv (ExplicitBitVect): The ExplicitBitVect object to be converted.

    Returns:
        int: The integer representation of the ExplicitBitVect object.
    """
    return int(ebv.ToBitString(), base=2)


def _morgan_from_smiles(smiles: Optional[str], radius: int = 2, nbits: int = 2048) -> Optional[ExplicitBitVect]:
    """Generates a Morgan/ECFP fingerprint from a smiles string.

    Returns:
        ExplicitBitVect | None: The fingerprint or None
    """
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)  # type: ignore
    if mol is None:
        return None
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nbits)


@lru_cache(maxsize=100_000)
def morgan_from_smiles(smiles: Optional[str], radius: int = 2, nbits: int = 2048) -> Optional[ExplicitBitVect]:
    """Generates a Morgan/ECFP fingerprint from a smiles string.

    Returns:
        np.ndarray | None: The fingerprint or None
    """
    return _morgan_from_smiles(smiles, radius=radius, nbits=nbits)


def morgan_from_smiles_list(
    smiles: List[str],
    radius: int = 2,
    nbits: int = 2048,
    n_jobs: Optional[int] = None,
) -> List[Optional[ExplicitBitVect]]:
    """Generate morgan fingerprints for a list of smiles

    Args:
        smiles (List[str]): List of smiles
        radius (int, optional): Radius for Morgan fingerprint. Defaults to 2.
        nbits (int, optional): Number of bits for Morgan fingerprint. Defaults to 2048.
        n_jobs (int, optional): Number of processes to use for multiprocessing. Defaults to None.

    Returns:
        List[np.ndarray | None]: List of fingerprints or None for invalid smiles
    """
    if n_jobs is None:
        fps = [morgan_from_smiles(s, radius=radius, nbits=nbits) for s in smiles]
    else:
        with Pool(n_jobs) as p:
            fps = p.map(partial(_morgan_from_smiles, radius=radius, nbits=nbits), smiles)

    return fps


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Cached version of canonicalization"""
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)  # type: ignore
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)  # type: ignore


# ------------ cached version of canonicalization and fingerprint calculation ------------
class CachedCalculator:
    """Compute a function using multiprocessing and cache results"""

    def __init__(self, single_function, n_jobs=8):
        self.cache = {}
        self.n_jobs = n_jobs
        self.single_function = single_function

    def process_list(self, input_list):
        not_in_cache = [x for x in input_list if x not in self.cache]
        if len(not_in_cache) == 0:
            return [self.cache[x] for x in input_list]

        n_jobs = min(self.n_jobs, len(not_in_cache))
        if self.n_jobs == 1:
            new_values = [self.single_function(x) for x in not_in_cache]
        else:
            with Pool(n_jobs) as p:
                new_values = p.map(self.single_function, not_in_cache)

        self.cache.update(dict(zip(not_in_cache, new_values)))
        return [self.cache[x] for x in input_list]


class FingerprintCalculator(CachedCalculator):
    def __init__(self, n_jobs: int=8):
        super().__init__(_morgan_from_smiles, n_jobs=n_jobs)


class Canonicalizer(CachedCalculator):
    """Canonicalize molecules using multiprocessing and cache results"""

    def __init__(self, n_jobs: int=8):
        super().__init__(canonicalize_smiles, n_jobs=n_jobs)


@lru_cache(maxsize=1_000_000)
def calculate_properties(smiles) -> Tuple[float, float, Set[int]]:
    mol = Chem.MolFromSmiles(smiles)  # type: ignore
    mw = ExactMolWt(mol)
    logp = MolLogP(mol)

    fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
    ecfp4bits = set(fp.GetNonzeroElements().keys())
    return mw, logp, ecfp4bits


class PropertyCalculator(CachedCalculator):
    def __init__(self, n_jobs=8):
        super().__init__(calculate_properties, n_jobs=n_jobs)


class ExoticFingerprintRatio:
    def __init__(self, known_bits_fn: str):
        with open(known_bits_fn, "r") as f:
            self.known_bits = set(json.load(f))

    def get_ratio(self, fp_bits):
        if len(fp_bits) == 0:
            return 0.0
        return len(self.known_bits.intersection(fp_bits)) / len(fp_bits)
