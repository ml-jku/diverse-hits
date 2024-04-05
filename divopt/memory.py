from typing import List, Optional, Sequence, Set

from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from divopt.chem import FingerprintCalculator


def passes_threshold(a, b, distance_threshold):
    d = 1 - DataStructs.FingerprintSimilarity(a, b)
    return d >= distance_threshold


class MemoryUnit:
    """
    A simple memory unit/diversity filter that keeps track of generated molecules and promotes
    diverse generation.

    Args:
        distance_threshold (float): The distance threshold for determining if a compound is novel
        score_threshold (float): The score threshold that a compound needs to have to be a centroid.
        fingerprint_calculator: The fingerprint calculator object.

    Attributes:
        distance_threshold (float): The distance threshold for determining similarity between fingerprints.
        score_threshold (float): The score threshold for adding a fingerprint to memory.
        fp_memory_centroid (List[ExplicitBitVect]): The list of fingerprints stored in memory.
        smiles_memory_centroid (Set[Optional[str]]): The set of SMILES strings stored in memory.
        zero_score_cache (Set[Optional[str]]): The set of SMILES strings with a score of zero.
            A compound with a zero memory score will always have a score of zero, while a compound
            with a 1 may get a score of zero in the future.
        fingerprint_calculator: The fingerprint calculator object.

    Methods:
        score_and_update_memory(query_smiles, scores): Calculates the scores for the given query SMILES and 
        updates the memory. add_to_memory(smiles): Adds the given SMILES strings to memory.
    """

    def __init__(self, distance_threshold: float, score_threshold: float, n_jobs: int) -> None:
        self.distance_threshold = distance_threshold
        self.score_threshold = score_threshold
        self.fp_memory_centroid: List[ExplicitBitVect] = []
        self.smiles_memory_centroid: Set[Optional[str]] = set()
        self.zero_score_cache: Set[Optional[str]] = {None}
        self.fingerprint_calculator = FingerprintCalculator(n_jobs=n_jobs)

    def score_and_update_memory(self, query_smiles: Sequence[Optional[str]], scores: List[float]) -> List[int]:
        """
        Calculates the scores for the given query SMILES and updates the memory.

        Args:
            query_smiles (List[str]): The list of query SMILES.
            scores (List[float]): The list of scores corresponding to the query SMILES.

        Returns:
            List[int]: A list of binary values indicating whether each query SMILES is novel (1) or not (0).
        """
        if len(query_smiles) == 0:
            return []

        # We can already filter out the smiles that are already in the zero_score_cache
        # Smiles in the zero_score_cache are already proven to have a similar compound in the memory
        smiles_scores_to_evaluate = [
            (smile, score) for smile, score in zip(query_smiles, scores) if smile not in self.zero_score_cache
        ]

        # if nothing to evaluate, return all zeros
        if len(smiles_scores_to_evaluate) == 0:
            return [0 for _ in query_smiles]

        smiles_to_evaluate, scores_to_evaluate = zip(*smiles_scores_to_evaluate)

        query_fps = self.fingerprint_calculator.process_list(smiles_to_evaluate)
        assert all(fp is not None for fp in query_fps)

        # Check which smiles are novel. If novel and they pass the score threshold, add them to memory.
        # The region around them is now considered 'explored' and they will receive a score of zero.
        # If they are novel but do not pass the score threshold, they will receive a score of 1.
        # If they are not novel they receive a score of zero and are added to the zero_score_cache
        for query_fp, score, smiles in zip(query_fps, scores_to_evaluate, smiles_to_evaluate):
            is_novel = smiles not in self.zero_score_cache and all(
                passes_threshold(query_fp, m, self.distance_threshold) for m in self.fp_memory_centroid[::-1]
            )

            if is_novel:
                if score >= self.score_threshold:
                    self.fp_memory_centroid.append(query_fp)
                    self.smiles_memory_centroid.add(smiles)
                    # If a compound is added to memory, it will always have a score of zero
                    self.zero_score_cache.add(smiles)
            else:
                self.zero_score_cache.add(smiles)
        return [0 if s in self.zero_score_cache else 1 for s in query_smiles]

    def add_to_memory(self, smiles: List[str]) -> None:
        """Adds the given SMILES strings to memory."""
        fps = self.fingerprint_calculator.process_list(smiles)
        assert all(fp is not None for fp in fps)
        self.fp_memory_centroid.extend(fps)
        self.smiles_memory_centroid.update(smiles)
