from typing import List, Optional

import numpy as np
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.scoring_function import ScoringFunction
from tqdm import tqdm


class VSOptimizer(GoalDirectedGenerator):
    """
    Goal-directed molecule generator that will look through the molecules in a file.
    """

    def __init__(self, smiles_file: str, batch_size: int, shuffle=False) -> None:
        self.shuffle = shuffle    
        self.batch_size = batch_size
        
        with open(smiles_file) as f:
            self.smiles = f.read().split()
            
        if self.shuffle:
            np.random.shuffle(self.smiles)
        

    def generate_optimized_molecules(
        self,
        scoring_function: ScoringFunction,
        number_molecules: int,
        starting_population: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Will iterate through the reference set of SMILES strings and select the best molecules.
        """
        library_size = len(self.smiles)
        n_batches = (library_size // self.batch_size) + 1

        scores = []
        for i in tqdm(range(n_batches)):
            smiles_batch = self.smiles[i * self.batch_size : (i + 1) * self.batch_size]
            scores_batch = scoring_function.score_list(smiles_batch)
            scores += scores_batch

        top_k_idx = np.argpartition(scores, -number_molecules)[-number_molecules:]
        return np.array(self.smiles)[top_k_idx].tolist()
