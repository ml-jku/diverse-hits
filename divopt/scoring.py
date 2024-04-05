import json
import os
import pickle
from abc import abstractmethod
from time import time
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from divopt.chem import (
    Canonicalizer,
    ExoticFingerprintRatio,
    FingerprintCalculator,
    PropertyCalculator,
    ebv2np,
)
from divopt.memory import MemoryUnit
from divopt.utils import filterdf


class ScoringFunction(object):
    """
    Canonicalization should be done here for caching.
    """

    def __init__(
        self,
        time_budget: Optional[float],
        sample_budget: Optional[int],
        memory_distance_threshold: Optional[float],
        memory_score_threshold: Optional[float],
        invalid_score: float = 0.0,
        n_jobs: int = 8,
        print_progress: bool = True,
        samples_per_minute_threshold: int = 100,
        raise_timeout: bool = True,
    ):
        super().__init__()

        if time_budget is None and sample_budget is None:
            raise ValueError("Either time_budget or sample_budget must be set")

        assert samples_per_minute_threshold > 0
        assert n_jobs > 0
        assert invalid_score >= 0

        self.time_budget = time_budget
        self.sample_budget = sample_budget
        self.samples_per_minute_threshold = samples_per_minute_threshold
        self.raise_timeout = raise_timeout
        self.n_jobs = n_jobs
        self.invalid_score = invalid_score
        self.print_progress = print_progress

        self.canonicalizer = Canonicalizer(n_jobs=n_jobs)

        # both distance threshold and score must be set or not
        if (memory_distance_threshold is None) != (memory_score_threshold is None):
            raise ValueError("Both memory_distance_threshold and memory_score_threshold must be set or not")
        if memory_distance_threshold is not None and memory_score_threshold is not None:
            self.memory_unit: Optional[MemoryUnit]= MemoryUnit(
                distance_threshold=memory_distance_threshold,
                score_threshold=memory_score_threshold,
                n_jobs=n_jobs,
            )
        else:
            self.memory_unit = None

    def start_timer_and_reset(self) -> None:
        """
        Starts the timer and resets the necessary variables for scoring.

        This method initializes the necessary variables for scoring, including the history,
        start time, generation start time, cumulative score duration, cumulative generation duration,
        cumulative memory duration, loop count, score cache, and memory cache.

        Returns:
            None
        """
        self.history: Dict[str, List[Union[float, str]]] = {
            "CanSmiles": [],
            "Score": [],
            "Total time [s]": [],
            "Scoring time [s]": [],
            "Generation time [s]": [],
            "Memory time [s]": [],
        }

        self.start_time = time()
        self.gen_start = self.start_time

        self.cum_times: dict[str, float] = {
            "Generation": 0,
            "Scoring": 0,
            "Memory": 0,
            "Total": 0,
        }

        self.loop = 0
        self.score_cache: Dict[Optional[str], float] = {None: self.invalid_score}

    def _finished(self) -> bool:
        """
        Checks if the search process is finished based on the time and sample budgets.

        Returns:
            bool: True if the search process is finished, False otherwise.
        """
        time_passed = time() - self.start_time
        n_unique_smiles = len(set(self.history["CanSmiles"]))

        if self.time_budget is None:
            time_budget_exhausted = True
        else:
            time_budget_exhausted = time_passed > self.time_budget

        if self.sample_budget is None:
            sample_budget_exhausted = True
        else:
            sample_budget_exhausted = n_unique_smiles >= self.sample_budget

        finished = time_budget_exhausted and sample_budget_exhausted

        # Some methods fail to generate more unique samples after some time
        # So if the time budget is exhausted but the sample budget is not, we still want to finish
        # if the number of unique samples per minute gets too low. Otherwise the method may run forever
        unique_samples_per_minute = n_unique_smiles / time_passed * 60
        too_slow = unique_samples_per_minute < self.samples_per_minute_threshold
        return finished or (too_slow and time_budget_exhausted)

    def get_history(self) -> pd.DataFrame:
        """
        Returns the history of the scoring object as a pandas DataFrame.

        Returns:
            pd.DataFrame: The history of the scoring object.
        """
        df = pd.DataFrame(self.history)
        return df

    def _track_progress(
        self,
        can_smiles_list,
        property_scores,
        gen_duration,
        score_duration,
        memory_duration,
        generation_history_path,
    ):
        """
        Track the progress of the scoring process.

        Args:
            can_smiles_list (list): List of canonical SMILES strings.
            target_property_scores (list): List of target property scores.
            cum_total_duration (float): Cumulative total duration.
            cum_score_duration (float): Cumulative score duration.
            cum_gen_duration (float): Cumulative generation duration.
            cum_memory_duration (float): Cumulative memory duration.
            cum_total_doublecheck (int): Cumulative total double check.
            generation_history_path (str): Path to the generation history file.

        Returns:
            None
        """

        def get_time_range(start: float, duration: float, n: int) -> np.ndarray:
            time_per_compound = duration / n
            return np.linspace(start + time_per_compound, start + duration, n)

        n_compounds = len(can_smiles_list)

        batch_times: dict[str, float] = {
            "Generation": gen_duration,
            "Scoring": score_duration,
            "Memory": memory_duration,
            "Total": gen_duration + score_duration + memory_duration,
        }

        # evenly distribute the time over the compounds and update cumulative times
        batch_time_lists: dict[str, np.ndarray] = {}
        for t_name in batch_times:
            t_batch = batch_times[t_name]
            t_cum_start = self.cum_times[t_name]
            batch_time_lists[t_name] = get_time_range(t_cum_start, t_batch, n_compounds)
            self.cum_times[t_name] += t_batch

        # CanSmiles	Score	Total time	Total score time	Total gen time	Total memory time
        self.history["CanSmiles"].extend(can_smiles_list)
        self.history["Score"].extend(property_scores)

        for k, v in batch_time_lists.items():
            self.history[f"{k} time [s]"].extend(v)

        for k, v in self.history.items(): # type: ignore
            assert len(v) == len(self.history["CanSmiles"])

        # write smiles to file if the optimizer is running in parallel
        # and the scoring function is copied to the workers
        if generation_history_path is not None:
            records = list(
                zip(
                    can_smiles_list,
                    property_scores,
                    batch_time_lists["Total"],
                    batch_time_lists["Scoring"],
                    batch_time_lists["Generation"],
                    batch_time_lists["Memory"],
                )
            )

            with open(generation_history_path, "a") as f:
                for record in records:
                    f.write(",".join([str(x) for x in record]) + "\n")

    def _print_progress(self, can_smiles_list, final_scores, iteration_duration):
        mean_score = np.mean(final_scores)
        max_score = np.max(final_scores)
        memsize = len(self.memory_unit.fp_memory_centroid) if self.memory_unit is not None else 0
        n_valid = sum(1 for s in can_smiles_list if s is not None)
        valid = n_valid / len(final_scores)
        mols_per_sec = n_valid / iteration_duration
        print(
            f"Loop {self.loop:07} | Mean score: {mean_score:.3f} | Max score: {max_score:.3f} | Memsize: {memsize:04}, "
            f"Valid {valid:.3f} | {mols_per_sec:.3f} mols/sec"
        )

        self.loop += 1

    @abstractmethod
    def _raw_score_list(self, smiles_list: Sequence[Optional[str]]) -> List[float]:
        """Raw scoring function that takes a list of valid SMILES strings and returns a list of scores."""
        pass

    # function that scores but implements a cache
    def _score_with_cache(self, can_smiles_list: Sequence[Optional[str]]) -> List[float]:
        if len(can_smiles_list) == 0:
            return []

        # get smiles not yet in cache. This will be only valid molecules as
        # None's are already in the cache resulting in invalid scores
        # Use set comprehension to remove duplicates
        new_can_smiles: List[str] = list({smiles for smiles in can_smiles_list if smiles not in self.score_cache})  # type: ignore
        # get scores for smiles not in cache
        new_scores: List[float] = self._raw_score_list(new_can_smiles)

        # update cache
        self.score_cache.update(dict(zip(new_can_smiles, new_scores)))

        # join scores from cache and scores from new scores
        all_scores = [self.score_cache[s] for s in can_smiles_list]
        return all_scores

    def score_list(self, smiles_list: Sequence[Optional[str]], generation_history_path=None) -> List[float]:
        """Scores molecules and internally keeps track of the generation history.
        In each function call the time needed for scoring a batch of molecules
        is recorded. The time between two calls of this scoring function
        is assumed to be the time it takes to generate/select molecules
        for the next cycle of scoring. This allows to record both the
        scoring times and the generation times.

        Args:
            smiles_list (List[str]): Input molecules

        Returns:
            List[float]: Scores of the molecules in smiles_list
        """
        # stop if time budget exhausted
        if self._finished() and self.raise_timeout:
            raise TimeoutError("Time/sample budget exhausted")

        # check if numpy array or list
        if isinstance(smiles_list, np.ndarray):
            smiles_list = smiles_list.tolist()  # type: ignore

        if not isinstance(smiles_list, list):
            raise ValueError("Input must be a list of SMILES strings")

        if len(smiles_list) == 0:
            return []

        assert all(isinstance(s, str) or s is None for s in smiles_list)

        score_start = time()
        gen_duration = score_start - self.gen_start

        # ------------------------ target and property ------------------------
        # For invalid smiles this will give None
        can_smiles_list: List[Optional[str]] = self.canonicalizer.process_list(smiles_list)

        # calculate scores
        property_scores: List[float] = self._score_with_cache(can_smiles_list)

        score_end = time()
        score_duration = score_end - score_start

        # ------------------------ memory ------------------------
        # Cache is not as easy as these scores are non-stationary. But zero stays zero
        # None values receive a score of zero
        memory_start = time()

        if self.memory_unit is not None:
            memory_scores = self.memory_unit.score_and_update_memory(can_smiles_list, property_scores)
            final_scores = (np.array(property_scores) * np.array(memory_scores)).tolist()
        else:
            final_scores = property_scores

        memory_end = time()
        memory_duration = memory_end - memory_start
        # ------------------ Combine scores and track progress ------------------

        # track progress
        self._track_progress(
            can_smiles_list,
            property_scores,
            gen_duration,
            score_duration,
            memory_duration,
            generation_history_path=generation_history_path,
        )

        iteration_duration = gen_duration + score_duration + memory_duration
        if self.print_progress:
            self._print_progress(can_smiles_list, final_scores, iteration_duration)

        self.gen_start = time()
        return final_scores

    def __call__(self, smiles_list: Union[List[str], str]) -> Union[List[float], float]:
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
            return self.score_list(smiles_list)[0]
        return self.score_list(smiles_list)


class BenchmarkScoringFunction(ScoringFunction):
    """Wrapper for the scoring function used in the benchmarking paper."""

    def __init__(
        self,
        scoring_function_dir: str,
        use_property_constraints: bool,
        memory_known_active_init: bool,
        time_budget: Optional[float],
        sample_budget: Optional[int],
        memory_distance_threshold: Optional[float],
        memory_score_threshold: Optional[float],
        invalid_score: float = 0.0,
        n_jobs: int = 8,
        print_progress: bool = True,
        samples_per_minute_threshold: int = 100,
        raise_timeout: bool = True,
    ):
        super().__init__(
            time_budget,
            sample_budget,
            memory_distance_threshold,
            memory_score_threshold,
            invalid_score,
            n_jobs,
            print_progress=print_progress,
            raise_timeout=raise_timeout,
            samples_per_minute_threshold=samples_per_minute_threshold,
        )

        self.scoring_function_dir = scoring_function_dir
        self.use_property_constraints = use_property_constraints
        self.memory_known_active_init = memory_known_active_init

        if self.memory_known_active_init and self.memory_unit is None:
            raise ValueError("Memory known active initialization requires a memory unit")

        if self.memory_unit is not None:
            self.fingerprint_calculator = self.memory_unit.fingerprint_calculator
        else:
            self.fingerprint_calculator = FingerprintCalculator(n_jobs=n_jobs)

        if self.use_property_constraints:
            self.property_calculator = PropertyCalculator(n_jobs=n_jobs)

        if self.memory_known_active_init:
            self._init_memory_known_actives()
        if self.use_property_constraints:
            self._init_property_constraints()

        model_path = os.path.join(self.scoring_function_dir, "classifier.pkl")
        self.load_model(model_path=model_path)

    def load_model(self, model_path: str) -> None:
        """
        Load the model from a file.

        Args:
            model_path (str): The path to the model file.

        Returns:
            None
        """
        clf_path = os.path.join(model_path)
        with open(clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def _init_property_constraints(self) -> None:
        """
        Initializes the property constraints for the scoring function.

        This function loads the necessary data files and initializes the property constraints
        for the scoring function.

        Args:
            None

        Returns:
            None
        """
        known_bits_fn = os.path.join(self.scoring_function_dir, "../guacamol_known_bits.json")
        self.exotic_fp_calculator = ExoticFingerprintRatio(known_bits_fn)
        threshold_fn = os.path.join(self.scoring_function_dir, "../guacamol_thresholds.json")
        with open(threshold_fn, "r") as f:
            self.property_thresholds = json.load(f)

    def _init_memory_known_actives(self) -> None:
        """
        Initializes the memory unit with known active compounds.

        Reads the scoring function data from a CSV file and extracts the SMILES
        of the compounds labeled as active in the training split. Adds these
        compounds to the memory unit.

        Returns:
            None
        """
        scoring_function_df = pd.read_csv(os.path.join(self.scoring_function_dir, "splits.csv"))
        train_actives = filterdf(scoring_function_df, {"Split": "train", "label": 1}).smiles.to_list()
        if self.memory_unit is not None:
            self.memory_unit.add_to_memory(train_actives)

    def _check_property_constraints(self, smiles_list: List[str]) -> np.ndarray:
        """
        Check the property constraints for a list of SMILES strings.

        Args:
            smiles_list (str): List of SMILES strings.

        Returns:
            np.ndarray: Boolean array indicating whether each SMILES string satisfies the property constraints.
        """
        prop_tuple_list = self.property_calculator.process_list(smiles_list)

        # get list for each property
        mw_list, logp_list, ecfp4bits_list = zip(*prop_tuple_list)

        # transform the ecfp4bits to ratios of known bits
        ecfp4_ratios = [self.exotic_fp_calculator.get_ratio(fp_bits) for fp_bits in ecfp4bits_list]

        # Put everything into a dictionary
        prop_dict: Dict[str, List[float]] = {}
        prop_dict["MW"] = mw_list # type: ignore
        prop_dict["LogP"] = logp_list # type: ignore
        prop_dict["ECFP4bits"] = ecfp4_ratios # type: ignore

        # Check if all properties are within the thresholds
        all_okay = np.array([True] * len(smiles_list))
        for prop in prop_dict.keys():
            lower, upper = self.property_thresholds[prop]
            prop_okay = [lower < p < upper for p in prop_dict[prop]]
            all_okay = np.logical_and(all_okay, prop_okay)
        return all_okay

    def _raw_score_list(
        self,
        can_smiles_list: Sequence[Optional[str]],
    ) -> List[float]:
        """Calculate the target and property scores for a list of canonical smiles
        or None values in case of invalid smiles.

        Args:
            can_smiles_list (List[Optional[str]]): A list of valid smiles.

        Returns:
            List[float]: The target and property scores for each valid smile.

        """
        if len(can_smiles_list) == 0:
            return []

        # get smiles not yet in cache. This will be only valid molecules as
        # None's are already in the cache resulting in invalid scores
        # Use set comprehension to remove duplicates

        can_smiles_to_score: List[str] = list({smiles for smiles in can_smiles_list if smiles not in self.score_cache})  # type: ignore

        # if nothing is left to calculate return the cached scores
        if len(can_smiles_to_score) == 0:
            return [self.score_cache[s] for s in can_smiles_list]

        fp_list = self.fingerprint_calculator.process_list(can_smiles_to_score)
        assert not any(fp is None for fp in fp_list)
        fp_array = np.array([ebv2np(fp) for fp in fp_list])
        # Predict scores for valid molecules
        target_scores = self.clf.predict_proba(fp_array)[:, 1]

        if self.use_property_constraints:
            props_okay = self._check_property_constraints(can_smiles_to_score)
            new_target_property_scores = target_scores * np.array(props_okay)
        else:
            new_target_property_scores = target_scores

        # update cache
        self.score_cache.update(dict(zip(can_smiles_to_score, new_target_property_scores)))

        # join scores from cache and scores from new predictions
        target_property_scores = [self.score_cache[s] for s in can_smiles_list]
        return target_property_scores
