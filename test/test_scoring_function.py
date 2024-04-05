import os
import time
from typing import List

from divopt.scoring import BenchmarkScoringFunction, ScoringFunction

with open("test/test_smiles.txt", "r") as f:
    smiles_list = f.read().splitlines()


DELAY = 0.0001


class LengthScoringFunction(ScoringFunction):
    def __init__(
        self,
        time_budget: float | None,
        sample_budget: int | None,
        memory_distance_threshold: float | None,
        memory_score_threshold: float | None,
        invalid_score: float = 0,
        n_jobs: int = 8,
        print_progress: bool = True,
        samples_per_minute_threshold: int = 100,
    ):
        super().__init__(
            time_budget,
            sample_budget,
            memory_distance_threshold,
            memory_score_threshold,
            invalid_score,
            n_jobs,
            print_progress,
            samples_per_minute_threshold,
        )

    def _raw_score_list(self, smiles_list: List[str]) -> List[float]:
        time.sleep(DELAY * len(smiles_list))
        return [len(smiles) for smiles in smiles_list]


def test_budget_specified():
    try:
        LengthScoringFunction(
            time_budget=None,
            sample_budget=None,
            memory_distance_threshold=0.5,
            memory_score_threshold=0.5,
            invalid_score=0,
            n_jobs=8,
            print_progress=True,
            samples_per_minute_threshold=100,
        )
    except ValueError as e:
        assert "Either time_budget or sample_budget must be set" in str(e)


def test_mem_args():
    try:
        LengthScoringFunction(
            time_budget=60,
            sample_budget=None,
            memory_distance_threshold=0.5,
            memory_score_threshold=None,
            invalid_score=0,
            n_jobs=8,
            print_progress=True,
            samples_per_minute_threshold=100,
        )
    except ValueError as e:
        assert "Both memory_distance_threshold and memory_score_threshold must be set or not" in str(e)


def test_score_list():
    scoring_function = LengthScoringFunction(
        time_budget=60,
        sample_budget=None,
        memory_distance_threshold=None,
        memory_score_threshold=None,
        invalid_score=0,
        n_jobs=8,
        print_progress=True,
        samples_per_minute_threshold=100,
    )
    scoring_function.start_timer_and_reset()
    scores = scoring_function.score_list(smiles_list)
    assert scores == [len(smiles) for smiles in smiles_list]


def test_cache():
    scoring_function = LengthScoringFunction(
        time_budget=60,
        sample_budget=None,
        memory_distance_threshold=None,
        memory_score_threshold=None,
        invalid_score=0,
        n_jobs=8,
        print_progress=True,
        samples_per_minute_threshold=100,
    )
    scoring_function.start_timer_and_reset()
    _ = scoring_function.score_list(smiles_list)

    start = time.time()
    _ = scoring_function.score_list(smiles_list)
    duration2 = time.time() - start
    assert duration2 < DELAY * len(smiles_list) * 0.5


def test_benchmark_scoring_function():
    divopt_path = os.environ["DIVOPTPATH"]
    scoring_function_dir = os.path.join(divopt_path, "data/scoring_functions/drd2")
    scoring_function = BenchmarkScoringFunction(
        scoring_function_dir=scoring_function_dir,
        use_property_constraints=True,
        memory_known_active_init=True,
        time_budget=60,
        sample_budget=None,
        memory_distance_threshold=0.7,
        memory_score_threshold=0.5,
        invalid_score=0,
        n_jobs=8,
        print_progress=True,
        samples_per_minute_threshold=100,
    )
    scoring_function.start_timer_and_reset()
    scores = scoring_function.score_list(smiles_list)
    assert len(scores) == len(smiles_list)
    assert all(isinstance(score, float) for score in scores)
