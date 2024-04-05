import os

from rdkit import RDLogger

from divopt.evaluation.process_results import process_results_file
from divopt.scoring import BenchmarkScoringFunction

RDLogger.DisableLog("rdApp.*")


def import_optimizer(optimizer_name: str):
    if optimizer_name == "Mimosa":
        from optimizers.mimosa.optimizer import MimosaOptimizer as Optimizer
    elif optimizer_name == "LSTM-HC":
        from optimizers.guacamol_baselines.smiles_lstm_hc.goal_directed_generation import (
            SmilesRnnDirectedGenerator as Optimizer,
        )
    elif optimizer_name == "LSTM-PPO":
        from optimizers.guacamol_baselines.smiles_lstm_ppo.goal_directed_generation import (
            PPODirectedGenerator as Optimizer,
        )
    elif optimizer_name == "GA":
        from optimizers.guacamol_baselines.graph_ga.goal_directed_generation import (
            GB_GA_Generator as Optimizer,
        )
    elif optimizer_name == "GraphMCTS":
        from optimizers.guacamol_baselines.graph_mcts.goal_directed_generation import (
            GB_MCTS_Generator as Optimizer,
        )
    elif optimizer_name == "SmilesGA":
        from optimizers.guacamol_baselines.smiles_ga.goal_directed_generation import (
            ChemGEGenerator as Optimizer,
        )
    elif optimizer_name == "VS":
        from optimizers.virtualscreen.optimizer import VSOptimizer as Optimizer
    elif optimizer_name == "Stoned":
        from optimizers.stoned.optimizer import StonedOptimizer as Optimizer
    elif optimizer_name == "Reinvent":
        from optimizers.reinvent.run import ReinventOptimizer as Optimizer
    elif optimizer_name == "Mars":
        from optimizers.mars.optimizer import MarsOptimizer as Optimizer
    elif optimizer_name == "Gflownet":
        from optimizers.gflownet_recursion.optimizer import (
            GflownetOptimizer as Optimizer,
        )
    elif optimizer_name == "GflownetDF":
        from optimizers.gflownet_recursion.optimizer import (
            GflownetOptimizer as Optimizer,
        )
    elif optimizer_name == "AugHC":
        from optimizers.smiles_rnn.optimizer import AugmentedHCOptimizer as Optimizer
    elif optimizer_name == "BestAgentReminder":
        from optimizers.smiles_rnn.optimizer import (
            BestAgentReminderOptimizer as Optimizer,
        )
    elif optimizer_name == "AugMemory":
        from optimizers.smiles_rnn.augmented_memory import (
            AugmentedMemoryOptimizer as Optimizer,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_name} not found!")
    return Optimizer


def optimize(
    scoring_function_name: str,
    memory_score_threshold: float,
    memory_distance_threshold: float,
    memory_known_active_init: bool,
    use_property_constraints: bool,
    optimizer_name: str,
    optimizer_args: dict,
    time_budget: float,
    sample_budget: int,
    n_jobs_scoring_function: int = 8,
):
    scoring_function_name = scoring_function_name.lower()

    scoring_function_dir: str = os.path.join(
        os.environ["DIVOPTPATH"], f"data/scoring_functions/{scoring_function_name}"
    )

    is_valid_scoring_function = scoring_function_name in ["jnk3", "gsk3", "drd2"]

    if not is_valid_scoring_function:
        raise LookupError(f"Scoring function '{scoring_function_name}' not a valid choice!")

    # Setup scoring function
    print("Loading scoring function...")
    scoring_function = BenchmarkScoringFunction(
        time_budget=time_budget,
        sample_budget=sample_budget,
        memory_distance_threshold=memory_distance_threshold,
        memory_score_threshold=memory_score_threshold,
        memory_known_active_init=memory_known_active_init,
        use_property_constraints=use_property_constraints,
        scoring_function_dir=scoring_function_dir,
        n_jobs=n_jobs_scoring_function,
    )

    # Setup optimizer
    print(f"Loading optimizer {optimizer_name}")
    opt_class = import_optimizer(optimizer_name)
    optimizer = opt_class(**optimizer_args)
    print("Loading optimizer... Done")

    # run optimizer
    print("Starting optimization... ")
    try:
        scoring_function.start_timer_and_reset()
        optimizer.generate_optimized_molecules(scoring_function, 3)  # type: ignore
    except TimeoutError:
        print("Time/sample budget exhausted!")

    df = scoring_function.get_history()

    run_dir = os.path.dirname(args.config)
    results_fname = os.path.join(run_dir, "results.csv")
    df.to_csv(results_fname, index=False)

    process_results_file(results_fname)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # import the config file
    with open(args.config) as f:
        config = json.load(f)

    optimize(**config)
