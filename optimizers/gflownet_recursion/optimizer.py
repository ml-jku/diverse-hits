import glob
import os
import socket
from typing import Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gflownet.config import Config
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.trainer import FlatRewards, GFNTask, RewardScalar
from gflownet.utils.conditioning import TemperatureConditional
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import Dataset


class WrapperTask(GFNTask):
    """Defines a task for the using a provided scoring function to calcualte rewards for a GFN model."""

    def __init__(
        self,
        dataset: Dataset,
        cfg: Config,
        rng: Optional[np.random.Generator] = None,
        wrap_model: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        self._wrap_model = wrap_model
        self.dataset = dataset
        self.temperature_conditional = TemperatureConditional(cfg, rng)
        self.num_cond_dim = self.temperature_conditional.encoding_size()
        self.scoring_function = None

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y) / 8)

    def inverse_flat_reward_transform(self, rp):
        return rp * 8

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        return RewardScalar(self.temperature_conditional.transform(cond_info, flat_reward))

    def compute_flat_rewards(self, mols: List[RDMol], generation_history_path: str) -> Tuple[FlatRewards, Tensor]:
        if self.scoring_function is None:
            raise ValueError("Scoring function not set")
        is_valid = torch.tensor([i is not None for i in mols]).bool()
        from rdkit import Chem

        # raise ValueError("This is not the scoring function you are looking for")

        smiles = [Chem.MolToSmiles(mol) for mol in mols]
        preds = self.scoring_function.score_list(smiles, generation_history_path)
        preds = self.flat_reward_transform(preds).clip(1e-4, 100).reshape((-1, 1))
        return FlatRewards(preds), is_valid

    def set_scoring_function(self, scoring_function):
        self.scoring_function = scoring_function


class WrapperTrainer(StandardOnlineTrainer):
    task: WrapperTask

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = False
        cfg.num_workers = 8
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20_000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        cfg.algo.global_batch_size = 64
        cfg.algo.offline_ratio = 0
        cfg.model.num_emb = 128
        cfg.model.num_layers = 4

        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 9
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.valid_offline_ratio = 0
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-3
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False

        cfg.replay.use = False
        cfg.replay.capacity = 10_000
        cfg.replay.warmup = 1_000

    def setup_task(self):
        self.task = WrapperTask(
            dataset=self.training_data,
            cfg=self.cfg,
            rng=self.rng,
            wrap_model=self._wrap_for_mp,
        )

    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(max_frags=self.cfg.algo.max_nodes, num_cond_dim=self.task.num_cond_dim)


def gather_history_from_log_dir(log_dir, columns):
    """Gathers the history from a log dir. This is necessary as gflownet uses multiple processes in parallel and the scoring function object is
    copied into each process. Therefore, the scoring function object is not updated with the history of the molecules generated in the other processes.
    """
    fnames = glob.glob(os.path.join(log_dir, "train", "generated_mols_*.csv"))
    partial_list = []
    for fname in fnames:
        partial_list.append(pd.read_csv(fname, header=None))

    df = pd.concat(partial_list)
    df.columns = columns
    df.sort_values("Total time [s]", ascending=True, inplace=True)
    return df


hps = {
    "log_dir": "./logs/debug_run_seh_frag",
    "device": "cpu",
    "overwrite_existing_exp": True,
    "num_training_steps": 10_000,
    "num_workers": 8,
    "opt": {
        "lr_decay": 20000,
    },
    "algo": {"sampling_tau": 0.99},
    "cond": {
        "temperature": {
            "sample_dist": "uniform",
            "dist_params": [0, 64.0],
        }
    },
}


class GflownetOptimizer:
    def __init__(
        self,
        num_training_steps=100_000,
        learning_rate=1e-4,
        momentum=0.9,
        num_workers=8,
        sampling_tau=0.99,
    ):
        hps["num_training_steps"] = num_training_steps
        hps["opt"]["learning_rate"] = learning_rate
        hps["opt"]["momentum"] = momentum
        hps["num_workers"] = num_workers
        hps["algo"]["sampling_tau"] = sampling_tau

        # create random temp dir for logging
        self.log_dir = os.path.join("/tmp", "gflownet_temp_results" + str(uuid4()))
        hps["log_dir"] = self.log_dir
        self.trial = WrapperTrainer(hps)

    def generate_optimized_molecules(self, scoring_function, *args, **kwargs):
        try:
            self.trial.task.set_scoring_function(scoring_function)
            self.trial.run()
        except TimeoutError:
            print("Interrupting as compute budget is reached.")
        except Exception as e:
            raise e
        finally:
            history_df = gather_history_from_log_dir(self.log_dir, columns=scoring_function.history.keys())
            scoring_function.history = {k: list(v) for k, v in history_df.items()}
