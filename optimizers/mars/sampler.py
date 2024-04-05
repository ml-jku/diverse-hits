import os
import math
import torch
import random
import logging as log
from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from torch.utils import data

# from torch.utils.tensorboard import SummaryWriter

from .common.train import train
from .common.chem import mol_to_dgl
from .common.utils import print_mols
from .datasets.utils import load_mols
from .datasets.datasets import ImitationDataset


class Sampler:
    def __init__(self, config, proposal, oracle, n_jobs=1):
        self.proposal = proposal
        self.config = config
        self.oracle = oracle

        self.writer = None
        self.run_dir = None

        ### for sampling
        self.step = None
        self.PATIENCE = 100
        self.patience = 100
        self.best_eval_res = 0.0
        self.best_avg_score = 0.0
        self.last_avg_size = 20
        self.train = config["train"]
        self.num_mols = config["num_mols"]
        # self.num_step = config['num_step']
        self.batch_size = config["batch_size"]
        self.fps_ref = (
            [
                AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048)
                for x in config["mols_ref"]
            ]
            if config["mols_ref"]
            else None
        )
        self.n_jobs = n_jobs
        if self.n_jobs > 1:
            from multiprocessing import Pool
            self.pool = Pool(self.n_jobs)

        ### for training editor
        if self.train:
            self.dataset = None
            self.DATASET_MAX_SIZE = config["dataset_size"]
            self.optimizer = torch.optim.Adam(
                self.proposal.editor.parameters(), lr=config["lr"]
            )

    def scores_from_dicts(self, dicts):
        """
        @params:
            dicts (list): list of score dictionaries
        @return:
            scores (list): sum of property scores of each molecule after clipping
        """
        scores = []
        # score_norm = sum(self.score_wght.values())
        for score_dict in dicts:
            score = 0.0
            for k, v in score_dict.items():
                score += v
            # score /= score_norm
            score = max(score, 0.0)
            scores.append(score)
        return scores

    def record(self, step, old_mols, old_dicts, acc_rates):
        ### average score and size
        old_scores = self.scores_from_dicts(old_dicts)
        avg_score = 1.0 * sum(old_scores) / len(old_scores)
        sizes = [mol.GetNumAtoms() for mol in old_mols]
        avg_size = sum(sizes) / len(old_mols)
        self.last_avg_size = avg_size

        ### successful rate and uniqueness
        fps_mols, unique = [], set()
        success_dict = {k: 0.0 for k in old_dicts[0].keys()}
        success, novelty, diversity = 0.0, 0.0, 0.0
        for i, score_dict in enumerate(old_dicts):
            all_success = True
            for k, v in score_dict.items():
                if v >= self.score_succ[k]:
                    success_dict[k] += 1.0
                else:
                    all_success = False
            success += all_success
            if all_success:
                fps_mols.append(old_mols[i])
                unique.add(Chem.MolToSmiles(old_mols[i]))
        success_dict = {k: v / len(old_mols) for k, v in success_dict.items()}
        success = 1.0 * success / len(old_mols)
        unique = 1.0 * len(unique) / (len(fps_mols) + 1e-6)

        ### novelty and diversity
        fps_mols = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in fps_mols]

        if self.fps_ref:
            n_sim = 0.0
            for i in range(len(fps_mols)):
                sims = DataStructs.BulkTanimotoSimilarity(fps_mols[i], self.fps_ref)
                if max(sims) >= 0.4:
                    n_sim += 1
            novelty = 1.0 - 1.0 * n_sim / (len(fps_mols) + 1e-6)
        else:
            novelty = 1.0

        similarity = 0.0
        for i in range(len(fps_mols)):
            sims = DataStructs.BulkTanimotoSimilarity(fps_mols[i], fps_mols[:i])
            similarity += sum(sims)
        n = len(fps_mols)
        n_pairs = n * (n - 1) / 2
        diversity = 1 - similarity / (n_pairs + 1e-6)

        diversity = min(diversity, 1.0)
        novelty = min(novelty, 1.0)
        evaluation = {
            "success": success,
            "unique": unique,
            "novelty": novelty,
            "diversity": diversity,
            "prod": success * novelty * diversity,
        }

        ### logging and writing tensorboard
        # log.info('Step: {:02d},\tScore: {:.7f}'.format(step, avg_score))
        # self.writer.add_scalar('score_avg', avg_score, step)
        # self.writer.add_scalar('size_avg', avg_size, step)
        # self.writer.add_scalars('success_dict', success_dict, step)
        # self.writer.add_scalars('evaluation', evaluation, step)
        # self.writer.add_histogram('acc_rates', torch.tensor(acc_rates), step)
        # self.writer.add_histogram('scores', torch.tensor(old_scores), step)
        # for k in old_dicts[0].keys():
        #     scores = [score_dict[k] for score_dict in old_dicts]
        #     self.writer.add_histogram(k, torch.tensor(scores), step)
        # print_mols(self.run_dir, step, old_mols, old_scores, old_dicts)

        ### early stop
        if (
            evaluation["prod"] > 0.1
            and evaluation["prod"] < self.best_eval_res + 0.01
            and avg_score > 0.1
            and avg_score < self.best_avg_score + 0.01
        ):
            self.patience -= 1
        else:
            self.patience = self.PATIENCE
            self.best_eval_res = max(self.best_eval_res, evaluation["prod"])
            self.best_avg_score = max(self.best_avg_score, avg_score)

    def acc_rates(self, new_scores, old_scores, fixings):
        """
        compute sampling acceptance rates
        @params:
            new_scores : scores of new proposed molecules
            old_scores : scores of old molcules
            fixings    : acceptance rate fixing propotions for each proposal
        """
        raise NotImplementedError

    def sample(self, mols_init):
        """
        sample molecules from initial ones
        @params:
            mols_init : initial molecules
        """

        ### sample
        def mol2smiles(mol):
            return Chem.MolToSmiles(mol)
        
        old_mols = [mol for mol in mols_init]
        old_dicts = [{} for i in old_mols]
        if self.n_jobs <= 1:
            old_smiles = [Chem.MolToSmiles(mol) for mol in old_mols]
        else:
            old_smiles = self.pool.map(mol2smiles, old_mols)
            
        old_scores = self.oracle.score_list(old_smiles)
        for ii, (smiles, value) in enumerate(zip(old_smiles, old_scores)):
            # value = self.oracle(smiles)
            old_dicts[ii][smiles] = value
        acc_rates = [0.0 for _ in old_mols]

        step = 1

        while True:
            if self.patience <= 0:
                break
            self.step = step

            new_mols, fixings = self.proposal.propose(old_mols)

            new_dicts = [{} for i in new_mols]
            
            if self.n_jobs <= 1:
                new_smiles = [Chem.MolToSmiles(mol) for mol in new_mols]
            else:
                new_smiles = self.pool.map(mol2smiles, new_mols)

            new_scores = self.oracle.score_list(new_smiles)
            for ii, (smiles, value) in enumerate(zip(new_smiles, new_scores)):
                new_dicts[ii][smiles] = value

            indices = [i for i in range(len(old_mols)) if new_scores[i] > old_scores[i]]

            acc_rates = self.acc_rates(new_scores, old_scores, fixings)
            acc_rates = [min(1.0, max(0.0, A)) for A in acc_rates]
            for i in range(self.num_mols):
                A = acc_rates[i]  # A = p(x') * g(x|x') / p(x) / g(x'|x)
                if random.random() > A:
                    continue
                old_mols[i] = new_mols[i]
                old_scores[i] = new_scores[i]
                old_dicts[i] = new_dicts[i]

            ### train editor
            if self.train:
                dataset = self.proposal.dataset
                dataset = data.Subset(dataset, indices)
                if self.dataset and len(self.dataset) > 0:
                    # print(dataset)
                    try:
                        self.dataset.merge_(dataset)
                    except:
                        print(f"Problem happned when merging data, pass this round.")
                else:
                    self.dataset = ImitationDataset.reconstruct(dataset)
                n_sample = len(self.dataset)
                if n_sample > 2 * self.DATASET_MAX_SIZE:
                    indices = [i for i in range(n_sample)]
                    random.shuffle(indices)
                    indices = indices[: self.DATASET_MAX_SIZE]
                    self.dataset = data.Subset(self.dataset, indices)
                    self.dataset = ImitationDataset.reconstruct(self.dataset)
                batch_size = int(self.batch_size * 20 / self.last_avg_size)
                log.info("formed a imitation dataset of size %i" % len(self.dataset))
                loader = data.DataLoader(
                    self.dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=ImitationDataset.collate_fn,
                )

                train(
                    model=self.proposal.editor,
                    loaders={"dev": loader},
                    optimizer=self.optimizer,
                    n_epoch=1,
                    log_every=25,
                    max_step=25,
                    metrics=[
                        "loss",
                        "loss_del",
                        "prob_del",
                        "loss_add",
                        "prob_add",
                        "loss_arm",
                        "prob_arm",
                    ],
                )

                if not self.proposal.editor.device == torch.device("cpu"):
                    torch.cuda.empty_cache()

                step += 1


class Sampler_SA(Sampler):
    def __init__(
        self,
        config,
        proposal,
        oracle,
    ):
        super().__init__(
            config,
            proposal,
            oracle,
        )
        self.k = 0
        self.step_cur_T = 0
        self.T = self.T_k(self.k)

    # @staticmethod
    def T_k(self, k):
        T_0 = 1.0  # .1
        BETA = self.config["beta"]
        ALPHA = self.config["alpha"]
        # return 1. * T_0 / (math.log(k + 1) + 1e-6)
        # return max(1e-6, T_0 - k * BETA)
        return ALPHA**k * T_0

    def update_T(self):
        STEP_PER_T = 5
        if self.step_cur_T == STEP_PER_T:
            self.k += 1
            self.step_cur_T = 0
            self.T = self.T_k(self.k)
        else:
            self.step_cur_T += 1
        self.T = max(self.T, 1e-2)
        return self.T

    def acc_rates(self, new_scores, old_scores, fixings):
        acc_rates = []
        T = self.update_T()
        # T = 1. / (4. * math.log(self.step + 8.))
        for i in range(self.num_mols):
            # A = min(1., math.exp(1. * (new_scores[i] - old_scores[i]) / T))
            A = min(1.0, 1.0 * new_scores[i] / max(old_scores[i], 1e-6))
            A = min(1.0, A ** (1.0 / T))
            acc_rates.append(A)
        return acc_rates


class Sampler_MH(Sampler):
    def __init__(
        self,
        config,
        proposal,
        oracle,
    ):
        super().__init__(
            config,
            proposal,
            oracle,
        )
        self.power = 30.0

    def acc_rates(self, new_scores, old_scores, fixings):
        acc_rates = []
        for i in range(self.num_mols):
            old_score = max(old_scores[i], 1e-5)
            A = ((new_scores[i] / old_score) ** self.power) * fixings[i]
            acc_rates.append(A)
        return acc_rates


class Sampler_Recursive(Sampler):
    def __init__(
        self,
        config,
        proposal,
        oracle,
    ):
        super().__init__(
            config,
            proposal,
            oracle,
        )

    def acc_rates(self, new_scores, old_scores, fixings):
        acc_rates = []
        for i in range(self.num_mols):
            A = 1.0
            acc_rates.append(A)
        return acc_rates
