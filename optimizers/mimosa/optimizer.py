import os
import sys

import torch

path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append(".")

from random import shuffle

from chemutils import *
from inference_utils import *
from online_train import *
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


class MimosaOptimizer:
    def __init__(
        self,
        population_size=50,
        offspring_size=500,
        lamb=0.1,
        train_epoch=3,
        train_data_size=800,
        smi_file="data/guacamol_v1_all.smiles",
    ):
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.lamb = lamb
        self.train_epoch = train_epoch
        self.train_data_size = train_data_size

        with open(smi_file, "r") as f:
            self.all_smiles = f.read().split("\n")

    def generate_optimized_molecules(self, scoring_function, *args, **kwargs):
        all_smiles_score_list = []

        model_ckpt = os.path.join(path_here, "pretrained_model/GNN.ckpt")  # mGNN only
        gnn = torch.load(model_ckpt)

        start_smiles_lst = [
            "C1(N)=NC=CC=N1",
            "C1(C)=NC=CC=N1",
            "C1(C)=CC=CC=C1",
            "C1(N)=CC=CC=C1",
            "CC",
            "C1(C)CCCCC1",
        ]
        shuffle(self.all_smiles)
        warmstart_smiles_lst = self.all_smiles[:1000]
        warmstart_smiles_score = scoring_function.score_list(warmstart_smiles_lst)
        warmstart_smiles_score_lst = list(
            zip(warmstart_smiles_lst, warmstart_smiles_score)
        )
        warmstart_smiles_score_lst.sort(
            key=lambda x: x[1], reverse=True
        )  #### [(smiles1, score1), (smiles2, score2), ... ]
        all_smiles_score_list.extend(warmstart_smiles_score_lst)

        all_smiles_score_list.sort(key=lambda x: x[1], reverse=True)
        good_smiles_list = all_smiles_score_list[:500]
        train_gnn(good_smiles_list, gnn, epoch=self.train_epoch)

        warmstart_smiles_lst = [
            i[0] for i in warmstart_smiles_score_lst[:50]
        ]  #### only smiles

        start_smiles_lst += warmstart_smiles_lst
        current_set = set(start_smiles_lst)

        while True:
            next_set = set()
            for smiles in current_set:
                smiles_set = optimize_single_molecule_one_iterate(smiles, gnn)
                next_set = next_set.union(smiles_set)

            smiles_lst = list(next_set)
            shuffle(smiles_lst)

            smiles_lst = smiles_lst[: self.offspring_size]

            score_lst = scoring_function.score_list(smiles_lst)

            smiles_score_lst = [
                (smiles, score) for smiles, score in zip(smiles_lst, score_lst)
            ]
            smiles_score_lst.sort(key=lambda x: x[1], reverse=True)
            current_set, _, _ = dpp(
                smiles_score_lst=smiles_score_lst,
                num_return=self.population_size,
                lamb=self.lamb,
            )  # Option II: DPP

            ### online train gnn
            all_smiles_score_list.extend(smiles_score_lst)
            all_smiles_score_list.sort(key=lambda x: x[1], reverse=True)
            # good_smiles_list = [i[0] for i in filter(lambda x:x[1] > 0.5, all_smiles_score_list)]
            # if len(good_smiles_list) < config['train_data_size']:
            #     good_smiles_list.extend(all_smiles_score_list[:config['train_data_size']])

            # import ipdb; ipdb.set_trace()
            # print(f"Training mol length: {len(good_smiles_list)}")
            good_smiles_list = all_smiles_score_list[: self.train_data_size]
            train_gnn(good_smiles_list, gnn, epoch=self.train_epoch)
