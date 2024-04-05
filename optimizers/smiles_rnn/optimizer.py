import os
import torch
import sys

from smiles_rnn.utils import set_default_device_cuda, to_tensor
from smiles_rnn.RL import AugmentedHillClimb, BestAgentReminder
import uuid


class AugmentedHCOptimizer:
    def __init__(
        self,
        batch_size=256,
        sigma=120,
        topk=0.25,
        learning_rate=0.0005,
        pretrained_model_path=None,
        n_steps=1_000_000,
    ):
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.batch_size = batch_size
        self.topk = topk
        self.n_steps = n_steps
        self.pretrained_model_path = pretrained_model_path

    def generate_optimized_molecules(self, scoring_function, *args, **kwargs):
        device = set_default_device_cuda("cuda")
        temp_dir = "/tmp/ahc" + str(uuid.uuid4())
        # Initalization
        if self.pretrained_model_path is None:
            self.pretrained_model_path = os.path.join(
                os.path.dirname(__file__), "prior/Prior_guacamol_chembl_Epoch-5.ckpt"
            )
        AHC = AugmentedHillClimb(
            device=device,
            model="RNN",
            agent=self.pretrained_model_path,
            scoring_function=scoring_function,
            save_dir=temp_dir,
            optimizer=torch.optim.Adam,
            learning_rate=self.learning_rate,
            is_molscore=False,
            batch_size=self.batch_size,
            sigma=self.sigma,
            topk=self.topk,
        )

        AHC.train(n_steps=self.n_steps, save_freq=100000)


class BestAgentReminderOptimizer:
    def __init__(
        self,
        batch_size=256,
        sigma=120,
        alpha=0.5,
        learning_rate=0.0005,
        pretrained_model_path=None,
        n_steps=1_000_000,
    ):
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.batch_size = batch_size
        self.alpha = alpha
        self.n_steps = n_steps
        self.pretrained_model_path = pretrained_model_path

    def generate_optimized_molecules(self, scoring_function, *args, **kwargs):
        device = set_default_device_cuda("cuda")
        temp_dir = "/tmp/ahc" + str(uuid.uuid4())
        # Initalization
        if self.pretrained_model_path is None:
            self.pretrained_model_path = os.path.join(
                os.path.dirname(__file__), "prior/Prior_guacamol_chembl_Epoch-5.ckpt"
            )

        bar = BestAgentReminder(
            device=device,
            model="RNN",
            agent=self.pretrained_model_path,
            scoring_function=scoring_function,
            save_dir=temp_dir,
            optimizer=torch.optim.Adam,
            learning_rate=self.learning_rate,
            is_molscore=False,
            batch_size=self.batch_size,
            sigma=self.sigma,
            alpha=self.alpha,
        )

        bar.train(n_steps=self.n_steps, save_freq=100000)
