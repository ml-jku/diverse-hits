import os
import torch
import pandas as pd
import numpy as np

from smiles_rnn.rnn import Model
from smiles_rnn.utils import get_randomized_smiles, to_tensor
from typing import Optional


class AugmentedMemory:
    def __init__(
        self,
        pretrained_model_path: str,
        batch_size: int = 64,
        sigma: int = 500,
        learning_rate: float = 0.0005,
        replay_buffer_size: int = 100,
        replay_number: int = 10,
        augmented_memory: bool = True,
        augmentation_rounds: int = 2,
    ):
        self.prior = Model.load_from_file(pretrained_model_path)
        self.agent = Model.load_from_file(pretrained_model_path)
        self.batch_size = batch_size
        self.sigma = sigma
        self.replay_buffer = ExperienceReplay(
            pretrained_model_path=pretrained_model_path,
            memory_size=replay_buffer_size,
            replay_number=replay_number,
        )
        self.augmented_memory = augmented_memory
        self.augmentation_rounds = augmentation_rounds
        self.optimizer = torch.optim.Adam(
            self.agent.network.parameters(), lr=learning_rate
        )


class ExperienceReplay:
    def __init__(
        self,
        pretrained_model_path: str,
        memory_size: int = 100,
        replay_number: int = 10,
    ):
        self.buffer = pd.DataFrame(columns=["smiles", "likelihood", "scores"])
        self.memory_size = memory_size
        self.replay_number = replay_number
        self.prior = Model.load_from_file(pretrained_model_path)

    def add_to_buffer(self, smiles, scores, neg_likelihood):
        """this method adds new SMILES to the experience replay buffer if they are better scoring"""
        df = pd.DataFrame(
            {
                "smiles": smiles,
                "likelihood": neg_likelihood.cpu().detach().numpy(),
                "scores": scores.cpu().detach().numpy(),
            }
        )
        self.buffer = pd.concat([self.buffer, df])
        self.purge_buffer()

    def purge_buffer(self):
        """
        this method slices the experience replay buffer to keep only
        the top memory_size number of best scoring SMILES
        """
        unique_df = self.buffer.drop_duplicates(subset=["smiles"])
        sorted_df = unique_df.sort_values("scores", ascending=False)
        self.buffer = sorted_df.head(self.memory_size)
        # do not store SMILES with 0 reward
        self.buffer = self.buffer.loc[self.buffer["scores"] != 0.0]

    def augmented_memory_replay(self, n_jobs):
        """this method augments all the SMILES in the replay buffer and returns their likelihoods"""
        if len(self.buffer) > 0:
            smiles = self.buffer["smiles"].values
            # randomize the smiles
            randomized_smiles_list = get_randomized_smiles(
                smiles, self.prior, n_jobs=n_jobs
            )
            scores = self.buffer["scores"].values
            prior_likelihood = to_tensor(
                -self.prior.likelihood_smiles(randomized_smiles_list)
            )
            return randomized_smiles_list, scores, prior_likelihood
        else:
            return [], [], []

    def sample_buffer(self):
        """this method randomly samples replay_number of SMILES from the experience replay buffer"""
        sample_size = min(len(self.buffer), self.replay_number)
        if sample_size > 0:
            sampled = self.buffer.sample(sample_size)
            smiles = sampled["smiles"].values
            scores = sampled["scores"].values
            prior_likelihood = to_tensor(sampled["likelihood"].values)
            return smiles, scores, prior_likelihood
        else:
            return [], [], []


class AugmentedMemoryOptimizer:
    def __init__(
        self,
        batch_size=64,
        sigma=500,
        replay_buffer_size=64,
        augmented_memory=True,
        augmentation_rounds=2,
        learning_rate=0.0005,
        pretrained_model_path: Optional[str] = None,
        n_steps: int = 1_000_000,
    ):
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.sigma = sigma
        self.replay_buffer_size = replay_buffer_size
        self.augmented_memory = augmented_memory
        self.augmentation_rounds = augmentation_rounds
        self.learning_rate = learning_rate
        if pretrained_model_path is None:
            self.pretrained_model_path = os.path.join(
                os.path.dirname(__file__), "prior/Prior_guacamol_chembl_Epoch-5.ckpt"
            )
        else:
            self.pretrained_model_path = pretrained_model_path

    def generate_optimized_molecules(self, scoring_function, *args, **kwargs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_default_dtype(torch.float)
        torch.set_default_device(device)

        model = AugmentedMemory(
            pretrained_model_path=self.pretrained_model_path,
            batch_size=self.batch_size,
            sigma=self.sigma,
            replay_buffer_size=self.replay_buffer_size,
            augmented_memory=self.augmented_memory,
            augmentation_rounds=self.augmentation_rounds,
            learning_rate=self.learning_rate,
        )

        for _ in range(self.n_steps):
            # sample SMILES
            (
                seqs,
                smiles,
                agent_likelihood,
                _,
                _,
                _,
            ) = model.agent.sample_sequences_and_smiles(batch_size=model.batch_size)
            # switch signs
            agent_likelihood = -agent_likelihood
            # get prior likelihood
            prior_likelihood = -model.prior.likelihood(seqs)
            # get scores
            scores = to_tensor(np.array(scoring_function(smiles))).to(device)
            # get augmented likelihood
            augmented_likelihood = prior_likelihood + model.sigma * scores
            loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
            # add "normal" experience replay
            loss, agent_likelihood = self.add_experience_replay(
                model=model,
                loss=loss,
                agent_likelihood=agent_likelihood,
                prior_likelihood=prior_likelihood,
                smiles=smiles,
                scores=scores,
                override=True,
                n_jobs=scoring_function.n_jobs,
            )
            loss = loss.mean()
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            # perform augmented memory
            for _ in range(model.augmentation_rounds):
                # augment the *sampled* SMILES
                randomized_smiles_list = get_randomized_smiles(
                    smiles, model.prior, n_jobs=scoring_function.n_jobs
                )
                # get prior likelihood of randomized SMILES
                prior_likelihood = -model.prior.likelihood_smiles(
                    randomized_smiles_list
                )
                # get agent likelihood of randomized SMILES
                agent_likelihood = -model.agent.likelihood_smiles(
                    randomized_smiles_list
                )
                # compute augmented likelihood with the "new" prior likelihood using randomized SMILES
                augmented_likelihood = prior_likelihood + model.sigma * scores
                # compute loss
                loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
                # add augmented experience replay using randomized SMILES
                loss, agent_likelihood = self.add_experience_replay(
                    model=model,
                    loss=loss,
                    agent_likelihood=agent_likelihood,
                    prior_likelihood=prior_likelihood,
                    smiles=smiles,
                    scores=scores,
                    override=False,
                    n_jobs=scoring_function.n_jobs,
                )
                loss = loss.mean()
                model.optimizer.zero_grad()
                loss.backward()
                model.optimizer.step()

    @staticmethod
    def add_experience_replay(
        model,
        loss,
        agent_likelihood,
        prior_likelihood,
        smiles,
        scores,
        override,
        n_jobs,
    ):
        # use augmented memory
        if model.augmented_memory and not override:
            if len(model.replay_buffer.buffer) == 0:
                return loss, agent_likelihood
            else:
                (
                    exp_smiles,
                    exp_scores,
                    exp_prior_likelihood,
                ) = model.replay_buffer.augmented_memory_replay(n_jobs=n_jobs)
        # sample normally from the replay buffer
        else:
            (
                exp_smiles,
                exp_scores,
                exp_prior_likelihood,
            ) = model.replay_buffer.sample_buffer()
        # concatenate the loss with experience replay SMILES added
        if len(exp_smiles) > 0:
            exp_agent_likelihood = -model.agent.likelihood_smiles(exp_smiles)
            exp_augmented_likelihood = exp_prior_likelihood + model.sigma * to_tensor(
                exp_scores
            )
            exp_loss = torch.pow(
                (to_tensor(exp_augmented_likelihood) - exp_agent_likelihood), 2
            )
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        model.replay_buffer.add_to_buffer(smiles, scores, prior_likelihood)

        return loss, agent_likelihood
