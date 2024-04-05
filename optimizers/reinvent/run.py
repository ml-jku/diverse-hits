import os
import numpy as np

from .utils import Variable, seq_to_smiles, unique
from .model import RNN
from .data_structs import Vocabulary, Experience
import torch

path_here = os.path.dirname(os.path.realpath(__file__))


class ReinventOptimizer:
    def __init__(
        self,
        learning_rate: float,
        sigma: float,
        batch_size: int,
        experience_replay: int,
    ):
        # super().__init__(args)
        # self.model_name = "reinvent"
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.batch_size = batch_size
        self.experience_replay = experience_replay
        
    def load_model(self):
        path_here = os.path.dirname(os.path.realpath(__file__))
        restore_prior_from = os.path.join(path_here, "data/Prior.ckpt")
        restore_agent_from = restore_prior_from
        voc = Vocabulary(init_from_file=os.path.join(path_here, "data/Voc"))

        Agent = RNN(voc)
        Agent.rnn.load_state_dict(torch.load(restore_agent_from))
        return Agent

    def generate_optimized_molecules(self, scoring_function, num_molecules=100):
        path_here = os.path.dirname(os.path.realpath(__file__))
        restore_prior_from = os.path.join(path_here, "data/Prior.ckpt")
        restore_agent_from = restore_prior_from
        voc = Vocabulary(init_from_file=os.path.join(path_here, "data/Voc"))

        Prior = RNN(voc)
        Agent = RNN(voc)

        # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
        # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
        # to the CPU.
        if torch.cuda.is_available():
            Prior.rnn.load_state_dict(
                torch.load(os.path.join(path_here, "data/Prior.ckpt"))
            )
            Agent.rnn.load_state_dict(torch.load(restore_agent_from))
        else:
            Prior.rnn.load_state_dict(
                torch.load(
                    os.path.join(path_here, "data/Prior.ckpt"),
                    map_location=lambda storage, loc: storage,
                )
            )
            Agent.rnn.load_state_dict(
                torch.load(
                    restore_agent_from, map_location=lambda storage, loc: storage
                )
            )

        # We dont need gradients with respect to Prior
        for param in Prior.rnn.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=self.learning_rate)

        # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
        # occur more often (which means the agent can get biased towards them). Using experience replay is
        # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
        experience = Experience(voc)

        print("Model initialized, starting training...")

        step = 0

        while True:
            # Sample from Agent
            seqs, agent_likelihood, _ = Agent.sample(self.batch_size)

            # Remove duplicates, ie only consider unique seqs
            unique_idxs = unique(seqs)
            seqs = seqs[unique_idxs]
            agent_likelihood = agent_likelihood[unique_idxs]

            # Get prior likelihood and score
            prior_likelihood, _ = Prior.likelihood(Variable(seqs))

            smiles_list = seq_to_smiles(seqs, voc)
            score = np.array(scoring_function.score_list(smiles_list))

            # Calculate augmented likelihood
            augmented_likelihood = (
                prior_likelihood.float() + self.sigma * torch.tensor(score).cuda()
            )
            loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

            # Experience Replay
            # First sample
            if self.experience_replay and len(experience) > self.experience_replay:
                exp_seqs, exp_score, exp_prior_likelihood = experience.sample(
                    self.experience_replay
                )
                exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
                exp_augmented_likelihood = exp_prior_likelihood + self.sigma * exp_score
                exp_loss = torch.pow(
                    (Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2
                )
                loss = torch.cat((loss, exp_loss), 0)
                agent_likelihood = torch.cat(
                    (agent_likelihood, exp_agent_likelihood), 0
                )

            # Then add new experience
            prior_likelihood = prior_likelihood.data.cpu().numpy()
            new_experience = zip(smiles_list, score, prior_likelihood)
            experience.add_experience(new_experience)

            # Calculate loss
            loss = loss.mean()

            # Add regularizer that penalizes high likelihood for the entire sequence
            loss_p = -(1 / agent_likelihood).mean()
            loss += 5 * 1e3 * loss_p

            # Calculate gradients and make an update to the network weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
