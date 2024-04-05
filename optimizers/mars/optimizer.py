import os
import torch
import random
from rdkit import Chem, RDLogger
import sys

path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
# sys.path.append(".")
from .proposal.models.editor_basic import BasicEditor
from .proposal.proposal import Proposal_Random, Proposal_Editor, Proposal_Mix
from .sampler import Sampler_SA, Sampler_MH, Sampler_Recursive
from .datasets.utils import load_mols, load_smi_file

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


config = {
    "train": True,
    "num_mols": 64,
    "batch_size": 128,
    "n_layers": 3,
    "alpha": 0.95,
    "lr": 0.0005,
    "n_atom_feat": 17,
    "n_bond_feat": 5,
    "n_node_hidden": 64,
    "n_edge_hidden": 128,
    "device": "cpu",
    "data_dir": os.path.join(path_here, "data/"),
    "root_dir": os.path.join(path_here, "proposal/"),
    "sampler": "sa",
    "proposal": "editor",
    "editor_dir": None,
    "mols_ref": None,
    "mols_init": "chembl_10k.txt",
    "vocab": "chembl",
    "vocab_size": 1000,
    "beta": 0.05,
    "max_size": 40,
    "dataset_size": 50000,
}


class MarsOptimizer:
    def __init__(
        self,
        num_mols=64,
        batch_size=128,
        n_layers=3,
        sampler="sa",
        proposal="editor",
        n_jobs:int=8,
        smi_file=None,
    ):
        self.config = config
        self.config["num_mols"] = num_mols
        self.config["batch_size"] = batch_size
        self.config["n_layers"] = n_layers
        self.config["sampler"] = sampler
        self.config["proposal"] = proposal
        self.smi_file = smi_file
        self.n_jobs = n_jobs

    def generate_optimized_molecules(self, scoring_function, *args, **kwargs):
        config = self.config
        config["device"] = torch.device(config["device"])

        ### estimator
        # if config['mols_ref']:
        #     config['mols_ref'] = load_mols(config['data_dir'], config['mols_ref'])

        ### proposal
        editor = (
            BasicEditor(config).to(config["device"])
            if not config["proposal"] == "random"
            else None
        )
        if config["editor_dir"] is not None:  # load pre-trained editor
            path = os.path.join(
                config["root_dir"], config["editor_dir"], "model_best.pt"
            )
            editor.load_state_dict(
                torch.load(path, map_location=torch.device(config["device"]))
            )
            print("successfully loaded editor model from %s" % path)
        if config["proposal"] == "random":
            proposal = Proposal_Random(config)
        elif config["proposal"] == "editor":
            proposal = Proposal_Editor(config, editor)
        elif config["proposal"] == "mix":
            proposal = Proposal_Mix(config, editor)

        ### sampler
        if config["sampler"] == "re":
            sampler = Sampler_Recursive(config, proposal, scoring_function)
        elif config["sampler"] == "sa":
            sampler = Sampler_SA(config, proposal, scoring_function)
        elif config["sampler"] == "mh":
            sampler = Sampler_MH(config, proposal, scoring_function)

        ### sampling
        if self.smi_file:
            mols = load_smi_file(self.smi_file, config["num_mols"])
        elif config["mols_init"]:
            mols = load_mols(config["data_dir"], config["mols_init"])
        else:
            raise ValueError("Must provide either smi_file or mols_init")        
        mols = random.choices(mols, k=config["num_mols"])
        mols_init = mols[: config["num_mols"]]
            
        sampler.sample(mols_init)
