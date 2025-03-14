import gzip
import io
import logging
import os
import random
from multiprocessing import Pool

import numpy as np
import rdkit.Chem.Draw as Draw
import torch

# import torch.utils.tensorboard.summary as tbxs
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem.rdmolops import RenumberAtoms

from .vocabulary import SMILESTokenizer

_ST = SMILESTokenizer()

logger = logging.getLogger("utils")


def to_tensor(tensor):
    if isinstance(tensor, list):
        tensor = torch.tensor(tensor)
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)


def set_default_device_cuda(device="gpu"):
    """Sets the default device (cpu or cuda) used for all tensors."""
    if not torch.cuda.is_available() or (device == "cpu"):
        device = torch.device("cpu")
    elif (device in ["gpu", "cuda"]) and torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.cuda.is_available():  # Assume an index
        raise NotImplementedError
        device = torch.device(f"cuda:{int(device)}")

    torch.set_default_dtype(torch.float)
    torch.set_default_device(device)
    return device


def read_smiles(file_path):
    """Read a smiles file separated by \n"""
    if any(["gz" in ext for ext in os.path.basename(file_path).split(".")[1:]]):
        logger.debug("'gz' found in file path: using gzip")
        with gzip.open(file_path) as f:
            smiles = f.read().splitlines()
            smiles = [smi.decode("utf-8") for smi in smiles]
    else:
        with open(file_path, "rt") as f:
            smiles = f.read().splitlines()
    return smiles


def save_smiles(smiles, file_path):
    """Save smiles to a file path seperated by \n"""
    if (not os.path.exists(os.path.dirname(file_path))) and (
        os.path.dirname(file_path) != ""
    ):
        os.makedirs(os.path.dirname(file_path))
    if any(["gz" in ext for ext in os.path.basename(file_path).split(".")[1:]]):
        logger.debug("'gz' found in file path: using gzip")
        with gzip.open(file_path, "wb") as f:
            _ = [
                f.write((smi + "\n").encode("utf-8"))
                for smi in smiles
                if smi is not None
            ]
    else:
        with open(file_path, "wt") as f:
            _ = [f.write(smi + "\n") for smi in smiles if smi is not None]
    return


def fraction_valid_smiles(smiles):
    i = 0
    mols = []
    for smile in smiles:
        try:
            mol = Chem.MolFromSmiles(smile)
            if mol:
                i += 1
                mols.append(mol)
        except TypeError:  # None passed as smile
            pass
    fraction = 100 * i / len(smiles)
    return round(fraction, 2), mols


def add_mol(writer, tag, mol, global_step=None, walltime=None, size=(300, 300)):
    """
    Adds a molecule to the images section of Tensorboard.
    """
    image = Draw.MolToImage(mol, size=size)
    add_image(writer, tag, image, global_step, walltime)


def add_mols(
    writer,
    tag,
    mols,
    mols_per_row=1,
    legends=None,
    global_step=None,
    walltime=None,
    size_per_mol=(300, 300),
):
    """
    Adds molecules in a grid.
    """
    image = Draw.MolsToGridImage(
        mols, molsPerRow=mols_per_row, subImgSize=size_per_mol, legends=legends
    )
    add_image(writer, tag, image, global_step, walltime)


def add_image(writer, tag, image, global_step=None, walltime=None):
    """
    Adds an image from a PIL image.
    """
    channel = len(image.getbands())
    width, height = image.size

    output = io.BytesIO()
    image.save(output, format="png")
    image_string = output.getvalue()
    output.close()

    summary_image = tbxs.Summary.Image(
        height=height,
        width=width,
        colorspace=channel,
        encoded_image_string=image_string,
    )
    summary = tbxs.Summary(value=[tbxs.Summary.Value(tag=tag, image=summary_image)])
    writer.file_writer.add_summary(summary, global_step, walltime)


def randomize_smiles(
    smi,
    n_rand=10,
    random_type="restricted",
):
    """
    Returns a random SMILES given a SMILES of a molecule.
    :param smi: A SMILES string
    :param n_rand: Number of randomized smiles per molecule
    :param random_type: The type (unrestricted, restricted) of randomization performed.
    :return : A random SMILES string of the same molecule or None if the molecule is invalid.
    """
    assert random_type in [
        "restricted",
        "unrestricted",
    ], f"Type {random_type} is not valid"
    mol = Chem.MolFromSmiles(smi)

    if not mol:
        return None

    if random_type == "unrestricted":
        rand_smiles = []
        for i in range(n_rand):
            rand_smiles.append(
                Chem.MolToSmiles(
                    mol, canonical=False, doRandom=True, isomericSmiles=False
                )
            )
        return list(set(rand_smiles))

    if random_type == "restricted":
        rand_smiles = []
        for i in range(n_rand):
            new_atom_order = list(range(mol.GetNumAtoms()))
            random.shuffle(new_atom_order)
            random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
            rand_smiles.append(
                Chem.MolToSmiles(random_mol, canonical=False, isomericSmiles=False)
            )
        return rand_smiles


def mol2random_smiles(mol) -> str:
    """
    Returns a randomized SMILES given an RDKit Mol object.
    :param mol: An RDKit Mol object
    :return : A random SMILES string of the same molecule or None if the molecule is invalid.
    from reinvent-chemistry
    """
    new_atom_order = list(range(mol.GetNumHeavyAtoms()))
    # reinvent-chemistry uses random.shuffle
    # use np.random.shuffle for reproducibility since PMO fixes the np seed
    np.random.shuffle(new_atom_order)
    random_mol = RenumberAtoms(mol, newOrder=new_atom_order)
    return MolToSmiles(random_mol, canonical=False, isomericSmiles=False)


def smiles2random_smiles(smiles):
    # check if the SMILES can be parsed
    # if not, return the original SMILES
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return smiles
    randomized_smiles = mol2random_smiles(mol)
    return randomized_smiles


def get_randomized_smiles(smiles_list, prior, n_jobs) -> list:
    """takes a list of SMILES and returns a list of randomized SMILES"""
    if len(smiles_list) == 0:
        return []
    
    if n_jobs > 1:
        with Pool(n_jobs) as p:
            randomized_smiles_list = p.map(smiles2random_smiles, smiles_list)
    else:
        randomized_smiles_list = map(smiles2random_smiles, smiles_list)

    results = []
    for smiles, randomized_smiles in zip(smiles_list, randomized_smiles_list):
        try:
            tokens = _ST.tokenize(randomized_smiles)
            encoded = prior.vocabulary.encode(tokens)
            results.append(randomized_smiles)
        except KeyError:
            results.append(smiles)
    return results
