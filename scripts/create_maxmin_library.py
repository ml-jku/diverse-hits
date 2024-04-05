import os
import pickle
from time import time

import numpy as np
from rdkit import SimDivFilters  # type: ignore

from divopt.chem import morgan_from_smiles_list

print("Warning: this took 4 days for me. You can skip this step and use the precomputed maxmin order.")

guacamol_dir = "../data"
original_fn = os.path.join(guacamol_dir, "guacamol_v1_all.smiles")

with open(original_fn) as f:
    smiles_original = f.read().split("\n")

# save smiles in random order
np.random.seed(0)

fp_file = "../data/guacamol_v1_all_fingerprints.p"
if os.path.isfile(fp_file):
    with open(fp_file, "rb") as f:
        fps = pickle.load(f)
else:
    fps = morgan_from_smiles_list(smiles_original, n_jobs=6)
    with open(fp_file, "wb") as f:
        pickle.dump(fps, f)


mmp = SimDivFilters.MaxMinPicker()
fps_sub = fps[:]
start_time = time()
picks = mmp.LazyBitVectorPick(fps_sub, len(fps_sub), len(fps_sub))

hours = (time() - start_time) / 60 / 60
print(f"Took {hours} hours")

smiles_maxmin_order = [smiles_original[i] for i in picks]
smiles_maxmin_fn = os.path.join(guacamol_dir, "guacamol_v1_all_maxmin_order.smiles")
with open(smiles_maxmin_fn, "w") as f:
    f.write("\n".join(smiles_maxmin_order))
