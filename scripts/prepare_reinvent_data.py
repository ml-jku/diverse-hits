# runs about 4 minutes
import os
import shutil

import pandas as pd

if not os.path.exists("reinvent-classifiers"):
    os.system("git clone https://github.com/tblaschke/reinvent-classifiers.git")
os.system(
    """sed -i "s/RDKIT_SMILES], axis=1, join='inner')/RDKIT_SMILES], axis=1)/g" reinvent-classifiers/prepare_from_excape.ipynb""" # noqa: E501
)
os.system(
    "jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute reinvent-classifiers/prepare_from_excape.ipynb" # noqa: E501
)

divopt_path = os.environ["DIVOPTPATH"]
scoring_function_base = os.path.join(divopt_path, "data/scoring_functions")

for target in ["DRD2"]:
    scoring_function_dir = os.path.join(scoring_function_base, target.lower())
    os.makedirs(scoring_function_dir, exist_ok=True)
    df = pd.read_pickle(os.path.join("reinvent-classifiers", f"{target}_df.pkl.gz"))
    df = df.reset_index()[["RDKIT_SMILES", "activity_label"]]
    df.columns = ["smiles", "label"]
    df.to_csv(os.path.join(scoring_function_dir, "all.txt"), index=False)

# delete reinvent-classifiers

shutil.rmtree("reinvent-classifiers")
