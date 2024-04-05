#!/bin/bash
# add the following paths to the pythonpath
echo export DIVOPTPATH=$(pwd) >> ~/.bashrc
source ~/.bashrc
pip install -e .

# create main environment divopt
conda env create -f envs/divopt.yml
conda activate divopt
pip install -e optimizers/smiles_rnn/
pip install -e .
conda deactivate

# create gflownet environment
conda env create -f envs/gflownet.yml
conda activate gflownet
pip install torch==1.13.1 # gflownet install needs torch already installed or it will fail
pip install -e ./optimizers/gflownet_recursion --find-links https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install -e .
conda deactivate

# download data
wget -O data/guacamol_v1_all.smiles https://ndownloader.figshare.com/files/13612745               
wget -O data/guacamol_v1_test.smiles https://ndownloader.figshare.com/files/13612757
wget -O data/guacamol_v1_valid.smiles https://ndownloader.figshare.com/files/13612766
wget -O data/guacamol_v1_train.smiles https://ndownloader.figshare.com/files/13612760

bunzip2 -d -k data/guacamol_v1_all_maxmin_order.smiles.bz2 
