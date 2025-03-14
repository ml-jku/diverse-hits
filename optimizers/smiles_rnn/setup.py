from setuptools import setup, find_packages

setup(
    name='SMILES-RNN',
    version='1.0',
    packages=find_packages(),
    license='MIT',
    author='Morgan Thomas',
    author_email='morganthomas263@gmail.com',
    description='A Generic SMILES-RNN, modified from and based off reinvent 2.0',
    scripts=['scripts/sample_model.py', 'scripts/train_prior.py', 'scripts/fine_tune.py', 'scripts/reinforcement_learning.py',
     'utility_scripts/deep_smiles.py', 'utility_scripts/randomize_smiles.py']
)
