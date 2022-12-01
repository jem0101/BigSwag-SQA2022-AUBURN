import numpy as np
import deepchem as dc

mols = [
    'C1=CC2=C(C=C1)C1=CC=CC=C21', 'O=C1C=CC(=O)C2=C1OC=CO2', 'C1=C[N]C=C1',
    'C1=CC=CC=C[C+]1', 'C1=[C]NC=C1', 'N[C@@H](C)C(=O)O', 'N[C@H](C)C(=O)O',
    'CC', 'O=C=O', 'C#N', 'CCN(CC)CC', 'CC(=O)O', 'C1CCCCC1', 'c1ccccc1'
]
print("Original set of molecules")
print(mols)

splitter = dc.splits.ScaffoldSplitter()
# TODO: This should be swapped for simpler splitter API once that's merged in.
dataset = dc.data.NumpyDataset(X=np.array(mols), ids=mols)
train, valid, test = splitter.train_valid_test_split(dataset)
# The return values are dc.data.Dataset objects so we need to extract
# the ids
print("Training set")
print(train)
print("Valid set")
print(valid)
print("Test set")
print(test)
