
import pandas as pd
from rdkit.Chem import AllChem
import numpy as np


def ZW6(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048).ToBitString()
def get_mol(smile):
    try:
        N = len(smile)
        nump = []
        for i in range(0, N):
            mol = AllChem.MolFromSmiles(smile[i])
            nump.append(mol)
        return nump
    except:
        nump = []
        mol = AllChem.MolFromSmiles(smile)
        nump.append(mol)
        return nump

def get_morgan_feature(smile):
    mol = get_mol(smile)
    data = []
    for i in range(len(mol)):
        try:
            data.append([smile[i], ZW6(mol[i])])
        except:
            continue
    jak_feature = pd.DataFrame(data, columns=['smiles','ZW6'])
    num_frame6 = []
    for i in range(len(jak_feature['ZW6'])):
        num_frame6.append([x for x in jak_feature['ZW6'][i]])
    jak_zw6 = pd.DataFrame(num_frame6,dtype=np.float)
    return jak_zw6