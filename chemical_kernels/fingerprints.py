from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem
import numpy as np



def fps_to_gram_matrix(fps, N):
    # Tanimoto
    sim = []
    for i in range(N):
        sim.append([])
        for j in range(N):
            sim[i].append(DataStructs.FingerprintSimilarity(fps[i],fps[j]))
    gram = np.array(sim)
    return (gram)



def smiles_to_mols(smiles):
    mols = []
    for i in range(len(smiles)):
        mols.append(Chem.MolFromSmiles(smiles[i]))
        
    return (mols)


def daylight_fps_tanimoto(smiles, maxPath = 7):
    mols = smiles_to_mols(smiles)
    N = len(mols)
    # compute daylight fingerprints
    fps = [FingerprintMols.FingerprintMol(mol, minPath =1, maxPath = maxPath, fpSize = 2048, bitsPerHash = 2,                                           useHs = True, tgtDensity = 0, minSize = 128) for mol in mols]
    gram = fps_to_gram_matrix(fps, N)
    return (gram)


def morgan_fps_tanimoto(smiles):
    mols = smiles_to_mols(smiles)
    N = len(mols)
    # compute morgan fingerprints
    m_fps = [AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048) for mol in mols]
    gram = fps_to_gram_matrix(m_fps, N)
    return (gram)

