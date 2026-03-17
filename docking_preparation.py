from rdkit import Chem
from rdkit.Chem import AllChem


def generate_3d_structure(smiles):

    mol = Chem.MolFromSmiles(smiles)

    mol = Chem.AddHs(mol)

    AllChem.EmbedMolecule(mol)

    AllChem.UFFOptimizeMolecule(mol)

    return mol