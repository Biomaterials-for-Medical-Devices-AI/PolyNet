from typing import Optional

from rdkit import Chem


def check_smiles(smiles: str) -> bool:
    """
    Check if the given SMILES string is valid.

    Args:
        smiles (str): The SMILES string to check.

    Returns:
        bool: True if the SMILES string is valid, False otherwise.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception as e:
        print(f"Error parsing SMILES: '{smiles}'. Exception: {e}")
        return False


def canonicalise_smiles(smiles: str) -> Optional[str]:
    """
    Convert a SMILES string to its canonical form.

    Args:
        smiles (str): The SMILES string to convert.

    Returns:
        Optional[str]: The canonical SMILES string if valid, otherwise None.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        print(f"Error converting SMILES to canonical form: '{smiles}'. Exception: {e}")
        return None
