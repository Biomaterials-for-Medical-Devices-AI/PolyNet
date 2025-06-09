from typing import Optional
from rdkit.Chem import BRICS
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


def sanitize_fragment(frag):
    """Remove dummy atoms from a BRICS fragment to make it matchable."""
    editable = Chem.EditableMol(frag)
    atoms_to_remove = [atom.GetIdx() for atom in frag.GetAtoms() if atom.GetAtomicNum() == 0]

    # Remove from the end to preserve indices
    for idx in sorted(atoms_to_remove, reverse=True):
        editable.RemoveAtom(idx)
    return editable.GetMol()


def fragment_and_match(smiles):
    """
    BRICS-decomposes a molecule and matches cleaned fragments back to the original.

    Parameters:
        smiles (str): SMILES string of the molecule.

    Returns:
        dict: {fragment_smiles: [list of atom index lists in original mol]}
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES provided.")

    # Get BRICS fragments with dummy atoms
    raw_frags = list(BRICS.BRICSDecompose(mol, returnMols=True))

    matches = {}
    for raw_frag in raw_frags:
        frag = sanitize_fragment(raw_frag)
        Chem.SanitizeMol(frag)

        substruct_matches = mol.GetSubstructMatches(frag, uniquify=True)
        if substruct_matches:
            frag_smiles = Chem.MolToSmiles(frag)
            matches[frag_smiles] = [list(match) for match in substruct_matches]

    return matches
