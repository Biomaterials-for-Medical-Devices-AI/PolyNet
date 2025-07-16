from collections import defaultdict
from typing import Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS


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
        try:
            Chem.SanitizeMol(frag)
        except Exception as e:
            print(f"Error sanitizing fragment: {Chem.MolToSmiles(raw_frag)}. Exception: {e}")
            continue

        substruct_matches = mol.GetSubstructMatches(frag, uniquify=True)
        if substruct_matches:
            frag_smiles = Chem.MolToSmiles(frag)
            matches[frag_smiles] = [list(match) for match in substruct_matches]

    return matches


def count_atom_types(df: pd.DataFrame, col_name: str) -> dict:
    """
    Count how many molecules contain at least one atom of each type in a SMILES column.

    Args:
        df (pd.DataFrame): DataFrame containing SMILES strings.
        col_name (str): Name of the column containing SMILES.

    Returns:
        dict: Mapping from atom type (str) to count of molecules containing it.
    """
    atom_counts = defaultdict(int)

    for smiles in df[col_name]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue  # skip invalid SMILES

        atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
        for symbol in atom_symbols:
            atom_counts[symbol] += 1

    return dict(atom_counts)


def filter_by_atom_type(df: pd.DataFrame, col_name: str, atom_symbol: str) -> pd.DataFrame:
    """
    Return rows where the SMILES contains at least one atom of the specified type.

    Args:
        df (pd.DataFrame): DataFrame with SMILES strings.
        col_name (str): Name of the column with SMILES.
        atom_symbol (str): Atom symbol to search for (e.g., 'P', 'N', 'O').

    Returns:
        pd.DataFrame: Filtered DataFrame with matching rows.
    """
    mask = []

    for smiles in df[col_name]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mask.append(False)
            continue

        symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
        mask.append(atom_symbol in symbols)

    return df[mask].copy()
