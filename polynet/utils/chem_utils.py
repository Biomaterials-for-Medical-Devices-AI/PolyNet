from collections import defaultdict
from typing import Optional

from canonicalize_psmiles.canonicalize import canonicalize as ext_canonicalize
import pandas as pd
from psmiles import PolymerSmiles
from rdkit import Chem
from rdkit.Chem import BRICS

from polynet.options.enums import AtomFeatures, BondFeatures, StringRepresentation


class PS(PolymerSmiles):
    def __init__(self, psmiles, deactivate_warnings=True):
        super().__init__(psmiles, deactivate_warnings)
        self.deactivate_warnings = deactivate_warnings

    @property
    def canonicalize(self) -> PolymerSmiles:
        """Canonicalize the PSMILES string

        Returns:
            PolymerSmiles: canonicalized PSMILES string
        """
        return PolymerSmiles(
            ext_canonicalize(self.psmiles), deactivate_warnings=self.deactivate_warnings
        )


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


def identify_psmiles(smiles: str):
    return smiles.count("*") >= 2


def determine_string_representation(df, smiles_cols):
    for col in smiles_cols:
        if not df[col].apply(identify_psmiles).all():
            return StringRepresentation.Smiles
    return StringRepresentation.PSmiles


def canonicalise_psmiles(psmiles: str) -> Optional[str]:
    try:
        psmiles = PS(psmiles=psmiles, deactivate_warnings=True).canonicalize
        return psmiles.psmiles
    except Exception as e:
        print(f"Error converting SMILES to canonical form: '{psmiles}'. Exception: {e}")
        return None


def check_smiles_cols(col_names, df):

    invalid_smiles = {}
    for col in col_names:
        invalid_smiles[col] = []
        for smiles in df[col]:
            if not check_smiles(smiles):
                invalid_smiles[col].append(str(smiles))

    # Remove empty lists
    invalid_smiles = {k: v for k, v in invalid_smiles.items() if v}

    return invalid_smiles


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


def count_atom_property_frequency(
    df: pd.DataFrame, col_name: str, property: AtomFeatures
) -> pd.DataFrame:
    """
    Count how many molecules contain at least one atom with each unique property value.

    Args:
        df (pd.DataFrame): DataFrame containing SMILES strings.
        col_name (str): Column name with SMILES.
        property (AtomFeatures): Atom-level property to analyze.

    Returns:
        pd.DataFrame: Frequency of molecules containing atoms with each unique property value.
    """
    if not hasattr(Chem.Atom, property):
        raise ValueError(f"{property} is not a valid Atom property method in RDKit")

    counts = defaultdict(int)

    for smiles in df[col_name]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # Collect all unique values of the property in this molecule
        values = set()
        for atom in mol.GetAtoms():
            val = getattr(atom, property)()
            values.add(val)

        # Count each unique value once per molecule
        for val in values:
            counts[val] += 1

    return counts


def count_bond_property_frequency(df: pd.DataFrame, col_name: str, property: BondFeatures) -> dict:
    """
    Count how many molecules contain at least one bond with each unique property value.

    Args:
        df (pd.DataFrame): DataFrame containing SMILES strings.
        col_name (str): Column name with SMILES.
        property (BondFeatures): Bond-level property to analyze.

    Returns:
        dict: Mapping from property value to count of molecules containing it.
    """
    if not hasattr(Chem.Bond, property):
        raise ValueError(f"{property} is not a valid Bond property method in RDKit")

    counts = defaultdict(int)

    for smiles in df[col_name]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # Collect all unique values of the bond property in this molecule
        values = set()
        for bond in mol.GetBonds():
            val = getattr(bond, property)()
            values.add(val)

        # Count each unique value once per molecule
        for val in values:
            counts[val] += 1

    return counts


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
