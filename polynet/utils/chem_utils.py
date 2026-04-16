from collections import defaultdict
import logging
from typing import Optional

from canonicalize_psmiles.canonicalize import canonicalize
import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold

from polynet.config.enums import AtomFeature, BondFeature, FragmentationMethod, StringRepresentation

logger = logging.getLogger(__name__)


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
            return StringRepresentation.SMILES
    return StringRepresentation.PSMILES


def canonicalise_psmiles(psmiles: str) -> Optional[str]:
    try:
        psmiles = canonicalize(psmiles)
        return psmiles
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


def fragment_and_match(
    smiles, fragmentation_approach: FragmentationMethod = FragmentationMethod.BRICS
):
    """
    Fragment a molecule using the chosen method and return mapping to atom indices.

    Parameters
    ----------
    smiles : str
    mode : str, one of ["brics", "functional_groups", "murcko"]

    Returns
    -------
    dict
        {fragment_smiles : [ [atom indices], ... ] }
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES input.")

    match fragmentation_approach:

        case FragmentationMethod.BRICS:
            frags = _fragments_brics(mol)

        case FragmentationMethod.MurckoScaffold:
            frags = _fragments_murcko(mol)

        # TODO: fix these fragment approaches to follow the new behaviour.
        # case "functional_groups":
        #     frags = _fragments_functional_groups(mol)

        case _:
            raise ValueError(f"Unknown fragmentation mode: {fragmentation_approach}")

    # ----- match fragments back to original -----

    return frags


def _fragments_brics(mol) -> dict:
    """
    Fragment a molecule using BRICS rules and return a direct atom-index mapping.

    Bonds are broken with ``FragmentOnBonds``, and the resulting atom-to-fragment
    mapping is read from ``GetMolFrags(..., fragsMolAtomMapping=...)``.  This
    avoids substructure matching entirely: atom indices come straight from the
    bond-breaking operation and are guaranteed to correspond to the original
    molecule's atom numbering.

    ``FragmentOnBonds`` appends dummy atoms (attachment-point markers) with
    indices ``>= mol.GetNumAtoms()``.  These are stripped from the mapping and
    from each fragment SMILES via ``_eliminate_dummy`` before canonicalisation.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        A sanitized RDKit molecule.

    Returns
    -------
    dict
        ``{canonical_fragment_smiles: [[atom_indices], ...]}``.
        Atom indices are 0-based positions in *mol*.  Fragments that appear
        more than once (e.g. repeated ring systems) accumulate as separate
        lists under the same SMILES key.
    """
    bond_indices = [
        mol.GetBondBetweenAtoms(a1, a2).GetIdx()
        for (a1, a2), _ in BRICS.FindBRICSBonds(mol)
    ]

    # No breakable bonds — the whole molecule is a single fragment
    if not bond_indices:
        smi = Chem.MolToSmiles(mol, canonical=True)
        return {smi: [list(range(mol.GetNumAtoms()))]}

    frag_mol = Chem.FragmentOnBonds(mol, bond_indices)

    frags_map: list = []
    frags = Chem.GetMolFrags(
        frag_mol, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=frags_map
    )

    # Strip dummy-atom indices (>= original atom count) introduced by FragmentOnBonds
    n_atoms = mol.GetNumAtoms()
    frags_map = [[i for i in mapping if i < n_atoms] for mapping in frags_map]

    frag_smiles = [Chem.MolToSmiles(_eliminate_dummy(f), canonical=True) for f in frags]

    frag_dict: dict = defaultdict(list)
    for smi, idxs in zip(frag_smiles, frags_map):
        frag_dict[smi].append(idxs)

    return dict(frag_dict)


# def _fragments_functional_groups(mol):

#     frags = []
#     fg_smarts = GetFunctionalGroupSmarts()

#     for name, smarts in fg_smarts.items():
#         patt = Chem.MolFromSmarts(smarts)
#         if patt is None:
#             continue

#         if mol.HasSubstructMatch(patt):
#             frags.append(patt)

#     return frags


def _fragments_murcko(mol) -> dict:
    """
    Fragment a molecule by cutting scaffold–sidechain bonds and return a direct
    atom-index mapping.

    The Murcko scaffold (ring systems joined by linker chains) is identified with
    ``MurckoScaffold.GetScaffoldForMol``.  The scaffold atoms are mapped back to
    the original molecule via a single substructure match; every bond that crosses
    the scaffold–sidechain boundary is then cut with ``FragmentOnBonds`` and the
    resulting atom-to-fragment mapping is read from ``fragsMolAtomMapping`` — no
    further substructure matching is needed.

    This mirrors the implementation of ``_fragments_brics`` so the two methods
    produce the same dict contract and can be used interchangeably in the
    masking explainability pipeline.

    Edge cases
    ----------
    * **Acyclic molecule** (empty scaffold): returned as a single fragment.
    * **Scaffold match failure**: returned as a single fragment.
    * **No boundary bonds** (entire molecule is the scaffold): returned as a
      single fragment.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        A sanitized RDKit molecule.

    Returns
    -------
    dict
        ``{canonical_fragment_smiles: [[atom_indices], ...]}``.
        Atom indices are 0-based positions in *mol*.  Sidechain fragments that
        appear more than once accumulate as separate lists under the same key.
    """
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)

    # Acyclic molecule — scaffold is empty
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        smi = Chem.MolToSmiles(mol, canonical=True)
        return {smi: [list(range(mol.GetNumAtoms()))]}

    # Map scaffold atoms back to original molecule indices
    match = mol.GetSubstructMatch(scaffold)
    if not match:
        smi = Chem.MolToSmiles(mol, canonical=True)
        return {smi: [list(range(mol.GetNumAtoms()))]}

    scaffold_atoms = set(match)

    # Bonds that straddle the scaffold–sidechain boundary
    cut_bond_indices = [
        bond.GetIdx()
        for bond in mol.GetBonds()
        if (bond.GetBeginAtomIdx() in scaffold_atoms) != (bond.GetEndAtomIdx() in scaffold_atoms)
    ]

    # Entire molecule is the scaffold — nothing to cut
    if not cut_bond_indices:
        smi = Chem.MolToSmiles(mol, canonical=True)
        return {smi: [list(range(mol.GetNumAtoms()))]}

    frag_mol = Chem.FragmentOnBonds(mol, cut_bond_indices)

    frags_map: list = []
    frags = Chem.GetMolFrags(
        frag_mol, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=frags_map
    )

    # Strip dummy-atom indices (>= original atom count) introduced by FragmentOnBonds
    n_atoms = mol.GetNumAtoms()
    frags_map = [[i for i in mapping if i < n_atoms] for mapping in frags_map]

    frag_smiles = [Chem.MolToSmiles(_eliminate_dummy(f), canonical=True) for f in frags]

    frag_dict: dict = defaultdict(list)
    for smi, idxs in zip(frag_smiles, frags_map):
        frag_dict[smi].append(idxs)

    return dict(frag_dict)


def _eliminate_dummy(mol):
    for a in reversed([a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]):
        mol = Chem.RWMol(mol)
        mol.RemoveAtom(a)
        mol = mol.GetMol()
    return mol


def count_atom_property_frequency(
    df: pd.DataFrame, col_name: str, property: AtomFeature
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


def count_bond_property_frequency(df: pd.DataFrame, col_name: str, property: BondFeature) -> dict:
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
