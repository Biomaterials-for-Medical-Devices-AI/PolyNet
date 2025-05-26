from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
import streamlit as st
import plotly.graph_objects as go

# Approximate covalent atomic radii in Ångströms (used for visual scaling)
# Source: average covalent radii
atomic_radii = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
    "Si": 1.11,
    "B": 0.85,
}

# Atom colors
atom_colors = {
    "H": "white",
    "C": "gray",
    "N": "blue",
    "O": "red",
    "F": "green",
    "Cl": "green",
    "Br": "darkred",
    "I": "purple",
    "S": "yellow",
    "P": "orange",
    "Si": "orange",
    "B": "salmon",
}


def plot_molecule_3d(smiles: str):
    """
    Generate a 3D plot of a molecule from SMILES using Plotly and RDKit, with atoms sized by covalent radius.

    Args:
        smiles (str): SMILES string of the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES string.")
        return

    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
        st.error("3D embedding failed.")
        return

    AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    atom_coords = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    # Base size scale factor for atoms
    scale = 30

    atom_trace = go.Scatter3d(
        x=[pos.x for pos in atom_coords],
        y=[pos.y for pos in atom_coords],
        z=[pos.z for pos in atom_coords],
        mode="markers+text",
        marker=dict(
            size=[scale * atomic_radii.get(symbol, 0.8) for symbol in atom_symbols],
            color=[atom_colors.get(symbol, "lightgray") for symbol in atom_symbols],
            opacity=0.95,
            line=dict(width=0.5, color="black"),
        ),
        text=atom_symbols,
        textposition="top center",
    )

    bond_traces = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        xi, yi, zi = atom_coords[i].x, atom_coords[i].y, atom_coords[i].z
        xj, yj, zj = atom_coords[j].x, atom_coords[j].y, atom_coords[j].z
        bond_traces.append(
            go.Scatter3d(
                x=[xi, xj, None],
                y=[yi, yj, None],
                z=[zi, zj, None],
                mode="lines",
                line=dict(color="black", width=3),
            )
        )

    fig = go.Figure(data=[atom_trace] + bond_traces)
    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)
