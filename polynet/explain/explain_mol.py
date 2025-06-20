from io import BytesIO
import math

from PIL import Image
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
from scipy.stats import gaussian_kde

from polynet.utils.chem_utils import fragment_and_match


def get_fragment_importance(mols, node_masks, explain_algorithm):

    frags_importances = {}

    for mol in mols:
        mol_idx = mol.idx
        frag_importance = node_masks[mol_idx][explain_algorithm]

        num_atoms = 0

        for smiles in mol.mols:

            frags = fragment_and_match(smiles)

            for frag_smiles, atom_indices in frags.items():

                if len(frag_smiles) < 3:
                    continue

                if frag_smiles not in frags_importances:
                    frags_importances[frag_smiles] = []

                for idx in atom_indices:
                    idx = [num_atoms + i for i in idx]
                    frag_score = np.sum(frag_importance[idx])
                    frags_importances[frag_smiles].append(frag_score)

            num_atoms += Chem.MolFromSmiles(smiles).GetNumAtoms()

    return frags_importances


def plot_mols_with_weights(
    smiles_list,
    weights_list,
    grid_size=None,
    cbar=True,
    legend=None,
    min_weight=None,
    max_weight=None,
    colormap="coolwarm",
):
    """
    Plots multiple molecules with their respective atomic weights in a single image with enhanced visuals.

    :param smiles_list: List of SMILES strings
    :param weights_list: List of lists containing weights for each molecule
    :param save_path: Path to save the image (optional)
    :param grid_size: Tuple specifying the grid layout (rows, cols), optional
    :param cbar: Boolean indicating whether to display a color bar
    :param legend: List of legend titles for each molecule (optional)
    """

    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    for mol, weights in zip(mols, weights_list):
        assert mol is not None, "Invalid SMILES string in input"
        assert (
            len(weights) == mol.GetNumAtoms()
        ), f"Number of weights must match number of atoms in molecule. Length of weights: {len(weights)}, Number of atoms: {mol.GetNumAtoms()}"

    num_mols = len(mols)
    if grid_size is None:
        cols = math.ceil(math.sqrt(num_mols))
        rows = math.ceil(num_mols / cols)
    else:
        rows, cols = grid_size

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 3, rows * 3), gridspec_kw={"right": 0.9 if cbar else 1.0}
    )
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    if min_weight is None or max_weight is None:
        vmin = min(min(weights) for weights in weights_list)
        vmax = max(max(weights) for weights in weights_list)
        max_val = max(abs(vmin), abs(vmax))

    elif min_weight is not None and max_weight is not None:
        vmin = min_weight
        vmax = max_weight
        max_val = max(abs(vmin), abs(vmax))

    cmap = plt.get_cmap(colormap)
    norm = plt.Normalize(vmin=-max_val, vmax=max_val)

    for i, (mol, weights) in enumerate(zip(mols, weights_list)):
        # Generate 2D coordinates for better visualization
        AllChem.Compute2DCoords(mol)

        d2d = Draw.MolDraw2DCairo(600, 600)
        Draw.SetACS1996Mode(d2d.drawOptions(), Draw.MeanBondLength(mol))
        d2d.drawOptions().clearBackground = True  # No grey box
        d2d.drawOptions().addStereoAnnotation = True  # Add stereo annotations
        d2d.drawOptions().addAtomIndices = False  # Customize atom indices
        d2d.drawOptions().bondLineWidth = 2  # Thicker bond lines
        # d2d.drawOptions().useBWAtomPalette()

        _ = SimilarityMaps.GetSimilarityMapFromWeights(
            mol=mol,
            weights=weights,
            draw2d=d2d,
            colorMap=colormap,
            alpha=0.7,
            contourLines=10,  # Add contour lines for better weight visualization
        )

        d2d.FinishDrawing()
        png_data = d2d.GetDrawingText()
        image = Image.open(BytesIO(png_data))

        axes[i].imshow(image, aspect="auto")
        axes[i].axis("off")
        if legend and i < len(legend):
            axes[i].set_title(legend[i], fontsize=12, pad=10)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if cbar:
        cbar_ax = fig.add_axes([0.91, 0.3, 0.015, 0.4])  # Adjusted for better proportion
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, label="Weight Scale")
        cbar.ax.tick_params(labelsize=10)

    plt.tight_layout(rect=[0, 0, 0.9 if cbar else 1, 1])

    return fig


def plot_mols_with_numeric_weights(
    smiles_list, weights_list, grid_size=None, legend=None, decimals=2
):
    """
    Plots multiple molecules and displays the atomic weights as text next to each atom.

    :param smiles_list: List of SMILES strings
    :param weights_list: List of lists containing weights for each molecule
    :param grid_size: Tuple specifying the grid layout (rows, cols), optional
    :param legend: List of legend titles for each molecule (optional)
    :param decimals: Number of decimal places to round weights in labels
    """
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    for mol, weights in zip(mols, weights_list):
        assert mol is not None, "Invalid SMILES string in input"
        assert (
            len(weights) == mol.GetNumAtoms()
        ), f"Number of weights must match number of atoms in molecule. Got {len(weights)} weights for {mol.GetNumAtoms()} atoms."

    num_mols = len(mols)
    if grid_size is None:
        cols = math.ceil(math.sqrt(num_mols))
        rows = math.ceil(num_mols / cols)
    else:
        rows, cols = grid_size

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, (mol, weights) in enumerate(zip(mols, weights_list)):
        AllChem.Compute2DCoords(mol)

        drawer = rdMolDraw2D.MolDraw2DCairo(600, 600)
        opts = drawer.drawOptions()
        opts.clearBackground = True
        opts.addStereoAnnotation = True
        opts.addAtomIndices = False
        opts.bondLineWidth = 2

        # Set weight as atom label using the correct C++ binding
        for idx, weight in enumerate(weights):
            label = f"{weight:.{decimals}f}"
            opts.atomLabels[idx] = label

        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        image = Image.open(BytesIO(drawer.GetDrawingText()))

        axes[i].imshow(image, aspect="auto")
        axes[i].axis("off")
        if legend and i < len(legend):
            axes[i].set_title(legend[i], fontsize=12, pad=10)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return fig


def plot_attribution_distribution(
    attribution_dict,
    figsize=(8, 10),
    neg_color="#1f77b4",
    pos_color="#ff7f0e",
    kde_bandwidth=0.2,
    linewidth=0.7,
    show_averages=True,
):
    """
    Plot attribution distribution with KDE for multiple values per functional group.

    Parameters:
    -----------
    attribution_dict : dict
        Keys are functional groups (str), values are lists of attribution values
    figsize : tuple, optional
        Figure size (width, height)
    neg_color : str, optional
        Color for negative attributions (blue default)
    pos_color : str, optional
        Color for positive attributions (orange default)
    kde_bandwidth : float, optional
        Bandwidth for KDE smoothing (default 0.2)
    linewidth : float, optional
        Width of distribution lines (default 0.7)
    show_averages : bool, optional
        Whether to show average markers (default True)

    Returns:
    --------
    matplotlib.figure.Figure
    """

    # Prepare data
    labels = list(attribution_dict.keys())
    values = list(attribution_dict.values())

    # Create figure with adjusted layout
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.3)  # Make room for y-axis labels

    # Calculate positions for each label
    y_pos = np.arange(len(labels))

    # Plot zero line
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-", zorder=1)

    # Plot each distribution
    verts_pos = []
    verts_neg = []

    for i, (label, vals) in enumerate(zip(labels, values)):
        vals = np.array(vals)

        # Skip if no data
        if len(vals) == 0:
            continue

        # Add small noise if all values are identical
        if len(vals) > 1 and np.allclose(vals, vals[0]):
            vals = vals + np.random.normal(0, 1e-6, size=len(vals))

        # Calculate average
        avg = np.mean(vals)

        # Create KDE for positive values
        pos_vals = vals[vals > 0]
        if len(pos_vals) > 1:
            try:
                kde_pos = gaussian_kde(pos_vals, bw_method=kde_bandwidth)
                x_pos = np.linspace(0, max(3, vals.max() * 1.5), 100)
                density_pos = kde_pos(x_pos)
                # Scale density for visualization
                density_pos = density_pos / (density_pos.max() + 1e-8) * 0.4
                # Create vertices for polygon
                verts_pos.append(list(zip(x_pos, i + density_pos)) + [(0, i)])
            except np.linalg.LinAlgError:
                pass  # Skip if KDE fails

        # Create KDE for negative values
        neg_vals = vals[vals < 0]
        if len(neg_vals) > 1:
            try:
                kde_neg = gaussian_kde(neg_vals, bw_method=kde_bandwidth)
                x_neg = np.linspace(min(-3, vals.min() * 1.5), 0, 100)
                density_neg = kde_neg(x_neg)
                # Scale density for visualization
                density_neg = density_neg / (density_neg.max() + 1e-8) * 0.4
                # Create vertices for polygon
                verts_neg.append(list(zip(x_neg, i - density_neg)) + [(0, i)])
            except np.linalg.LinAlgError:
                pass  # Skip if KDE fails

    # Plot positive distributions (orange)
    if verts_pos:
        poly_pos = PolyCollection(
            verts_pos,
            facecolors=pos_color,
            edgecolors=pos_color,
            linewidths=linewidth,
            alpha=0.7,
            zorder=2,
        )
        ax.add_collection(poly_pos)

    # Plot negative distributions (blue)
    if verts_neg:
        poly_neg = PolyCollection(
            verts_neg,
            facecolors=neg_color,
            edgecolors=neg_color,
            linewidths=linewidth,
            alpha=0.7,
            zorder=3,
        )
        ax.add_collection(poly_neg)

    # Add average markers if requested
    if show_averages:
        for i, vals in enumerate(values):
            if len(vals) > 0:
                avg = np.mean(np.array(vals))  # Ensure we use original values for average
                color = pos_color if avg > 0 else neg_color
                ax.plot(avg, i, "o", color=color, markersize=8, markeredgecolor="white", zorder=4)
                # Add average value text
                ax.text(
                    avg + 0.1 * np.sign(avg),
                    i,
                    f"{avg:.2f}",
                    va="center",
                    ha="left" if avg > 0 else "right",
                    fontsize=9,
                    zorder=5,
                )

    # Customize axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Attribution", fontsize=12)
    ax.set_title("Attribution Distribution", fontsize=14, pad=20)

    # Set symmetrical x-limits if needed
    xlim = max(3, np.abs(ax.get_xlim()).max())
    ax.set_xlim(-xlim, xlim)

    # Add grid lines
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    # Custom spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    return fig
