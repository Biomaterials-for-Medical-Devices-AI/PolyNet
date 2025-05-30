from io import BytesIO
import math

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import SimilarityMaps


def plot_mols_with_weights(
    smiles_list,
    weights_list,
    grid_size=None,
    cbar=False,
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
        print(weights_list)
        vmin = min(min(weights) for weights in weights_list)
        print("vmin", vmin)
        vmax = max(max(weights) for weights in weights_list)
        print("vmax", vmax)
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

        print("weights", len(weights))
        print("mol", mol)
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

    return fig
    # plt.tight_layout(rect=[0, 0, 0.9 if cbar else 1, 1])
    # if save_path:
    #     plt.savefig(save_path, dpi=600, bbox_inches="tight", transparent=True)
    # else:
    #     plt.show()

    # plt.close()
    # plt.clf()
