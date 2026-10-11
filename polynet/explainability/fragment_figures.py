"""Display-only SME fragment tags and publication exports, without inference.

The cache's original fragment identifiers are retained in the mapping. RDKit
SMARTS are generated from those structures; F tags use sorted identifiers from
the whole saved analysis, so filtering models, sets or top-N cannot renumber them.
SVG contains vector RDKit bonds and glyphs; PNG is redrawn at the requested DPI,
never enlarged from a screen preview. Dimensions are millimetres and fonts points.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import math
import xml.etree.ElementTree as ET

from PIL import Image
from matplotlib.colors import to_rgb
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

_SVG = "http://www.w3.org/2000/svg"
_MM_TO_PT = 72 / 25.4
# Fine SVG coordinates keep physical dimensions/fonts independent of rounding
# to the integer panel sizes required by RDKit. These remain vector coordinates.
_SVG_UNITS_PER_MM = 100
_MAX_PNG_PIXELS = 40_000_000


def _fragment_molecule(fragment: str):
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(fragment)
        if mol is None:
            mol = Chem.MolFromSmarts(fragment)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError(f"Cannot depict fragment {fragment!r} as SMILES or SMARTS.")
    return mol


@lru_cache(maxsize=2048)
def _fragment_smarts(fragment: str) -> str:
    return Chem.MolToSmarts(_fragment_molecule(fragment))


def fragment_label_table(fragments) -> pd.DataFrame:
    """Return stable F tags, generated SMARTS and original cache identifiers.

    Supply every fragment in the saved analysis, not just the plotted subset.
    ``Fragment`` preserves the original SMILES (or SMARTS for a query fragment).
    """
    return pd.DataFrame(
        [
            {"Tag": f"F{i}", "SMARTS": _fragment_smarts(fragment), "Fragment": fragment}
            for i, fragment in enumerate(sorted(set(fragments)), 1)
        ],
        columns=["Tag", "SMARTS", "Fragment"],
    )


@dataclass(frozen=True)
class FragmentGridOptions:
    """Physical layout shared by vector and raster structure figures."""

    columns: int = 4
    mol_width_mm: float = 40
    mol_height_mm: float = 30
    horizontal_gap_mm: float = 4
    vertical_gap_mm: float = 4
    atom_font_pt: float = 10
    tag_font_pt: float = 10
    bond_length_mm: float = 5
    bond_width_pt: float = 0.7

    def __post_init__(self):
        if not isinstance(self.columns, int) or isinstance(self.columns, bool) or self.columns < 1:
            raise ValueError("The number of columns must be a positive integer.")
        for name in (
            "mol_width_mm",
            "mol_height_mm",
            "atom_font_pt",
            "tag_font_pt",
            "bond_length_mm",
            "bond_width_pt",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("horizontal_gap_mm", "vertical_gap_mm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")


def _layout(table: pd.DataFrame, options: FragmentGridOptions, units_per_mm: float):
    if table.empty:
        raise ValueError("Select at least one fragment to create a structure figure.")
    if not {"Tag", "Fragment"}.issubset(table.columns) or table["Tag"].duplicated().any():
        raise ValueError("The fragment table needs unique Tag values and a Fragment column.")
    columns = min(options.columns, len(table))
    rows = math.ceil(len(table) / columns)
    width = max(1, round(options.mol_width_mm * units_per_mm))
    label_mm = max(6, options.tag_font_pt * 1.8 / _MM_TO_PT)
    height = max(1, round((options.mol_height_mm + label_mm) * units_per_mm))
    gap_x = round(options.horizontal_gap_mm * units_per_mm)
    gap_y = round(options.vertical_gap_mm * units_per_mm)
    margin = round(3 * units_per_mm)
    total_width = margin * 2 + columns * width + (columns - 1) * gap_x
    total_height = margin * 2 + rows * height + (rows - 1) * gap_y
    return width, height, gap_x, gap_y, margin, columns, total_width, total_height, label_mm


def _draw_fragment(fragment, tag, width, height, label_mm, options, units_per_mm, svg, tag_color):
    mol = _fragment_molecule(fragment)
    mol.UpdatePropertyCache(strict=False)
    rdDepictor.Compute2DCoords(mol, canonOrient=True, forceRDKit=True)
    drawer_class = rdMolDraw2D.MolDraw2DSVG if svg else rdMolDraw2D.MolDraw2DCairo
    drawer = drawer_class(width, height)
    opts = drawer.drawOptions()
    opts.useBWAtomPalette()
    opts.clearBackground = True
    opts.padding = 0.08
    opts.fixedBondLength = options.bond_length_mm * units_per_mm
    opts.fixedFontSize = round(options.atom_font_pt * units_per_mm / _MM_TO_PT)
    opts.legendFontSize = round(options.tag_font_pt * units_per_mm / _MM_TO_PT)
    opts.legendFraction = label_mm * units_per_mm / height
    opts.setLegendColour(to_rgb(tag_color))
    opts.bondLineWidth = options.bond_width_pt * units_per_mm / _MM_TO_PT
    opts.scaleBondWidth = False
    opts.addStereoAnnotation = True
    drawer.DrawMolecule(mol, legend=str(tag))
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def fragment_structure_svg(
    table: pd.DataFrame, options: FragmentGridOptions | None = None, tag_colors: dict | None = None
) -> str:
    """A separate scalable vector figure, with each F tag beneath its structure.

    Row order follows the supplied table, usually distribution-plot order.
    RDKit draws the original fragment, preserving its chemistry and query features.
    ``tag_colors`` maps original fragment identifiers to colours; missing tags
    default to black. Only tag text changes colour, not atoms or bonds.
    """
    options = options or FragmentGridOptions()
    w, h, gx, gy, margin, cols, total_w, total_h, label_mm = _layout(
        table, options, _SVG_UNITS_PER_MM
    )
    root = ET.Element(
        f"{{{_SVG}}}svg",
        {
            "width": f"{total_w / _SVG_UNITS_PER_MM:g}mm",
            "height": f"{total_h / _SVG_UNITS_PER_MM:g}mm",
            "viewBox": f"0 0 {total_w} {total_h}",
        },
    )
    ET.SubElement(root, f"{{{_SVG}}}rect", width="100%", height="100%", fill="white")
    for i, row in enumerate(table.itertuples(index=False)):
        drawing = _draw_fragment(
            row.Fragment,
            row.Tag,
            w,
            h,
            label_mm,
            options,
            _SVG_UNITS_PER_MM,
            True,
            (tag_colors or {}).get(row.Fragment, "#000000"),
        )
        cell = ET.fromstring(drawing)
        cell.set("x", str(margin + (i % cols) * (w + gx)))
        cell.set("y", str(margin + (i // cols) * (h + gy)))
        cell.set("viewBox", f"0 0 {w} {h}")
        # Machine-readable identifiers supplement RDKit's vector legend glyphs.
        cell.set("data-fragment-tag", str(row.Tag))
        title = ET.SubElement(cell, f"{{{_SVG}}}title")
        title.text = f"{row.Tag}: {row.Fragment}"
        root.append(cell)
    ET.register_namespace("", _SVG)
    return ET.tostring(root, encoding="unicode")


def fragment_structure_png(
    table: pd.DataFrame,
    options: FragmentGridOptions | None = None,
    dpi: int = 600,
    tag_colors: dict | None = None,
) -> bytes:
    """Redraw at publication DPI and embed resolution metadata in a lossless PNG.

    Raster canvases above 40 million pixels are rejected before allocating memory;
    reduce DPI/layout size or use the vector SVG, which has no raster limit.
    ``tag_colors`` has the same meaning as for :func:`fragment_structure_svg`.
    """
    if not isinstance(dpi, int) or isinstance(dpi, bool) or dpi < 72 or dpi > 1200:
        raise ValueError("PNG resolution must be an integer between 72 and 1200 DPI.")
    options = options or FragmentGridOptions()
    units = dpi / 25.4
    w, h, gx, gy, margin, cols, total_w, total_h, label_mm = _layout(table, options, units)
    if total_w * total_h > _MAX_PNG_PIXELS:
        raise ValueError(
            "PNG exceeds 40 million pixels. Reduce DPI, molecule size or spacing, or download SVG."
        )
    canvas = Image.new("RGB", (total_w, total_h), "white")
    for i, row in enumerate(table.itertuples(index=False)):
        drawing = _draw_fragment(
            row.Fragment,
            row.Tag,
            w,
            h,
            label_mm,
            options,
            units,
            False,
            (tag_colors or {}).get(row.Fragment, "#000000"),
        )
        with Image.open(BytesIO(drawing)) as cell:
            canvas.paste(cell, (margin + (i % cols) * (w + gx), margin + (i // cols) * (h + gy)))
    output = BytesIO()
    canvas.save(output, format="PNG", dpi=(dpi, dpi))
    return output.getvalue()
