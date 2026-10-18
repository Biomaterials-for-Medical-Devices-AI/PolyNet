"""Page 6 SME fragment-label mappings and separate publication structure figures."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from polynet.app.components.images import show_image_full_width
from polynet.config.enums import ImportanceNormalisationMethod
from polynet.explainability.fragment_figures import (
    FragmentGridOptions,
    fragment_label_table,
    fragment_structure_png,
    fragment_structure_svg,
)
from polynet.explainability.masking import fragment_attributions_to_distribution
from polynet.explainability.visualization import ordered_attribution_fragments


def saved_fragment_table(analysis: dict) -> pd.DataFrame:
    """One fixed mapping for the whole loaded analysis, unaffected by GUI filtering."""
    if "_fragment_label_table" not in analysis:
        args = analysis["arguments"]
        scores = fragment_attributions_to_distribution(
            args["combined_explanations"],
            args["model_names"],
            args.get("target_class"),
            args["fragmentation_approach"],
            ImportanceNormalisationMethod.NoNormalisation,
        )
        analysis["_fragment_label_table"] = fragment_label_table(scores)
    return analysis["_fragment_label_table"].copy()


@st.cache_data(show_spinner=False, max_entries=4)
def _structure_svg(
    table: pd.DataFrame, options: FragmentGridOptions, tag_colors: dict | None = None
) -> str:
    return fragment_structure_svg(table, options, tag_colors)


@st.cache_data(show_spinner=False, max_entries=2)
def _structure_png(
    table: pd.DataFrame, options: FragmentGridOptions, dpi: int, tag_colors: dict | None = None
) -> bytes:
    return fragment_structure_png(table, options, dpi, tag_colors)


def fragment_exports(
    result, table: pd.DataFrame, top_n: int | None, prefix: str, tag_colors: dict | None = None
) -> None:
    """Export the plotted F tags, then optionally draw their separate structure key."""
    fragments = ordered_attribution_fragments(result.attribution_dict, top_n)
    plotted = table.set_index("Fragment").loc[fragments].reset_index()
    plotted = plotted[["Tag", "SMARTS", "Fragment"]]
    st.caption(
        "F tags stay fixed within this saved analysis. The exports contain exactly the fragments shown, in plot order; gaps in F numbering are expected after filtering. Fragment retains the original cache SMILES."
    )
    st.download_button(
        "Download F-to-SMARTS CSV",
        data=plotted.to_csv(index=False).encode("utf-8"),
        file_name="sme_fragment_mapping.csv",
        mime="text/csv",
        key=f"{prefix}_csv",
    )
    if st.checkbox("Show fragment mapping table", key=f"{prefix}_table"):
        st.dataframe(plotted, hide_index=True)
    if not st.checkbox("Create separate fragment structure figure", key=f"{prefix}_structures"):
        return
    with st.expander("Fragment structure figure settings", expanded=True):
        columns = int(
            st.number_input(
                "Structures per row", min_value=1, max_value=12, value=4, key=f"{prefix}_columns"
            )
        )
        dimensions = st.columns(2)
        width = dimensions[0].number_input(
            "Molecule panel width (mm)",
            min_value=15.0,
            max_value=100.0,
            value=40.0,
            step=1.0,
            key=f"{prefix}_width",
        )
        height = dimensions[1].number_input(
            "Molecule panel height (mm)",
            min_value=15.0,
            max_value=100.0,
            value=30.0,
            step=1.0,
            key=f"{prefix}_height",
        )
        spacing = st.columns(2)
        gap_x = spacing[0].number_input(
            "Horizontal spacing (mm)",
            min_value=0.0,
            max_value=30.0,
            value=4.0,
            step=1.0,
            key=f"{prefix}_gap_x",
        )
        gap_y = spacing[1].number_input(
            "Vertical spacing (mm)",
            min_value=0.0,
            max_value=30.0,
            value=4.0,
            step=1.0,
            key=f"{prefix}_gap_y",
        )
        fonts = st.columns(2)
        atom_font = fonts[0].number_input(
            "Atom label font (pt)",
            min_value=6.0,
            max_value=24.0,
            value=10.0,
            step=1.0,
            key=f"{prefix}_atom_font",
        )
        tag_font = fonts[1].number_input(
            "F tag font (pt)",
            min_value=6.0,
            max_value=24.0,
            value=10.0,
            step=1.0,
            key=f"{prefix}_tag_font",
        )
        bonds = st.columns(2)
        bond_length = bonds[0].number_input(
            "Bond length (mm)",
            min_value=2.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            key=f"{prefix}_bond_length",
            help="RDKit reduces this length when needed to fit a large fragment inside its panel.",
        )
        bond_width = bonds[1].number_input(
            "Bond line width (pt)",
            min_value=0.3,
            max_value=2.0,
            value=0.7,
            step=0.1,
            key=f"{prefix}_bond_width",
        )
    options = FragmentGridOptions(
        columns=columns,
        mol_width_mm=width,
        mol_height_mm=height,
        horizontal_gap_mm=gap_x,
        vertical_gap_mm=gap_y,
        atom_font_pt=atom_font,
        tag_font_pt=tag_font,
        bond_length_mm=bond_length,
        bond_width_pt=bond_width,
    )
    try:
        svg = _structure_svg(plotted, options, tag_colors)
        show_image_full_width(
            svg, caption="Fragment structure key — separate from the attribution plot"
        )
        st.download_button(
            "Download structure figure (SVG)",
            data=svg.encode("utf-8"),
            file_name="sme_fragment_structures.svg",
            mime="image/svg+xml",
            key=f"{prefix}_svg",
        )
    except (ValueError, RuntimeError) as exc:
        st.error(f"Could not draw the fragment structures: {exc}")
        return
    st.caption(
        "SVG retains vector bonds and atom labels at any size. PNG is redrawn at the chosen DPI. Physical panel size excludes the F tag space below each structure."
    )
    if st.checkbox("Prepare publication PNG", key=f"{prefix}_prepare_png"):
        dpi = st.selectbox("PNG resolution (DPI)", [300, 600, 1200], index=1, key=f"{prefix}_dpi")
        try:
            png = _structure_png(plotted, options, dpi, tag_colors)
            st.download_button(
                "Download structure figure (PNG)",
                data=png,
                file_name="sme_fragment_structures.png",
                mime="image/png",
                key=f"{prefix}_png",
            )
        except (ValueError, RuntimeError) as exc:
            st.warning(f"Could not prepare the PNG: {exc}")
