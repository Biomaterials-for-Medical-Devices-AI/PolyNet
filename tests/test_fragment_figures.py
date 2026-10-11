"""SME display labels, stable fragment keys and publication figure exports."""

from io import BytesIO
import json
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from streamlit.testing.v1 import AppTest

from polynet.app.components.forms.fragment_plots import saved_fragment_table
from polynet.app.services.explanation_results import capture_analysis, load_analysis
from polynet.config.enums import AttributionPlotType, FragmentationMethod, ProblemType
from polynet.config.paths import explanation_json_file_path
from polynet.explainability.fragment_figures import (
    FragmentGridOptions,
    fragment_label_table,
    fragment_structure_png,
    fragment_structure_svg,
)
from polynet.explainability.masking import MASKING_CACHE_VERSION, MASKING_CACHE_VERSION_KEY
from polynet.explainability.visualization import (
    ordered_attribution_fragments,
    plot_attribution_bar,
    plot_attribution_distribution,
    plot_attribution_strip,
)

SVG = "{http://www.w3.org/2000/svg}"


def test_fragment_tags_and_smarts_are_stable_and_preserve_chemistry():
    fragments = ["CO", "c1ccccc1", "C[C@H](O)F", "[NH4+]", "C(=O)O"]
    table = fragment_label_table(fragments)
    pd.testing.assert_frame_equal(table, fragment_label_table(list(reversed(fragments)) + ["CO"]))
    assert table.Tag.tolist() == ["F1", "F2", "F3", "F4", "F5"]
    for row in table.itertuples():
        mol = Chem.MolFromSmiles(row.Fragment)
        query = Chem.MolFromSmarts(row.SMARTS)
        assert query is not None and mol.HasSubstructMatch(query, useChirality=True)
    assert "@" in table.loc[table.Fragment == "C[C@H](O)F", "SMARTS"].iloc[0]
    assert "+" in table.loc[table.Fragment == "[NH4+]", "SMARTS"].iloc[0]
    assert list(fragment_label_table([]).columns) == ["Tag", "SMARTS", "Fragment"]


@pytest.mark.parametrize(
    "plotter", [plot_attribution_distribution, plot_attribution_bar, plot_attribution_strip]
)
def test_display_labels_preserve_scores_order_and_top_bottom_selection(plotter):
    rng = np.random.default_rng(13)
    scores = {
        fragment: list(rng.normal(mean, 0.1, 25))
        for fragment, mean in [("CO", -2), ("CCN", 0), ("c1ccccc1", 3)]
    }
    before = json.dumps(scores)
    table = fragment_label_table(scores)
    labels = dict(zip(table.Fragment, table.Tag))
    fig = plotter(scores, top_n=1, fragment_labels=labels)
    assert ordered_attribution_fragments(scores, 1) == ["c1ccccc1", "CO"]
    if plotter == plot_attribution_distribution:
        displayed = [text.get_text() for ax in fig.axes for text in ax.texts]
    else:
        displayed = [text.get_text() for text in fig.axes[0].get_yticklabels()]
    assert set(displayed) == {labels["CO"], labels["c1ccccc1"]}
    assert labels["CCN"] not in displayed
    assert json.dumps(scores) == before
    # Bar values and strip means retain their original numeric values.
    if plotter == plot_attribution_bar:
        assert [bar.get_width() for bar in fig.axes[0].patches] == pytest.approx(
            [np.mean(scores["CO"]), np.mean(scores["c1ccccc1"])]
        )
    plt.close("all")


def test_structure_svg_is_vector_physical_size_and_keeps_table_order():
    table = fragment_label_table(["CO", "c1ccccc1", "C[C@H](O)F"])
    table = table.iloc[[2, 0, 1]]
    root = ET.fromstring(fragment_structure_svg(table, FragmentGridOptions(columns=2)))
    assert root.attrib["width"].endswith("mm") and root.attrib["height"].endswith("mm")
    cells = root.findall(f"{SVG}svg")
    assert [cell.attrib["data-fragment-tag"] for cell in cells] == table.Tag.tolist()
    assert cells[0].attrib["y"] == cells[1].attrib["y"]
    assert float(cells[2].attrib["y"]) > float(cells[0].attrib["y"])
    assert len(root.findall(f".//{SVG}path")) > 10
    assert not root.findall(f".//{SVG}image")  # No embedded raster preview.
    assert [cell.find(f"{SVG}title").text for cell in cells] == [
        f"{row.Tag}: {row.Fragment}" for row in table.itertuples()
    ]
    changed = ET.fromstring(
        fragment_structure_svg(
            table, FragmentGridOptions(columns=3, mol_width_mm=50, horizontal_gap_mm=8)
        )
    )
    assert changed.attrib["viewBox"] != root.attrib["viewBox"]


def test_png_is_redrawn_at_requested_dpi_with_matching_physical_dimensions():
    table = fragment_label_table(["CO", "c1ccccc1"])
    options = FragmentGridOptions(columns=2, mol_width_mm=30, mol_height_mm=20)
    sizes = []
    for dpi in [150, 300]:
        with Image.open(BytesIO(fragment_structure_png(table, options, dpi))) as image:
            assert image.info["dpi"] == pytest.approx((dpi, dpi), abs=0.02)
            assert image.getextrema() != ((255, 255),) * 3
            sizes.append(image.size)
    assert sizes[1] == pytest.approx(np.array(sizes[0]) * 2, abs=3)
    svg = ET.fromstring(fragment_structure_svg(table, options))
    width_mm = float(svg.attrib["width"][:-2])
    height_mm = float(svg.attrib["height"][:-2])
    assert np.array(sizes[1]) * 25.4 / 300 == pytest.approx([width_mm, height_mm], abs=0.2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"columns": 0},
        {"mol_width_mm": -1},
        {"atom_font_pt": 0},
        {"horizontal_gap_mm": -1},
        {"vertical_gap_mm": float("nan")},
    ],
)
def test_invalid_physical_layout_is_rejected(kwargs):
    with pytest.raises(ValueError):
        FragmentGridOptions(**kwargs)


def test_invalid_structures_empty_figures_and_oversized_png_are_reported():
    with pytest.raises(ValueError, match="Cannot depict"):
        fragment_label_table(["definitely not a molecule"])
    with pytest.raises(ValueError, match="at least one fragment"):
        fragment_structure_svg(fragment_label_table([]))
    table = fragment_label_table(["CO"])
    with pytest.raises(ValueError, match="40 million pixels"):
        fragment_structure_png(
            table, FragmentGridOptions(mol_width_mm=300, mol_height_mm=300), 1200
        )
    with pytest.raises(ValueError, match="DPI"):
        fragment_structure_png(table, dpi=0)
    assert "<svg" in fragment_structure_svg(fragment_label_table(["[#6;R]-[#8]"]))


def _plot_app(root):
    from pathlib import Path
    from types import SimpleNamespace

    from polynet.app.components.forms.explanation_plots import explanation_plots_form

    explanation_plots_form(Path(root), SimpleNamespace())


def _capture_plot(result):
    import streamlit as st

    st.session_state["scores"] = result.attribution_dict
    st.session_state["labels"] = [
        text.get_text() for text in result.figure.axes[0].get_yticklabels()
    ]


def _widget(widgets, label):
    return next(widget for widget in widgets if widget.label == label)


def test_page6_f_tags_exports_and_layout_never_compute_or_rewrite_scores(tmp_path, monkeypatch):
    from polynet.app.components.forms import explanation_plots, fragment_plots
    from polynet.explainability import explain

    raw = {
        model: {
            "1": {
                "1": {
                    "chemistry_masking": {
                        "regression": {
                            "CCOCN": {
                                "brics": {
                                    "CO": [sign * 1.0],
                                    "CN": [sign * 2.0],
                                    "CCO": [sign * 3.0],
                                }
                            }
                        }
                    }
                }
            }
        }
        for model, sign in [("GCN", 1), ("GAT", -1)]
    }
    raw[MASKING_CACHE_VERSION_KEY] = MASKING_CACHE_VERSION
    cache = explanation_json_file_path(tmp_path)
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps(raw))
    path = capture_analysis(
        "GNN",
        "global",
        dict(
            models={"GCN_1": None, "GAT_1": None},
            experiment_path=tmp_path,
            dataset=[SimpleNamespace(idx=1, mols=["CCOCN"])],
            explain_mols=[1],
            problem_type=ProblemType.Regression,
            fragmentation_approach=FragmentationMethod.BRICS,
        ),
    )
    snapshot_bytes, cache_bytes = path.read_bytes(), cache.read_bytes()
    record = load_analysis(path)
    table = saved_fragment_table(record)
    before = table.copy(deep=True)
    table.iloc[0, 0] = "changed"
    pd.testing.assert_frame_equal(saved_fragment_table(record), before)

    def forbidden(*args, **kwargs):
        pytest.fail("Presentation attempted model inference")

    monkeypatch.setattr(explain, "compute_and_cache_masking", forbidden)
    monkeypatch.setattr(explanation_plots, "display_gnn_global", _capture_plot)
    exported = {}

    def capture_svg(table, options):
        exported["table"] = table.copy()
        exported["options"] = options
        return fragment_structure_svg(table, options)

    monkeypatch.setattr(fragment_plots, "_structure_svg", capture_svg)
    download_button = fragment_plots.st.download_button

    def capture_download(*args, **kwargs):
        exported.setdefault("downloads", {})[kwargs["file_name"]] = kwargs["data"]
        return download_button(*args, **kwargs)

    monkeypatch.setattr(fragment_plots.st, "download_button", capture_download)
    app = AppTest.from_function(_plot_app, kwargs={"root": str(tmp_path)}, default_timeout=30).run()
    app.checkbox(key="analyse_explanations_enabled").check().run()
    _widget(app.radio, "Plot type").set_value(AttributionPlotType.Bar).run()
    _widget(app.multiselect, "GNN architectures to include").set_value(["GCN"]).run()
    scores = app.session_state["scores"]
    smarts_labels = app.session_state["labels"]
    assert set(smarts_labels) == set(before.SMARTS)
    _widget(app.radio, "Fragment labels").set_value("F tags").run()
    assert app.session_state["scores"] == scores
    assert set(app.session_state["labels"]) == set(before.Tag)
    _widget(app.checkbox, "Create separate fragment structure figure").check().run()
    assert exported["table"].Fragment.tolist() == ["CCO", "CN", "CO"]
    _widget(app.number_input, "Fragments to show (top N + bottom N)").set_value(1).run()
    assert exported["table"].Fragment.tolist() == ["CCO", "CO"]
    assert exported["table"].Tag.tolist() == ["F1", "F3"]
    _widget(app.multiselect, "GNN architectures to include").set_value(["GAT"]).run()
    assert exported["table"].Fragment.tolist() == ["CO", "CCO"]
    assert exported["table"].Tag.tolist() == ["F3", "F1"]
    csv = pd.read_csv(BytesIO(exported["downloads"]["sme_fragment_mapping.csv"]))
    pd.testing.assert_frame_equal(csv, exported["table"].reset_index(drop=True))
    _widget(app.number_input, "Structures per row").set_value(1).run()
    _widget(app.number_input, "Molecule panel width (mm)").set_value(50.0).run()
    _widget(app.number_input, "Horizontal spacing (mm)").set_value(8.0).run()
    assert exported["options"].columns == 1 and exported["options"].mol_width_mm == 50
    _widget(app.checkbox, "Prepare publication PNG").check().run()
    assert not app.exception and not app.error
    assert len(app.get("download_button")) == 3  # CSV, vector SVG, high-resolution PNG.
    cells = ET.fromstring(exported["downloads"]["sme_fragment_structures.svg"]).findall(f"{SVG}svg")
    assert [cell.attrib["data-fragment-tag"] for cell in cells] == ["F3", "F1"]
    with Image.open(BytesIO(exported["downloads"]["sme_fragment_structures.png"])) as image:
        assert image.info["dpi"] == pytest.approx((600, 600), abs=0.02)
    assert path.read_bytes() == snapshot_bytes and cache.read_bytes() == cache_bytes
