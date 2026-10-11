"""SME display labels, stable fragment keys and publication figure exports."""

from io import BytesIO
import json
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from PIL import Image
import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import to_hex, to_rgb
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
    attribution_fragment_colors,
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


@pytest.mark.parametrize(
    "plotter,style",
    [
        (plot_attribution_distribution, AttributionPlotType.Ridge),
        (plot_attribution_bar, AttributionPlotType.Bar),
        (plot_attribution_strip, AttributionPlotType.Strip),
    ],
)
@pytest.mark.parametrize("match_distribution", [False, True])
def test_tag_colours_match_distribution_or_custom_without_recolouring_data(
    plotter, style, match_distribution
):
    scores = {"CO": [-3.0, -2.0, -1.0], "CCN": [-0.2, 0.0, 0.2], "C=O": [1.0, 3.0, 5.0]}
    labels = {"CO": "F1", "CCN": "F2", "C=O": "F3"}
    before = json.dumps(scores)
    negative, positive = "#0088ff", "#ff3300"
    matched = attribution_fragment_colors(scores, style, 1, negative, positive)
    colors = matched if match_distribution else dict.fromkeys(scores, "#228833")
    fig = plotter(
        scores,
        top_n=1,
        neg_color=negative,
        pos_color=positive,
        fragment_labels=labels,
        fragment_label_colors=colors,
    )
    texts = (
        [text for ax in fig.axes for text in ax.texts]
        if style == AttributionPlotType.Ridge
        else fig.axes[0].get_yticklabels()
    )
    assert {text.get_text() for text in texts} == {"F1", "F3"}
    for text in texts:
        fragment = next(frag for frag, tag in labels.items() if tag == text.get_text())
        assert to_rgb(text.get_color()) == pytest.approx(to_rgb(colors[fragment]))
    # A custom tag colour must not affect the existing data palettes.
    if style == AttributionPlotType.Bar:
        for bar, fragment in zip(fig.axes[0].patches, ["CO", "C=O"]):
            assert bar.get_facecolor()[:3] == pytest.approx(matched[fragment])
    elif style == AttributionPlotType.Strip:
        for collection, fragment in zip(fig.axes[0].collections[1::2], ["CO", "C=O"]):
            assert collection.get_facecolors()[0][:3] == pytest.approx(matched[fragment])
    else:
        for ax, fragment in zip(fig.axes, ["C=O", "CO"]):
            assert ax.collections[0].get_facecolors()[0][:3] == pytest.approx(matched[fragment])
    assert json.dumps(scores) == before
    plt.close("all")


@pytest.mark.parametrize(
    "plotter", [plot_attribution_distribution, plot_attribution_bar, plot_attribution_strip]
)
@pytest.mark.parametrize("position", [None, "tag", "right"])
def test_observation_counts_follow_fragment_occurrences_and_top_bottom_selection(plotter, position):
    scores = {"CO": [-3.0, -2.0, -1.0], "CCN": [-0.2, 0.0], "C=O": [1.0, 3.0, 5.0, 7.0]}
    before = json.dumps(scores)
    labels = {"CO": "F1", "CCN": "F2", "C=O": "F3"}
    fig = plotter(scores, top_n=1, fragment_labels=labels, observation_count_position=position)
    texts = [text.get_text() for ax in fig.axes for text in ax.texts]
    tags = (
        [text for text in texts if text.startswith("F")]
        if plotter == plot_attribution_distribution
        else [text.get_text() for text in fig.axes[0].get_yticklabels()]
    )
    assert set(tags) == ({"F1 (n=3)", "F3 (n=4)"} if position == "tag" else {"F1", "F3"})
    assert {text for text in texts if text.startswith("n=")} == (
        {"n=3", "n=4"} if position == "right" else set()
    )
    assert json.dumps(scores) == before
    if plotter == plot_attribution_strip and position == "right":
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        colourbar_left = fig.axes[1].get_window_extent(renderer).x0
        for text in fig.axes[0].texts:
            if text.get_text().startswith("n="):
                assert text.get_window_extent(renderer).x1 < colourbar_left
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


def test_structure_tag_colours_are_exported_without_recolouring_molecules():
    table = fragment_label_table(["CO", "C=O"])
    colors = {"CO": "#ff0000", "C=O": "#0000ff"}
    root = ET.fromstring(fragment_structure_svg(table, tag_colors=colors))
    for cell, row in zip(root.findall(f"{SVG}svg"), table.itertuples()):
        paths = cell.findall(f".//{SVG}path")
        legends = [path for path in paths if "legend" in path.attrib.get("class", "")]
        chemical_paths = [path for path in paths if "legend" not in path.attrib.get("class", "")]
        assert legends
        assert all(
            to_hex(colors[row.Fragment]).upper()
            in path.attrib.get("fill", path.attrib.get("style", "")).upper()
            for path in legends
        )
        assert not any(
            to_hex(colors[row.Fragment]).upper() in path.attrib.get("style", "").upper()
            for path in chemical_paths
        )
    with Image.open(BytesIO(fragment_structure_png(table, dpi=150, tag_colors=colors))) as image:
        pixels = np.asarray(image)
        coloured = (pixels[:, :, 0] != pixels[:, :, 1]) | (pixels[:, :, 1] != pixels[:, :, 2])
        y_positions = np.where(coloured)[0]
        assert len(y_positions) > 100
        assert y_positions.min() > image.height * 0.75


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


def _capture_plot(result, **kwargs):
    import streamlit as st

    st.session_state["scores"] = result.attribution_dict
    st.session_state["labels"] = [
        text.get_text() for text in result.figure.axes[0].get_yticklabels()
    ]
    st.session_state["label_colors"] = {
        text.get_text(): to_hex(text.get_color())
        for text in result.figure.axes[0].get_yticklabels()
    }


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

    def capture_svg(table, options, tag_colors=None):
        exported["table"] = table.copy()
        exported["options"] = options
        exported["tag_colors"] = tag_colors
        return fragment_structure_svg(table, options, tag_colors)

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
    original_labels = app.session_state["labels"]
    assert _widget(app.radio, "Fragment labels").value == "Fragment SMILES"
    assert set(original_labels) == set(before.Fragment)
    assert set(app.session_state["label_colors"].values()) == {"#000000"}
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
    _widget(app.color_picker, "Tag colour").set_value("#228833").run()
    assert set(app.session_state["label_colors"].values()) == {"#228833"}
    assert set(to_hex(color) for color in exported["tag_colors"].values()) == {"#228833"}
    current_scores = app.session_state["scores"]
    _widget(app.radio, "Tag colour mode").set_value("Match distribution").run()
    assert app.session_state["scores"] == current_scores
    assert app.session_state["label_colors"] == {
        row.Tag: to_hex(exported["tag_colors"][row.Fragment])
        for row in exported["table"].itertuples()
    }
    assert len(set(app.session_state["label_colors"].values())) == 2
    _widget(app.radio, "Observation counts").set_value("Next to tag").run()
    assert set(app.session_state["labels"]) == {"F3 (n=1)", "F1 (n=1)"}
    assert exported["table"].Tag.tolist() == ["F3", "F1"]
    _widget(app.radio, "Observation counts").set_value("Right side").run()
    assert set(app.session_state["labels"]) == {"F3", "F1"}
    assert app.session_state["scores"] == current_scores
    _widget(app.radio, "Observation counts").set_value("Hidden").run()
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
    _widget(app.radio, "Fragment labels").set_value("Fragment SMILES").run()
    assert app.session_state["scores"]["CO"] == [-1.0 / 3]
    assert set(app.session_state["labels"]) == {"CO", "CCO"}
    assert not app.get("download_button")
    assert path.read_bytes() == snapshot_bytes and cache.read_bytes() == cache_bytes
