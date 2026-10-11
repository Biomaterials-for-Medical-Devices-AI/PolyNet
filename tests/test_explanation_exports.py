"""Exports preserve vector artists and requested raster detail for every explanation style."""

from io import BytesIO
import xml.etree.ElementTree as ET

from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from polynet.app.components.figure_export import figure_bytes
from polynet.explainability.shap_explain import plot_global_shap, plot_local_shap
from polynet.explainability.visualization import (
    plot_attribution_bar,
    plot_attribution_distribution,
    plot_attribution_strip,
    plot_mols_with_weights,
)

SVG = "{http://www.w3.org/2000/svg}"


@pytest.mark.parametrize(
    "style",
    [
        "ridge",
        "bar",
        "strip",
        "beeswarm",
        "shap-bar",
        "violin",
        "waterfall",
        "force",
        "local-bar",
        "heatmap",
    ],
)
def test_every_explanation_style_exports_vector_and_lossless_formats(style):
    rng = np.random.default_rng(7)
    values = rng.normal(size=(30, 3))
    names = ["CO", "CCN", "C=O"]
    if style in ("ridge", "bar", "strip"):
        plotter = {
            "ridge": plot_attribution_distribution,
            "bar": plot_attribution_bar,
            "strip": plot_attribution_strip,
        }[style]
        fig = plotter(dict(zip(names, values.T)), observation_count_position="right")
    elif style in ("beeswarm", "shap-bar", "violin"):
        fig = plot_global_shap(
            values,
            rng.normal(size=values.shape),
            names,
            plot_type="bar" if style == "shap-bar" else style,
        )
    elif style == "heatmap":
        fig = plot_mols_with_weights(["CCO"], [[-0.3, 0.1, 0.6]], drawing_dpi=150)
    else:
        fig = plot_local_shap(
            values[0],
            names,
            base_value=1.2,
            data=np.ones(3),
            plot_type="bar" if style == "local-bar" else style,
        )
    original_dpi = fig.dpi
    original_size = fig.get_size_inches().copy()
    artists = list(fig.findobj())
    rasterized = {artist: artist.get_rasterized() for artist in artists}
    root = ET.fromstring(figure_bytes(fig, "svg", 150))
    assert root.findall(f".//{SVG}path")
    # Force gradients and molecule heatmaps are embedded images;
    # the figure's lines, text, distributions and observation markers are vector.
    assert bool(root.findall(f".//{SVG}image")) == (style in ("force", "heatmap"))
    assert figure_bytes(fig, "pdf", 150).startswith(b"%PDF-")
    with Image.open(BytesIO(figure_bytes(fig, "png", 150))) as image:
        assert image.info["dpi"] == pytest.approx((150, 150), abs=0.02)
        assert np.asarray(image).std() > 0
    assert fig.dpi == original_dpi
    np.testing.assert_equal(fig.get_size_inches(), original_size)
    assert all(artist in fig.findobj() for artist in artists)
    assert all(artist.get_rasterized() == value for artist, value in rasterized.items())
    plt.close("all")


def test_png_dpi_redraws_without_changing_figure_data():
    fig, ax = plt.subplots(figsize=(2, 2))
    line = ax.plot([1, 2, 3], [4, 5, 6])[0]
    sizes = []
    for dpi in [150, 300]:
        with Image.open(BytesIO(figure_bytes(fig, "png", dpi))) as image:
            sizes.append(image.size)
            assert image.info["dpi"] == pytest.approx((dpi, dpi), abs=0.02)
    # Font hinting makes tight bounding boxes differ by a few pixels across DPI.
    assert sizes[1] == pytest.approx(np.array(sizes[0]) * 2, abs=5)
    np.testing.assert_equal(line.get_ydata(), [4, 5, 6])
    plt.close(fig)


def test_molecule_heatmap_resolution_is_drawn_rather_than_upscaled():
    sizes = []
    for dpi in [150, 300]:
        fig = plot_mols_with_weights(["CCO"], [[-0.3, 0.1, 0.6]], drawing_dpi=dpi)
        image = fig.axes[0].images[0].get_array()
        sizes.append(image.shape[:2])
        assert image.std() > 0
        plt.close(fig)
    assert sizes == [(450, 450), (900, 900)]


@pytest.mark.parametrize(
    "file_format,dpi", [("jpg", 300), ("png", 0), ("svg", True), ("pdf", 1201)]
)
def test_invalid_export_settings_are_reported(file_format, dpi):
    fig = plt.figure()
    with pytest.raises(ValueError):
        figure_bytes(fig, file_format, dpi)
    plt.close(fig)


@pytest.mark.parametrize("style", ["parity", "confusion", "pvalue", "box"])
def test_results_plots_export_in_each_publication_format(style):
    from polynet.visualization.plots import (
        plot_bootstrap_boxplots,
        plot_confusion_matrix,
        plot_parity,
        plot_pvalue_matrix,
    )

    if style == "parity":
        fig = plot_parity([1, 2, 3], [1.1, 1.9, 3.2], width=4, height=4)
    elif style == "confusion":
        fig = plot_confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1], width=4, height=4)
    elif style == "pvalue":
        fig = plot_pvalue_matrix(
            np.array([[1, 0.03], [0.03, 1]]), ["GCN", "GAT"], width=4, height=4
        )
    else:
        fig = plot_bootstrap_boxplots({"GCN": [1, 2, 3], "GAT": [2, 3, 4]}, width=4, height=4)
    svg = ET.fromstring(figure_bytes(fig, "svg", 150))
    assert svg.findall(f".//{SVG}path")
    assert figure_bytes(fig, "pdf", 150).startswith(b"%PDF-")
    with Image.open(BytesIO(figure_bytes(fig, "png", 150))) as image:
        assert image.info["dpi"] == pytest.approx((150, 150), abs=0.02)
    plt.close(fig)


def _comparison_app():
    from types import SimpleNamespace

    import matplotlib.pyplot as plt
    import pandas as pd

    from polynet.app.components.figure_export import display_exportable_figure
    from polynet.app.components.forms.analyse_results import (
        compare_metrics_form,
        compare_predictions_form,
    )
    from polynet.config.constants import DataSet

    data = SimpleNamespace(problem_type="regression", target_variable_name="Tg")
    predictions = pd.DataFrame(
        {
            "True Tg": [1, 2, 3, 4],
            "GCN Predicted Tg": [1.1, 1.8, 3.4, 3.9],
            "GAT Predicted Tg": [1.3, 2.1, 3.1, 4.1],
            "Set": [DataSet.Test] * 4,
        }
    )
    metrics = {
        str(i): {
            name: {DataSet.Test: {"MAE": value}}
            for name, value in [("GCN", 0.1 + i * 0.01), ("GAT", 0.2 + i * 0.02)]
        }
        for i in range(1, 4)
    }
    first = compare_predictions_form(predictions, "Tg", data)
    second = compare_metrics_form(metrics, data, {DataSet.Training: 8, DataSet.Test: 2})
    for figure, label in [(first, "predictions"), (second, "metrics")]:
        if figure is not None:
            display_exportable_figure(
                figure,
                export_options={"dpi": int(figure.dpi), "prefix": "comparisons"},
                plot_id=(label,),
            )
            plt.close(figure)


def test_prediction_and_metric_matrices_have_independent_controls_and_exports():
    app = AppTest.from_function(_comparison_app, default_timeout=30).run()
    app.multiselect[0].set_value(["GCN", "GAT"]).run()
    app.multiselect(key="comparemodelmetrics").set_value(["GCN", "GAT"]).run()
    assert not app.exception and not app.error
    assert len(app.get("download_button")) == 2
    masks = [item for item in app.checkbox if item.label == "Mask upper triangle"]
    assert len(masks) == 2 and masks[0].key != masks[1].key
    masks[0].check().run()
    assert not app.exception and not app.error
    assert [item.value for item in app.checkbox if item.label == "Mask upper triangle"] == [
        True,
        False,
    ]


def test_saved_training_plots_download_original_bytes(tmp_path, monkeypatch):
    from polynet.app.components import plots

    original = BytesIO()
    Image.new("RGB", (200, 100), "white").save(original, "PNG", dpi=(300, 300))
    expected = original.getvalue()
    for name in ["GCN_1_parity_plot.png", "GAT_2_confusion_matrix.png"]:
        (tmp_path / name).write_bytes(expected)
    downloaded = []
    monkeypatch.setattr(plots.st, "image", lambda *args, **kwargs: None)
    monkeypatch.setattr(plots.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        plots.st, "download_button", lambda *args, **kwargs: downloaded.append(kwargs)
    )
    plots.display_plots(tmp_path, export_plots=True)
    assert {row["file_name"] for row in downloaded} == {
        "GCN_1_parity_plot.png",
        "GAT_2_confusion_matrix.png",
    }
    assert all(row["data"] == expected and row["mime"] == "image/png" for row in downloaded)


def test_oversized_raster_export_keeps_vector_download_available():
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.plot([1, 2], [3, 4])
    with pytest.raises(ValueError, match="40 million"):
        figure_bytes(fig, "png", 1200)
    assert figure_bytes(fig, "svg", 1200).startswith(b"<?xml")
    plt.close(fig)


def _export_app():
    from types import SimpleNamespace

    import matplotlib.pyplot as plt
    import pandas as pd
    import streamlit as st

    from polynet.app.services.explain_model import display_gnn_global, display_gnn_local
    from polynet.app.services.explain_tml import display_tml_global, display_tml_local

    dpi = int(st.number_input("Export DPI", value=150, min_value=72, max_value=1200))
    options = {"dpi": dpi, "prefix": "test-exports"}
    figures = []

    def result(**kwargs):
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot([0, 1], [0, 1])
        figures.append(fig)
        return SimpleNamespace(figure=fig, mol_figure=fig, warning=None, **kwargs)

    global_result = result(
        n_mols=2,
        n_models=2,
        n_frags_total=2,
        n_shown=2,
        target_class=None,
        normalisation_type="none",
    )
    display_gnn_global(global_result, export_options=options)
    display_tml_global({"rdkit": global_result, "morgan": global_result}, export_options=options)
    local = [
        result(
            mol_idx="1",
            sample_idx="1",
            model_label=name,
            info_msg="Saved scores",
            true_label="1",
            predicted_label="1",
            attribution_df=pd.DataFrame({"score": [1]}),
        )
        for name in ["GCN — bootstrap 1", "GAT — bootstrap 2"]
    ]
    display_gnn_local(local, export_options=options)
    display_tml_local({"rdkit": local, "morgan": local}, export_options=options)
    plt.close("all")


def test_each_population_descriptor_molecule_and_model_has_an_independent_export_button(
    monkeypatch,
):
    import streamlit as st

    from polynet.app.components import figure_export

    def capture(figure, file_format, dpi):
        st.session_state["last_export"] = (file_format, dpi)
        return figure_bytes(figure, file_format, dpi)

    monkeypatch.setattr(figure_export, "figure_bytes", capture)
    app = AppTest.from_function(_export_app, default_timeout=30).run()
    assert not app.exception and not app.error
    buttons = app.get("download_button")
    assert len(buttons) == 9 and all(button.label == "Export plot" for button in buttons)
    keys = [widget.key for widget in app.selectbox]
    assert len(set(keys)) == 9
    app.selectbox[8].set_value("png").run()
    assert app.selectbox[0].value == "svg"
    assert app.session_state["last_export"] == ("png", 150)
    app.number_input[0].set_value(300).run()
    assert app.session_state["last_export"] == ("png", 300)
    app.selectbox[8].set_value("pdf").run()
    assert app.session_state["last_export"] == ("pdf", 300)
    assert not app.exception and not app.error
