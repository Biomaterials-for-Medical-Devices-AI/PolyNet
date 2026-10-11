"""Lossless and vector exports of the current figure, without inference."""

from __future__ import annotations

import hashlib
from io import BytesIO
import re

import matplotlib as mpl
import streamlit as st

_MIMES = {"svg": "image/svg+xml", "pdf": "application/pdf", "png": "image/png"}


def figure_bytes(figure, file_format: str = "svg", dpi: int = 300) -> bytes:
    """Export the supplied figure without altering its size, artists or data.

    SVG embeds font outlines; PDF embeds TrueType fonts. PNG is lossless at the
    requested DPI. Raster images already inside a figure retain their source detail.
    """
    if file_format not in _MIMES:
        raise ValueError("Choose SVG, PDF or PNG for the plot export.")
    if not isinstance(dpi, int) or isinstance(dpi, bool) or not 50 <= dpi <= 1200:
        raise ValueError("Export DPI must be an integer between 50 and 1200.")
    if file_format == "png":
        width, height = figure.get_size_inches()
        if (width + 0.2) * (height + 0.2) * dpi**2 > 40_000_000:
            raise ValueError("PNG exceeds 40 million pixels. Lower DPI or export SVG/PDF.")
    output = BytesIO()
    # Matplotlib normally rasterises colourbars (and SHAP can rasterise points).
    # Retain those artists as vectors for publication, restoring flags afterwards.
    rasterized = [artist for artist in figure.findobj() if artist.get_rasterized()]
    try:
        if file_format in ("svg", "pdf"):
            for artist in rasterized:
                artist.set_rasterized(False)
        with mpl.rc_context({"svg.fonttype": "path", "pdf.fonttype": 42}):
            figure.savefig(
                output,
                format=file_format,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.1,
                facecolor="white",
            )
    finally:
        for artist in rasterized:
            artist.set_rasterized(True)
    return output.getvalue()


def display_exportable_figure(
    figure, *, container=st, export_options: dict | None = None, plot_id=("plot",)
) -> None:
    """Show a figure with a per-plot export button when page 6 enables exports.

    The preview stays at screen resolution; only downloads use the chosen DPI.
    Other callers omit ``export_options`` and keep their existing preview.
    """
    if export_options is None:
        container.pyplot(figure, use_container_width=True)
        return
    container.pyplot(figure, use_container_width=True, dpi=150)
    identity = hashlib.sha256(repr(plot_id).encode()).hexdigest()[:12]
    key = f"{export_options['prefix']}_{identity}"
    formats = {"svg": "SVG (vector)", "pdf": "PDF (vector)", "png": "PNG (lossless)"}
    file_format = container.selectbox(
        "Export format",
        list(formats),
        format_func=formats.get,
        key=f"{key}_format",
        help="SVG and PDF keep lines and text sharp at any size. Embedded molecule heatmaps retain their drawn resolution; PNG uses the selected export DPI.",
    )
    name = "-".join(str(part) for part in plot_id if part is not None)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-.") or "plot"
    try:
        data = figure_bytes(figure, file_format, export_options["dpi"])
        container.download_button(
            "Export plot",
            data=data,
            file_name=f"plot-{name}.{file_format}",
            mime=_MIMES[file_format],
            key=f"{key}_download",
            on_click="ignore",
        )
    except (ValueError, RuntimeError, OSError) as exc:
        container.warning(f"Unable to export this plot: {exc}")
