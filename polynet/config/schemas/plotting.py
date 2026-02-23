"""
polynet.config.schemas.plotting
================================
Pydantic schema for plot styling and output options.
"""

from pydantic import Field, field_validator

from polynet.config.schemas.base import PolynetBaseModel


class PlottingConfig(PolynetBaseModel):
    """
    Configuration for all plot styling and output settings.

    These options are applied globally across every plot produced by the
    ``visualization`` module. Individual plot functions may override specific
    values where needed.
    """

    # --- Font and text ---
    axis_font_size: int = Field(default=12, ge=4, le=72, description="Font size for axis labels.")
    axis_tick_size: int = Field(
        default=10, ge=4, le=72, description="Font size for axis tick labels."
    )
    title_font_size: int = Field(default=14, ge=4, le=72, description="Font size for plot titles.")
    font_family: str = Field(default="sans-serif", description="Matplotlib font family.")

    # --- Layout ---
    height: float = Field(default=6.0, gt=0, description="Figure height in inches.")
    width: float = Field(default=8.0, gt=0, description="Figure width in inches.")
    angle_rotate_xaxis_labels: int = Field(
        default=0, ge=-180, le=180, description="Rotation angle for x-axis tick labels."
    )
    angle_rotate_yaxis_labels: int = Field(
        default=0, ge=-180, le=180, description="Rotation angle for y-axis tick labels."
    )

    # --- Style ---
    colour_scheme: str = Field(
        default="tab10", description="Matplotlib colour cycle / palette name."
    )
    colour_map: str = Field(
        default="viridis", description="Matplotlib colormap for continuous data."
    )

    # --- Output ---
    dpi: int = Field(
        default=150, ge=72, le=1200, description="Resolution in dots per inch for saved figures."
    )
    save_plots: bool = Field(default=False, description="Whether to save plots to disk.")

    @field_validator("font_family")
    @classmethod
    def valid_font_family(cls, v: str) -> str:
        allowed = {"serif", "sans-serif", "monospace", "cursive", "fantasy"}
        if v not in allowed:
            raise ValueError(f"font_family must be one of {allowed}, got '{v}'.")
        return v
