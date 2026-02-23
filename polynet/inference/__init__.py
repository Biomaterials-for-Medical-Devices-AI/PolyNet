"""
polynet.inference
=================
Prediction collection and DataFrame assembly for trained polynet models.

Both GNN and TML inference produce the same wide DataFrame format so
that downstream metric calculation and plotting are model-agnostic.

::

    from polynet.inference import get_predictions_df_gnn, get_predictions_df_tml
"""

from polynet.inference.gnn import get_predictions_df_gnn
from polynet.inference.tml import get_predictions_df_tml
from polynet.inference.utils import prepare_probs_df

__all__ = ["get_predictions_df_gnn", "get_predictions_df_tml", "prepare_probs_df"]
