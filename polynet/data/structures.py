"""
polynet.data.structures
=======================
Detection, validation and canonicalisation of polymer structure strings.

Every entry point (GUI, CLI, Python API) prepares the structure columns of a
dataset with the same function, ``prepare_structures``, before any
featurisation:

1. **detect** the string representation (SMILES or PSMILES);
2. **validate** every structure, raising a ``ValueError`` that lists example
   invalid entries per column (missing structures count as invalid and are
   shown as ``<missing>``);
3. **canonicalise** the structures (optional, on by default), so the same
   molecule is always written the same way.

The per-representation behaviour lives in two lookup tables
(``_VALIDATORS``, ``_CANONICALISERS``); supporting another representation
means adding one entry to each.

Public API
----------
::

    from polynet.data.structures import prepare_structures

    df, representation = prepare_structures(df, smiles_cols=["smiles"])
"""

from __future__ import annotations

from collections.abc import Callable
import logging

import pandas as pd

from polynet.config.enums import StringRepresentation
from polynet.utils.chem_utils import (
    canonicalise_psmiles,
    canonicalise_smiles,
    check_smiles,
    identify_psmiles,
)

logger = logging.getLogger(__name__)

# RDKit parses both SMILES and PSMILES (the ``*`` attachment points are dummy atoms).
_VALIDATORS: dict[StringRepresentation, Callable[[str], bool]] = {
    StringRepresentation.SMILES: check_smiles,
    StringRepresentation.PSMILES: check_smiles,
}
_CANONICALISERS: dict[StringRepresentation, Callable[[str], str | None]] = {
    StringRepresentation.SMILES: canonicalise_smiles,
    StringRepresentation.PSMILES: canonicalise_psmiles,
}

# Number of invalid examples shown per column in error messages.
_N_EXAMPLES = 5


def detect_string_representation(df: pd.DataFrame, smiles_cols: list[str]) -> StringRepresentation:
    """
    Return the representation of the structure columns.

    PSMILES if every value of every column has at least two ``*`` attachment
    points, otherwise SMILES. Missing / non-string values count as not PSMILES
    (they are reported by ``find_invalid_structures``).
    """
    for col in smiles_cols:
        if not df[col].apply(lambda s: isinstance(s, str) and identify_psmiles(s)).all():
            return StringRepresentation.SMILES
    return StringRepresentation.PSMILES


def _is_missing(value) -> bool:
    return not isinstance(value, str) or value.strip() == ""


def find_invalid_structures(
    df: pd.DataFrame, smiles_cols: list[str], representation: StringRepresentation
) -> dict[str, list[str]]:
    """
    Return the structures that cannot be parsed, per column.

    Parameters
    ----------
    df:
        Dataset.
    smiles_cols:
        Structure columns to check.
    representation:
        Representation of those columns (selects the parser).

    Returns
    -------
    dict[str, list[str]]
        ``{column: [invalid values]}``; columns with no invalid value are
        omitted, so an empty dict means everything is valid. Missing values
        (NaN / empty) are invalid and reported as ``"<missing>"``.
    """
    is_valid = _VALIDATORS[StringRepresentation(representation)]
    invalid: dict[str, list[str]] = {}
    for col in smiles_cols:
        bad = [
            "<missing>" if _is_missing(s) else str(s)
            for s in df[col]
            if _is_missing(s) or not is_valid(s)
        ]
        if bad:
            invalid[col] = bad
    return invalid


def invalid_structures_message(invalid: dict[str, list[str]], representation: str) -> str:
    """Human-readable error listing example invalid structures per column."""
    lines = [
        f"  - column '{col}': {len(values)} invalid, e.g. {', '.join(values[:_N_EXAMPLES])}"
        for col, values in invalid.items()
    ]
    return f"Invalid {representation} structures found:\n" + "\n".join(lines)


def canonicalise_structures(
    df: pd.DataFrame, smiles_cols: list[str], representation: StringRepresentation
) -> pd.DataFrame:
    """
    Return a copy of ``df`` with the structure columns canonicalised.

    Raises
    ------
    ValueError
        If a structure could not be canonicalised (listing examples).
    """
    canonicalise = _CANONICALISERS[StringRepresentation(representation)]
    out = df.copy()
    failed: dict[str, list[str]] = {}
    for col in smiles_cols:
        canonical = out[col].apply(canonicalise)
        bad = out.loc[canonical.isna(), col].astype(str).tolist()
        if bad:
            failed[col] = bad
        out[col] = canonical
    if failed:
        raise ValueError(
            "Could not canonicalise some structures:\n"
            + "\n".join(
                f"  - column '{col}': {', '.join(v[:_N_EXAMPLES])}" for col, v in failed.items()
            )
        )
    return out


def prepare_structures(
    df: pd.DataFrame,
    smiles_cols: list[str],
    representation: StringRepresentation | str | None = None,
    canonicalise: bool = True,
) -> tuple[pd.DataFrame, StringRepresentation]:
    """
    Detect, validate and (optionally) canonicalise the structure columns.

    Parameters
    ----------
    df:
        Dataset. Not modified.
    smiles_cols:
        Structure columns.
    representation:
        Representation declared by the user (e.g. ``data.string_representation``).
        If given and different from the detected one, a warning is logged and
        the declared representation is used. ``None`` uses the detected one.
    canonicalise:
        Whether to canonicalise the structures (``data.canonicalise_smiles``).

    Returns
    -------
    tuple[pd.DataFrame, StringRepresentation]
        ``(prepared_df, representation)``.

    Raises
    ------
    ValueError
        If any structure is invalid or missing (the message lists examples
        per column) or cannot be canonicalised.
    """
    detected = detect_string_representation(df, smiles_cols)
    if representation is None:
        representation = detected
    else:
        representation = StringRepresentation(representation)
        if representation != detected:
            logger.warning(
                "The structure columns look like %s, but the config declares %s; using %s.",
                detected.value,
                representation.value,
                representation.value,
            )

    invalid = find_invalid_structures(df, smiles_cols, representation)
    if invalid:
        raise ValueError(invalid_structures_message(invalid, representation.value))

    if canonicalise:
        df = canonicalise_structures(df, smiles_cols, representation)
        logger.info(f"Canonicalised {representation.value} in columns {list(smiles_cols)}.")
    return df, representation
