"""
tests/test_fragmentation_methods.py
===================================
Only the implemented fragmentation methods (BRICS, Murcko scaffold) are
accepted; the unimplemented RECAP and functional-group options were removed.
"""

from pydantic import ValidationError
import pytest

from polynet.config.enums import FragmentationMethod
from polynet.config.schemas.explainability import ExplainabilityConfig
from polynet.utils.chem_utils import fragment_and_match


def test_only_implemented_methods_exist():
    assert {m.value for m in FragmentationMethod} == {"brics", "murcko_scaffold"}


@pytest.mark.parametrize("method", list(FragmentationMethod))
def test_every_method_fragments_a_molecule(method):
    frags = fragment_and_match("CC(=O)OCCc1ccccc1", method)
    assert frags and all(isinstance(idx, list) for occ in frags.values() for idx in occ)


@pytest.mark.parametrize("removed", ["recap", "functional_groups"])
def test_removed_methods_are_rejected_at_config_load(removed):
    with pytest.raises(ValidationError, match="fragmentation"):
        ExplainabilityConfig(fragmentation=removed)
