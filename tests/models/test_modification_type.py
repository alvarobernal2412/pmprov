from models.artifacts import ModificationType

def test_all_values_present():
    assert ModificationType.ADDITION.value == "addition"
    assert ModificationType.REMOVAL.value == "removal"
    assert ModificationType.RENAMING.value == "renaming"
    assert ModificationType.CASTING.value == "casting"
    assert ModificationType.NORMALIZATION.value == "normalization"
    assert ModificationType.ENRICHMENT.value == "enrichment"
    assert ModificationType.OBFUSCATION.value == "obfuscation"
    assert ModificationType.RECALCULATION.value == "recalculation"
    assert ModificationType.OTHER.value == "other"

def test_fine_grained_values_removed():
    for old in ("SCHEMA_CHANGE", "ROW_FILTER", "ROW_ADD", "VALUE_CHANGE", "TYPE_CHANGE"):
        assert not hasattr(ModificationType, old), f"{old} should have been removed"
