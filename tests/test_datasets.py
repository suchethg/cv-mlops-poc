"""Dataset IDs must be content hashes; splits must never leak patients."""
import pytest

from surgseg.datasets import check_no_leakage, dataset_id, golden_test_cases

ENTRIES = [{"path": "train/v1/c1/frames/f0.png", "sha256": "a" * 64},
           {"path": "test/v2/c2/frames/f0.png", "sha256": "b" * 64}]


def test_dataset_id_is_deterministic():
    assert dataset_id("d", ENTRIES, ["v2"], "labels-v1") == dataset_id("d", ENTRIES, ["v2"], "labels-v1")


def test_dataset_id_changes_if_any_file_changes():
    changed = [ENTRIES[0], dict(ENTRIES[1], sha256="c" * 64)]
    assert dataset_id("d", ENTRIES, ["v2"], "labels-v1") != dataset_id("d", changed, ["v2"], "labels-v1")


def test_dataset_id_changes_if_labels_or_split_change():
    base = dataset_id("d", ENTRIES, ["v2"], "labels-v1")
    assert base != dataset_id("d", ENTRIES, ["v2"], "labels-v2")
    assert base != dataset_id("d", ENTRIES, ["v1"], "labels-v1")


def test_leakage_check_rejects_a_patient_in_both_splits():
    clips = [{"case_id": "v1", "patient_pseudonym": "pt-1"},
             {"case_id": "v2", "patient_pseudonym": "pt-1"}]
    with pytest.raises(ValueError, match="leakage"):
        check_no_leakage(clips, test_cases=["v2"])


def test_golden_test_set_never_changes(s3):
    first, created = golden_test_cases(s3, ["v1", "v2", "v3", "v4", "v5"], 0.2, "run/1")
    again, created_again = golden_test_cases(s3, ["v1", "v2", "v3", "v4", "v5", "v6"], 0.5, "run/2")
    assert created and not created_again
    assert first == again and len(first) == 1