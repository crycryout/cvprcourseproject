"""Synthetic metadata tests validate scope checks; these are not GPU benchmarks."""
import pytest
from cvpr_project.hardware import require_device_scope


def test_mig_is_not_mistaken_for_a_full_card():
    actual = {"name": "NVIDIA H800 PCIe MIG 2g.20gb", "multiprocessors": 30,
              "total_memory_bytes": 21072183296}
    with pytest.raises(ValueError, match="Full H800 required"):
        require_device_scope(actual)


def test_same_model_name_does_not_allow_a_different_physical_device():
    reference = {"name": "NVIDIA H800 PCIe", "multiprocessors": 114,
                 "total_memory_bytes": 85017493504, "device_uuid_sha256": "reference-device"}
    require_device_scope(dict(reference), reference)
    with pytest.raises(ValueError, match="differs"):
        require_device_scope({**reference, "device_uuid_sha256": "another-device"}, reference)
