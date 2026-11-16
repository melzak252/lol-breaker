"""Focused safety contracts for fresh C0-D reproduction orchestration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.models.c0_distilled_reproduction import _verify_raw, _sha256


def test_execution_gate_rejects_unverified_raw_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema": "c0_raw_causal_anchor_bank_v2", "artifacts": {}})
    )
    with pytest.raises(ValueError, match="unexpected schema|lacks hash"):
        _verify_raw(manifest, {})


def test_raw_bank_gate_checks_all_artifact_hashes_and_shapes(tmp_path: Path) -> None:
    artifacts = {}
    ids = np.asarray([f"r{i}" for i in range(40636)])
    np.save(tmp_path / "row_ids.npy", ids, allow_pickle=False)
    names = (
        ["row_ids.npy"]
        + [
            f"{part}_{h}.npy"
            for h in (90, 180, 365)
            for part in ("history", "mask", "count")
        ]
        + ["map_membership.json"]
    )
    np.save(
        tmp_path / "history_90.npy", np.zeros((40636, 2, 5, 1, 33), dtype=np.float16)
    )
    np.save(tmp_path / "mask_90.npy", np.zeros((40636, 2, 5, 1), dtype=np.uint8))
    np.save(tmp_path / "count_90.npy", np.zeros((40636, 2, 5), dtype=np.uint16))
    (tmp_path / "map_membership.json").write_text("{}")
    for horizon in (180, 365):
        np.save(
            tmp_path / f"history_{horizon}.npy",
            np.zeros((40636, 2, 5, 1, 33), dtype=np.float16),
        )
        np.save(
            tmp_path / f"mask_{horizon}.npy", np.zeros((40636, 2, 5, 1), dtype=np.uint8)
        )
        np.save(
            tmp_path / f"count_{horizon}.npy", np.zeros((40636, 2, 5), dtype=np.uint16)
        )
    for name in names:
        artifacts[name] = _sha256(tmp_path / name)
    manifest = {
        "schema": "c0_raw_causal_anchor_bank_v2",
        "artifacts": artifacts,
        "artifact_sha256": hashlib.sha256(
            json.dumps(artifacts, sort_keys=True).encode()
        ).hexdigest(),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / "verification_receipt.json").write_text(
        json.dumps(
            {
                "status": "VERIFIED",
                "row_count": 40636,
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": _sha256(manifest_path),
            }
        )
    )
    assert _verify_raw(manifest_path, {})["row_count"] == 40636
    (tmp_path / "mask_90.npy").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        _verify_raw(manifest_path, {})
