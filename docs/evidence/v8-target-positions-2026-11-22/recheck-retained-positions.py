"""Recheck retained actual fixture receipts; execute no optimizer update.

Run from the native repository with PYTHONPATH=src. Original prepared fixture
roots named by the retained configurations must still be present. This audit
uses their explicit controlled byte-codec boundary, not production admission.
"""

import hashlib
import json
import tarfile
from pathlib import Path

from megatoken.data import PreparedCorpus
from megatoken.generation.codec import ByteCodec
from megatoken.loading import load_model_config
from megatoken.training.config import TrainConfig
from megatoken.training.positions_v8 import (
    compare_position_inventory,
    independent_position_inventory,
    validate_position_exposure,
)


def independent(config_raw, expected):
    config = TrainConfig.from_dict(config_raw)
    manifest = json.loads((Path(config.datasets[0]) / "manifest.json").read_text())
    codec = ByteCodec.byte_fallback()
    assert manifest["fingerprint"] == expected["preparedManifestFingerprint"]
    assert manifest["codec_fingerprint"] == codec.fingerprint
    model = load_model_config(
        config.model,
        **{
            **config.model_overrides,
            "expert_backend": config.expert_backend,
            "checkpoint_stages": config.gradient_checkpointing,
        },
    )
    with PreparedCorpus(config.datasets, split="train") as corpus:
        rebuilt = independent_position_inventory(
            (corpus.fetch(index)[0] for index in range(len(corpus))),
            codec=codec,
            config=config,
            model_config=model,
            modalities=[0],
            prepared_manifest_fingerprint=expected["preparedManifestFingerprint"],
            expected_source_ids=expected["sourceIds"],
        )
    assert rebuilt == expected
    return rebuilt


def main():
    root = Path(__file__).resolve().parent
    names = ("controlled-complete-position-proof.json", "controlled-resume-position-proof.json")
    complete, resume = (json.loads((root / name).read_text()) for name in names)
    inventory = independent(complete["config"], complete["independentPreparedPositionInventory"])
    comparison = compare_position_inventory(
        inventory,
        complete["optimizerPositionExposure"],
        target_exposure=complete["targetExposure"],
        global_step=complete["optimizerUpdates"],
    )
    assert comparison["exactOnePassPreparedTextPositions"] is True
    stage = complete["phaseTransition"]["positionExposure"]
    validate_position_exposure(
        stage, registry_digest=inventory["registryDigest"], objective=stage["objective"]
    )
    assert stage["previousStage"] == complete["optimizerPositionExposure"]
    with tarfile.open(root / "controlled-runtime-reports.tar.gz") as archive:
        config = json.load(archive.extractfile("resume/config.json"))
    resumed_inventory = independent(config, resume["independentPreparedPositionInventory"])
    for name in ("firstPositionExposure", "resumedPositionExposure"):
        validate_position_exposure(
            resume[name],
            registry_digest=resumed_inventory["registryDigest"],
            objective=resumed_inventory["objective"],
        )
    assert (
        resume["resumedPositionExposure"]["previousDigest"]
        == resume["firstPositionExposure"]["digest"]
    )
    report = {
        "schema": "semantic-v8-stricter-position-validation/1",
        "status": "passed",
        "auditCommit": "8eae121e86cd0cb2735a5bc4f1e3e82026d738f9",
        "actualOptimizerCommit": "16b6a989c79d5e4a1910a03ccee73e8817b5a95b",
        "actualLocalOptimizerCommit": "436b129d46cc3c4c4bf97bab3779e93803472656",
        "optimizerUpdatesPerformed": 0,
        "independentPreparedInventoriesRebuilt": 2,
        "preparedManifestBindingsChecked": 2,
        "actualPositionStatesValidated": 4,
        "completePositionComparison": comparison,
        "exactPriorStagePreserved": True,
        "exactResumeParentPreserved": True,
        "proofSha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
        },
        "scope": "Stricter canonical objective validation of unchanged actual fixture receipts, with fresh prepared-record reconstruction; source/tokenizer boundaries remain injected",
        "routerInternalPositionCoverage": "unmeasured",
        "learnedSuccess": None,
    }
    (root / "stricter-position-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
