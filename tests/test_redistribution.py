"""Every text source records whether its prepared rows may be published, and the risky ones say no."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from megatoken.data.sources import REDISTRIBUTION_GROUPS, TermsReview, load_source_manifest

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = sorted(
    path
    for path in (ROOT / "configs/data").glob("sources.*.json")
    if not path.name.startswith("sources.media")
)


def all_sources():
    for path in MANIFESTS:
        for source in json.loads(path.read_text())["sources"]:
            yield path.name, source


def test_the_text_manifests_are_the_ones_expected():
    assert len(MANIFESTS) == 11
    assert sum(1 for _ in all_sources()) > 1000


def test_every_text_source_records_a_decision_and_its_basis():
    for name, source in all_sources():
        terms = source["terms"]
        where = f"{name}: {source['source_id']}"
        assert terms.get("redistribute") in REDISTRIBUTION_GROUPS, where
        assert len(terms.get("redistribute_basis", "")) > 20, where
    # The decision is read back by the loader.
    for path in MANIFESTS:
        for spec in load_source_manifest(path):
            assert spec.terms.redistribute in REDISTRIBUTION_GROUPS


def test_the_manifests_keep_their_exact_formatting():
    for path in MANIFESTS:
        text = path.read_text()
        assert text == json.dumps(json.loads(text), indent=2, ensure_ascii=True) + "\n", path.name


GATED = re.compile(r"gated|click agree|accept the|manual approval|auto-?approv|request access", re.I)
UNCLEAR_LICENCE = re.compile(
    r"not stated|^none|unknown|^other|research|custom|\bnd\b|no-?deriv|gpl|rail|impact|lmsys|llama|"
    r"gemma|nvidia|microsoft|midtool|source-specific|cdla-sharing|per[- ]file|per row|^cc \(|inferred|"
    r"rail-d|^source",
    re.I,
)


def test_gated_unlicensed_nd_and_custom_licence_sources_are_not_published():
    for name, source in all_sources():
        terms = source["terms"]
        where = f"{name}: {source['source_id']} ({terms['license']})"
        if name.endswith("-gated.json") or GATED.search(terms["notes"] + source.get("notes", "")):
            assert terms["redistribute"] == "no", where
        if UNCLEAR_LICENCE.search(terms["license"]):
            assert terms["redistribute"] == "no", where


PERMISSIVE = re.compile(r"^(apache-2\.0|mit|cc-by-4\.0|odc-by|cc0|cdla-permissive-2\.0|openmdw-1\.1)", re.I)
SHARE_ALIKE = re.compile(r"^(cc-by-sa|cc-by-4\.0|cc by-sa|cc-by-4\.0 and cc-by-sa)", re.I)
NON_COMMERCIAL = re.compile(r"^cc-by-nc", re.I)


def test_a_published_source_states_a_licence_of_its_group():
    for name, source in all_sources():
        terms = source["terms"]
        group, licence = terms["redistribute"], terms["license"]
        where = f"{name}: {source['source_id']} ({licence})"
        if group == "permissive":
            assert PERMISSIVE.match(licence), where
        elif group == "share_alike":
            # A permissive licence appears here only when the notes record CC-BY-SA rows in it.
            assert SHARE_ALIKE.match(licence) or PERMISSIVE.match(licence), where
            assert "sa" in (licence + terms["redistribute_basis"]).lower(), where
        elif group == "non_commercial":
            assert NON_COMMERCIAL.match(licence), where


def test_every_group_is_used_and_the_default_is_the_safe_one():
    used = {source["terms"]["redistribute"] for _, source in all_sources()}
    assert used == set(REDISTRIBUTION_GROUPS)
    assert TermsReview().group == "no" and TermsReview(redistribute="permissive").group == "permissive"
    with pytest.raises(ValueError):
        TermsReview(redistribute="yes")
