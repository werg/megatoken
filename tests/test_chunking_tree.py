"""Tree-distance semantics and a regression for deeply nested source evidence."""

from __future__ import annotations

import random
import time

import pytest

from megatoken.data.parsing.engine import parse_content
from megatoken.data.parsing.planner import _TreeDistances
from megatoken.data.parsing.types import Region


def test_native_hierarchy_counts_ancestry_and_preserves_trailing_whitespace():
    regions = [
        Region(0, 100, "module", "python", path=("module",)),
        Region(5, 30, "function", "python", path=("module", "first")),
        Region(8, 20, "code_block", "python", path=("module", "first", "body")),
        Region(40, 70, "function", "python", path=("module", "second")),
    ]
    tree = _TreeDistances(regions, code_scale=1.5)
    # Entering first/body costs two descents. Moving from body to the following
    # function costs two ascents and one descent, all in code context.
    assert tree.between(0, 100) == 7.5
    assert tree.between(5, 30) == 1.5
    assert tree.between(8, 30) == 0.0
    assert tree.between(70, 100) == 0.0
    assert 20 not in tree.offsets and 30 not in tree.offsets and 70 not in tree.offsets


def test_native_paths_override_geometry_when_the_parser_supplies_a_hierarchy():
    regions = [
        Region(0, 100, "section", "text", path=("document", "one")),
        Region(10, 20, "section", "text", path=("document", "two", "body")),
    ]
    tree = _TreeDistances(regions, code_scale=4.0)
    # Despite containment, the supplied paths identify a different branch: one
    # ascent to document followed by two descents, not an inferred child edge.
    assert tree.between(0, 100) == 3.0


def test_singleton_native_paths_are_real_names_not_generic_view_markers():
    regions = [
        Region(0, 10, "function", "python", path=("first",)),
        Region(20, 30, "function", "python", path=("second",)),
    ]
    assert _TreeDistances(regions, code_scale=1.5).between(0, 30) == 3.0


@pytest.mark.parametrize("path", [(), ("0:100:0",)])
def test_generic_regions_infer_containment_with_or_without_engine_view_prefix(path):
    regions = [
        Region(0, 100, "code_block", "unknown", path=path),
        Region(5, 20, "code_block", "unknown", path=path),
        Region(30, 50, "code_block", "unknown", path=path),
    ]
    tree = _TreeDistances(regions, code_scale=1.5)
    assert tree.between(0, 100) == 4.5  # One descent, then a sibling transition.
    assert tree.between(5, 20) == 0.0
    assert tree.between(50, 100) == 0.0


def test_generic_children_attach_to_the_supplied_native_parent_path():
    view = "0:100:0"
    regions = [
        Region(0, 100, "function", "python", path=(view, "function")),
        Region(5, 20, "code_block", "python", path=(view,)),
        Region(30, 50, "code_block", "python", path=(view,)),
    ]
    assert _TreeDistances(regions, code_scale=1).between(0, 100) == 3.0


def test_generic_crossings_do_not_turn_overlapping_siblings_into_ancestors():
    regions = [
        Region(0, 10, "code_block", "text"),
        Region(5, 15, "code_block", "text"),
        Region(6, 9, "code_block", "text"),
    ]
    # The last interval is assigned to the latest containing sibling. The sweep
    # eventually moves from the first sibling up to root and down two levels.
    assert _TreeDistances(regions, code_scale=1).between(0, 15) == 3.0


def test_coextensive_fallback_annotations_do_not_create_fake_depth():
    outer = Region(0, 100, "code_block", "unknown", path=("0:100:0",))
    child = Region(5, 20, "function", "unknown", path=("0:100:0",))
    expected = _TreeDistances([outer, child], code_scale=1.5)
    repeated = _TreeDistances([outer] * 100 + [child], code_scale=1.5)
    assert repeated.between(0, 100) == expected.between(0, 100) == 1.5


def test_code_scaling_tracks_all_active_context_and_stops_after_its_exit():
    regions = [
        Region(0, 40, "function", "python", path=("code",)),
        Region(10, 20, "section", "text", path=("code", "docstring")),
        Region(50, 70, "section", "text", path=("document",)),
    ]
    tree = _TreeDistances(regions, code_scale=2)
    assert tree.between(0, 40) == 2.0
    assert tree.between(40, 70) == 3.0  # Subsequent prose uses ordinary edge weight.
    assert _TreeDistances(regions, code_scale=0).between(0, 40) == 0.0


def test_binary_lifting_handles_long_shared_prefix_and_unequal_branch_depth():
    prefix = tuple(f"ancestor-{index}" for index in range(80))
    first = (*prefix, *(f"inner-{index}" for index in range(20)))
    second = (*prefix, "other")
    regions = [
        Region(0, 100, "function", "python", path=first),
        Region(101, 200, "function", "python", path=second),
    ]
    assert _TreeDistances(regions, code_scale=2).between(0, 200) == 42.0


def test_varied_native_branches_match_known_shared_prefix_distances():
    generator = random.Random(4912)
    paths = [("root",)]
    for index in range(150):
        parent = generator.choice(paths)
        paths.append((*parent, f"branch-{index}"))
    selected = [generator.choice(paths) for _ in range(200)]
    regions = [
        Region(index * 10, index * 10 + 5, "function", "python", path=path)
        for index, path in enumerate(selected)
    ]
    tree = _TreeDistances(regions, code_scale=1.5)
    expected = 0
    for index in range(1, len(selected)):
        previous, current = selected[index - 1], selected[index]
        common = 0
        for left, right in zip(previous, current, strict=False):
            if left != right:
                break
            common += 1
        expected += (len(previous) + len(current) - 2 * common) * 1.5
        assert tree.between(0, index * 10 + 5) == expected


def test_actual_recursive_generic_parser_paths_retain_nonzero_movement():
    source = b"{" * 24 + b"value" + b"}" * 24
    parsed = parse_content(source, content_kind="code", language="unknown", backend="fallback")
    containers = [region for region in parsed.regions if region.kind == "code_block"]
    assert len(containers) == 24
    assert all(len(region.path) == 1 for region in containers)
    assert _TreeDistances(containers, code_scale=1.5).between(0, len(source)) == 23 * 1.5


def test_twenty_thousand_nested_regions_do_not_require_quadratic_work(record_property):
    count = 20_000
    end = 2 * count + 1
    regions = [
        Region(index, end - index, "code_block", "unknown", path=(f"0:{end}:0",))
        for index in range(count)
    ]
    started = time.monotonic()
    tree = _TreeDistances(regions, code_scale=1.5)
    elapsed = time.monotonic() - started
    record_property("nested_region_count", count)
    record_property("construction_seconds", elapsed)
    # The former active-region scan grows quadratically and takes tens of seconds
    # here. The event sweep is normally under one second; leave generous CI slack.
    assert elapsed < 5.0
    assert len(tree.offsets) == count
    assert tree.between(0, end) == (count - 1) * 1.5
    assert tree.between(count, end) == 0.0


def test_generic_hierarchy_is_independent_of_region_traversal_order():
    regions = [Region(i, 201 - i, "code_block", "unknown") for i in range(100)]
    expected = _TreeDistances(regions, code_scale=1.5)
    random.Random(473).shuffle(regions)
    shuffled = _TreeDistances(regions, code_scale=1.5)
    assert shuffled.offsets == expected.offsets
    assert shuffled.cumulative == expected.cumulative


def test_empty_evidence_and_non_container_leaves_have_no_tree_distance():
    regions = [Region(1, 20, "word", "text"), Region(30, 30, "code_block", "python")]
    tree = _TreeDistances(regions, code_scale=2)
    assert tree.between(0, 100) == 0.0
