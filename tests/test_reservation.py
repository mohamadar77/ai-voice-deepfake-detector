"""Dataset-free tests for the speaker-grouped development reservation (#31).

Rows are synthesised in memory, so these tests touch no ASVspoof audio, no
network, no GPU and no trained checkpoint. The committed configuration under
``configs/`` is loaded as-is, so a change to the frozen rule breaks a test.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from voice_detector import reservation as r
from voice_detector.manifest import compute_manifest_hash, validate_manifest

_DEV_SPEAKERS = tuple(f"LA_{index:04d}" for index in range(1, 21))



# --- the committed specification --------------------------------------------


def test_committed_config_declares_the_frozen_rule(reservation_spec: r.ReservationSpec) -> None:
    assert reservation_spec.version == r.RESERVATION_SCHEMA_VERSION
    assert reservation_spec.method == r.ASSIGNMENT_METHOD
    assert reservation_spec.threshold_fraction == 0.25
    assert reservation_spec.coverage.require_attacks == r.KNOWN_ATTACKS
    # Asserted exactly, not as ">= 1" (which the loader already guarantees), so
    # a change to the committed requirement is visible here.
    assert reservation_spec.coverage.min_bonafide_clips_per_threshold_speaker == 1
    assert reservation_spec.coverage.require_both_labels_per_role is True
    # The reservation seed is deliberately independent of the project seed.
    assert isinstance(reservation_spec.seed, int)


def _write_spec(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


_VALID_SPEC = """
version: dev-reservation-v1
method: sha256-rank-v1
seed: 7
threshold_fraction: 0.25
coverage:
  require_attacks: [A01]
  min_bonafide_clips_per_threshold_speaker: 1
  require_both_labels_per_role: true
"""


@pytest.mark.parametrize(
    ("replace", "with_", "match"),
    [
        ("version: dev-reservation-v1", "version: dev-reservation-v9", "version"),
        ("method: sha256-rank-v1", "method: numpy-shuffle", "method"),
        ("seed: 7", "seed: half", "seed"),
        ("seed: 7", "seed: true", "seed"),
        ("threshold_fraction: 0.25", "threshold_fraction: 0", "threshold_fraction"),
        ("threshold_fraction: 0.25", "threshold_fraction: 1.0", "threshold_fraction"),
        ("require_attacks: [A01]", "require_attacks: []", "require_attacks"),
        ("require_attacks: [A01]", "require_attacks: [A01, A01]", "duplicates"),
        (
            "min_bonafide_clips_per_threshold_speaker: 1",
            "min_bonafide_clips_per_threshold_speaker: 0",
            "min_bonafide",
        ),
        (
            "require_both_labels_per_role: true",
            "require_both_labels_per_role: maybe",
            "require_both_labels_per_role",
        ),
    ],
)
def test_malformed_spec_is_rejected(
    tmp_path: Path, replace: str, with_: str, match: str
) -> None:
    path = _write_spec(tmp_path / "spec.yaml", _VALID_SPEC.replace(replace, with_))
    with pytest.raises(ValueError, match=match):
        r.load_reservation_spec(path)


def test_spec_without_coverage_section_is_rejected(tmp_path: Path) -> None:
    text = _VALID_SPEC.split("coverage:")[0]
    with pytest.raises(ValueError, match="coverage"):
        r.load_reservation_spec(_write_spec(tmp_path / "spec.yaml", text))


# --- deterministic assignment -----------------------------------------------


def test_reserved_count_rounds_half_up_and_clamps() -> None:
    assert r.n_reserved_speakers(20, 0.25) == 5
    # 18 * 0.25 = 4.5 rounds up, where banker's rounding would give 4.
    assert r.n_reserved_speakers(18, 0.25) == 5
    # Neither role may be empty, whatever the fraction asks for.
    assert r.n_reserved_speakers(4, 0.01) == 1
    assert r.n_reserved_speakers(4, 0.99) == 3
    assert r.n_reserved_speakers(2, 0.25) == 1
    with pytest.raises(ValueError, match="at least 2 speakers"):
        r.n_reserved_speakers(1, 0.25)


def test_assignment_is_deterministic_and_order_independent(
    reservation_spec: r.ReservationSpec,
) -> None:
    first = r.assign_development_roles(_DEV_SPEAKERS, reservation_spec)
    again = r.assign_development_roles(_DEV_SPEAKERS, reservation_spec)
    shuffled = r.assign_development_roles(tuple(reversed(_DEV_SPEAKERS)), reservation_spec)
    duplicated = r.assign_development_roles(_DEV_SPEAKERS + _DEV_SPEAKERS, reservation_spec)
    assert first == again == shuffled == duplicated
    assert sorted(first) == sorted(_DEV_SPEAKERS)
    reserved = [s for s, role in first.items() if role == r.ROLE_THRESHOLD]
    assert len(reserved) == 5


def test_assignment_depends_on_the_seed(reservation_spec: r.ReservationSpec) -> None:
    from dataclasses import replace as dc_replace

    memberships = {
        frozenset(
            speaker
            for speaker, role in r.assign_development_roles(
                _DEV_SPEAKERS, dc_replace(reservation_spec, seed=seed)
            ).items()
            if role == r.ROLE_THRESHOLD
        )
        for seed in (1, 2, 3, 4, 5)
    }
    # A seed that could not move the membership would make the seed decorative.
    assert len(memberships) > 1


def test_assignment_rejects_empty_or_blank_speakers(reservation_spec: r.ReservationSpec) -> None:
    with pytest.raises(ValueError, match="no development speakers"):
        r.assign_development_roles([], reservation_spec)
    with pytest.raises(ValueError, match="non-empty strings"):
        r.assign_development_roles(["LA_0001", ""], reservation_spec)


# --- writing roles into manifest rows ---------------------------------------


def test_roles_are_speaker_grouped_and_no_speaker_holds_both(
    reservation_spec: r.ReservationSpec,
    make_rows: Callable[..., list[dict[str, Any]]],
) -> None:
    rows = make_rows()
    assignment = r.assign_development_roles(r.development_speakers(rows), reservation_spec)
    updated = r.apply_development_roles(rows, assignment)

    roles_per_speaker: dict[str, set[str | None]] = {}
    for row in updated:
        if row["official_split"] == "development":
            roles_per_speaker.setdefault(row["speaker_id"], set()).add(row["development_role"])
    assert all(len(roles) == 1 for roles in roles_per_speaker.values())
    assert r.extract_role_assignment(updated) == assignment
    # The written manifest still satisfies the data contract.
    validate_manifest(updated)


def test_non_development_rows_keep_a_null_role(
    reservation_spec: r.ReservationSpec, make_rows: Callable[..., list[dict[str, Any]]]
) -> None:
    rows = make_rows()
    # A stale role on a train row must be cleared, not preserved.
    rows[0]["development_role"] = "select"
    assignment = r.assign_development_roles(r.development_speakers(rows), reservation_spec)
    updated = r.apply_development_roles(rows, assignment)
    assert all(
        row["development_role"] is None
        for row in updated
        if row["official_split"] != "development"
    )
    assert all(
        row["development_role"] in r.RESERVATION_ROLES
        for row in updated
        if row["official_split"] == "development"
    )


def test_apply_rejects_unassigned_and_foreign_speakers(
    reservation_spec: r.ReservationSpec, make_rows: Callable[..., list[dict[str, Any]]]
) -> None:
    rows = make_rows()
    assignment = r.assign_development_roles(r.development_speakers(rows), reservation_spec)
    partial = dict(assignment)
    partial.pop(_DEV_SPEAKERS[0])
    with pytest.raises(ValueError, match="no role assigned"):
        r.apply_development_roles(rows, partial)

    foreign = dict(assignment)
    foreign["LA_0201"] = r.ROLE_SELECT  # an evaluation speaker
    with pytest.raises(ValueError, match="outside the development split"):
        r.apply_development_roles(rows, foreign)

    bad_role = dict(assignment)
    bad_role[_DEV_SPEAKERS[0]] = "calibration"
    with pytest.raises(ValueError, match="roles must be one of"):
        r.apply_development_roles(rows, bad_role)


def test_development_row_without_a_speaker_is_rejected(
    reservation_spec: r.ReservationSpec,
    make_rows: Callable[..., list[dict[str, Any]]],
    make_row: Callable[..., dict[str, Any]],
) -> None:
    rows = make_rows()
    rows.append(make_row("LA_D_anon", speaker_id=None, official_split="development"))
    with pytest.raises(ValueError, match="no speaker_id"):
        r.development_speakers(rows)
    with pytest.raises(ValueError, match="no speaker_id"):
        r.plan_reservation(rows, reservation_spec)


def test_extract_rejects_a_speaker_with_two_roles(
    make_development_rows: Callable[..., list[dict[str, Any]]],
) -> None:
    rows = make_development_rows(speakers=("LA_0001",))
    rows[0]["development_role"] = r.ROLE_SELECT
    rows[1]["development_role"] = r.ROLE_THRESHOLD
    with pytest.raises(ValueError, match="more than one development_role"):
        r.extract_role_assignment(rows)


def test_extract_detects_a_half_written_reservation(
    make_development_rows: Callable[..., list[dict[str, Any]]],
) -> None:
    rows = make_development_rows(speakers=("LA_0001",))
    rows[0]["development_role"] = r.ROLE_SELECT  # the rest are still null
    with pytest.raises(ValueError, match="more than one development_role"):
        r.extract_role_assignment(rows)


def test_extract_on_an_untouched_manifest_is_empty(
    make_rows: Callable[..., list[dict[str, Any]]],
) -> None:
    assert r.extract_role_assignment(make_rows()) == {}


# --- coverage ---------------------------------------------------------------


def test_coverage_passes_and_reports_label_attack_counts(
    reservation_spec: r.ReservationSpec,
    make_rows: Callable[..., list[dict[str, Any]]],
) -> None:
    plan = r.plan_reservation(make_rows(), reservation_spec)
    coverage = plan.coverage
    assert coverage.ok, coverage.failures
    by_role = {role.role: role for role in coverage.roles}
    assert by_role[r.ROLE_THRESHOLD].n_speakers == 5
    assert by_role[r.ROLE_SELECT].n_speakers == 15
    assert by_role[r.ROLE_THRESHOLD].n_bonafide == 5 * 2
    assert by_role[r.ROLE_THRESHOLD].attacks == r.KNOWN_ATTACKS
    assert by_role[r.ROLE_THRESHOLD].missing_attacks == ()
    # Counts are reported per label and per attack, not just per role.
    spoof_a01 = next(
        item
        for item in by_role[r.ROLE_THRESHOLD].counts_by_label_attack
        if item["attack_id"] == "A01"
    )
    assert spoof_a01 == {"label": "spoof", "attack_id": "A01", "clips": 5}
    bonafide = next(
        item
        for item in by_role[r.ROLE_SELECT].counts_by_label_attack
        if item["label"] == "bonafide"
    )
    assert bonafide == {"label": "bonafide", "attack_id": None, "clips": 30}
    # Per-speaker evidence covers every development speaker.
    assert len(coverage.per_speaker) == 20


def test_coverage_fails_when_a_reserved_speaker_has_no_bonafide_clip(
    reservation_spec: r.ReservationSpec,
    make_rows: Callable[..., list[dict[str, Any]]],
) -> None:
    rows = make_rows()
    plan = r.plan_reservation(rows, reservation_spec)
    reserved = next(
        speaker for speaker, role in plan.assignment.items() if role == r.ROLE_THRESHOLD
    )
    stripped = [
        row
        for row in rows
        if not (row["speaker_id"] == reserved and row["label"] == "bonafide")
    ]
    coverage = r.plan_reservation(stripped, reservation_spec).coverage
    assert not coverage.ok
    assert any("bona fide" in failure for failure in coverage.failures)
    threshold_role = next(
        role for role in coverage.roles if role.role == r.ROLE_THRESHOLD
    )
    assert reserved in threshold_role.speakers_below_min_bonafide


def test_coverage_fails_when_a_role_is_missing_an_attack(
    reservation_spec: r.ReservationSpec, make_rows: Callable[..., list[dict[str, Any]]]
) -> None:
    rows = make_rows(attacks=("A01", "A02", "A03", "A04", "A05"))
    coverage = r.plan_reservation(rows, reservation_spec).coverage
    assert not coverage.ok
    assert any("A06" in failure for failure in coverage.failures)


def test_coverage_fails_when_a_role_has_only_one_class(
    reservation_spec: r.ReservationSpec, make_rows: Callable[..., list[dict[str, Any]]]
) -> None:
    rows = make_rows(n_bonafide=0)
    coverage = r.plan_reservation(rows, reservation_spec).coverage
    assert not coverage.ok
    assert any("bonafide" in failure for failure in coverage.failures)


# --- planning, the record and verification ----------------------------------


def test_freezing_roles_changes_the_manifest_hash(
    reservation_spec: r.ReservationSpec, make_rows: Callable[..., list[dict[str, Any]]]
) -> None:
    rows = make_rows()
    plan = r.plan_reservation(rows, reservation_spec)
    assert plan.manifest_hash_before == compute_manifest_hash(rows)
    assert plan.manifest_hash_after != plan.manifest_hash_before
    assert plan.manifest_hash_after == compute_manifest_hash(plan.rows)
    # The input rows are never mutated in place.
    assert all(row["development_role"] is None for row in rows)


def test_replanning_a_frozen_manifest_is_a_no_op(
    reservation_spec: r.ReservationSpec, make_rows: Callable[..., list[dict[str, Any]]]
) -> None:
    first = r.plan_reservation(make_rows(), reservation_spec)
    second = r.plan_reservation(first.rows, reservation_spec)
    assert second.already_frozen
    assert not second.conflicts_with_existing
    assert second.manifest_hash_before == second.manifest_hash_after
    assert second.assignment == first.assignment


def test_a_different_existing_reservation_is_flagged_as_a_conflict(
    reservation_spec: r.ReservationSpec,
    make_rows: Callable[..., list[dict[str, Any]]],
) -> None:
    from dataclasses import replace as dc_replace

    rows = make_rows()
    other = r.plan_reservation(rows, dc_replace(reservation_spec, seed=reservation_spec.seed + 1))
    # Both seeds and the speaker list are fixed, so the two memberships either
    # always differ or never do. Assert that precondition instead of skipping on
    # it, which would silently stop testing the conflict path.
    assert other.assignment != r.assign_development_roles(_DEV_SPEAKERS, reservation_spec)
    plan = r.plan_reservation(other.rows, reservation_spec)
    assert plan.conflicts_with_existing
    assert not plan.already_frozen


def test_frozen_record_holds_speaker_ids_only(
    reservation_spec: r.ReservationSpec, make_rows: Callable[..., list[dict[str, Any]]]
) -> None:
    plan = r.plan_reservation(make_rows(), reservation_spec)
    record = r.build_reservation_record(plan, note="coverage held at 0.25")

    assert record["schema_version"] == r.RESERVATION_SCHEMA_VERSION
    assert record["spec"]["seed"] == reservation_spec.seed
    assert record["n_speakers"] == {r.ROLE_SELECT: 15, r.ROLE_THRESHOLD: 5}
    assert record["achieved_threshold_fraction"] == 0.25
    assert record["manifest_hash_after"] == plan.manifest_hash_after
    assert record["coverage_ok"] is True
    assert record["note"] == "coverage held at 0.25"
    assert "disclosure" in record

    reserved = set(record["speakers"][r.ROLE_THRESHOLD])
    selected = set(record["speakers"][r.ROLE_SELECT])
    assert reserved.isdisjoint(selected)
    assert reserved | selected == set(_DEV_SPEAKERS)
    assert record["speakers"][r.ROLE_THRESHOLD] == sorted(reserved)

    # No clip identifiers, paths or audio leak into the committed record.
    blob = json.dumps(record)
    assert "_D_" not in blob
    assert "local_path" not in blob
    assert ".flac" not in blob


def test_verify_accepts_a_matching_manifest_and_rejects_a_moved_speaker(
    reservation_spec: r.ReservationSpec,
    make_rows: Callable[..., list[dict[str, Any]]],
) -> None:
    plan = r.plan_reservation(make_rows(), reservation_spec)
    record = r.build_reservation_record(plan)
    r.verify_frozen_reservation(plan.rows, record)

    moved = set(record["speakers"][r.ROLE_THRESHOLD])
    swapped = [
        {**row, "development_role": r.ROLE_SELECT}
        if row["speaker_id"] in moved and row["official_split"] == "development"
        else row
        for row in plan.rows
    ]
    with pytest.raises(ValueError, match="does not match the frozen record"):
        r.verify_frozen_reservation(swapped, record)


def test_verify_rejects_a_manifest_whose_clips_changed(
    reservation_spec: r.ReservationSpec, make_rows: Callable[..., list[dict[str, Any]]]
) -> None:
    plan = r.plan_reservation(make_rows(), reservation_spec)
    record = r.build_reservation_record(plan)
    dropped = [row for row in plan.rows if row["clip_id"] != "LA_E_1"]
    with pytest.raises(ValueError, match="manifest hash"):
        r.verify_frozen_reservation(dropped, record)


def test_verify_rejects_a_malformed_record(
    reservation_spec: r.ReservationSpec, make_rows: Callable[..., list[dict[str, Any]]]
) -> None:
    plan = r.plan_reservation(make_rows(), reservation_spec)
    record = r.build_reservation_record(plan)

    with pytest.raises(ValueError, match="schema_version"):
        r.verify_frozen_reservation(plan.rows, {})
    with pytest.raises(ValueError, match="no 'speakers' mapping"):
        r.verify_frozen_reservation(plan.rows, {**record, "speakers": None})
    with pytest.raises(ValueError, match="unknown role"):
        r.verify_frozen_reservation(
            plan.rows, {**record, "speakers": {"calibration": ["LA_0001"]}}
        )
    # A bare string would otherwise iterate into one key per character.
    with pytest.raises(ValueError, match="must be a list"):
        r.verify_frozen_reservation(
            plan.rows,
            {**record, "speakers": {r.ROLE_SELECT: "LA_0001", r.ROLE_THRESHOLD: ["LA_0002"]}},
        )


def test_verify_rejects_a_record_listing_one_speaker_under_both_roles(
    reservation_spec: r.ReservationSpec, make_rows: Callable[..., list[dict[str, Any]]]
) -> None:
    """A speaker in both roles must never be resolved by mapping order.

    Assigning into a plain dict would let whichever role iterated last win, so
    the same corrupted record would be caught or missed depending on key order
    -- and a speaker informing both a model choice and its operating point is
    exactly what the reservation forbids.
    """

    plan = r.plan_reservation(make_rows(), reservation_spec)
    record = r.build_reservation_record(plan)
    selected = record["speakers"][r.ROLE_SELECT][0]

    both_ways = [
        {
            r.ROLE_SELECT: record["speakers"][r.ROLE_SELECT],
            r.ROLE_THRESHOLD: [*record["speakers"][r.ROLE_THRESHOLD], selected],
        },
        # The same corruption with the roles the other way round, which a
        # last-write-wins mapping would have accepted.
        {
            r.ROLE_THRESHOLD: [*record["speakers"][r.ROLE_THRESHOLD], selected],
            r.ROLE_SELECT: record["speakers"][r.ROLE_SELECT],
        },
    ]
    for speakers in both_ways:
        with pytest.raises(ValueError, match="twice"):
            r.verify_frozen_reservation(plan.rows, {**record, "speakers": speakers})


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"schema_version": "dev-reservation-v99"}, "schema_version"),
        ({"spec": {"method": "numpy-shuffle", "seed": 1, "threshold_fraction": 0.25}}, "method"),
        (
            {"spec": {"method": r.ASSIGNMENT_METHOD, "seed": "nine", "threshold_fraction": 0.25}},
            "seed",
        ),
        (
            {"spec": {"method": r.ASSIGNMENT_METHOD, "seed": 1, "threshold_fraction": "a quarter"}},
            "threshold_fraction",
        ),
    ],
)
def test_verify_rejects_a_record_whose_own_rule_drifted(
    reservation_spec: r.ReservationSpec,
    make_rows: Callable[..., list[dict[str, Any]]],
    mutation: dict[str, Any],
    match: str,
) -> None:
    """The recorded schema and method are read back, not merely stored."""

    plan = r.plan_reservation(make_rows(), reservation_spec)
    record = r.build_reservation_record(plan)
    with pytest.raises(ValueError, match=match):
        r.verify_frozen_reservation(plan.rows, {**record, **mutation})


def test_verify_rejects_a_record_whose_membership_was_hand_edited(
    reservation_spec: r.ReservationSpec, make_rows: Callable[..., list[dict[str, Any]]]
) -> None:
    """Membership must be what the record's own seed and fraction derive.

    Swapping two speakers between roles and relabelling the matching manifest
    keeps every hash self-consistent, so only re-deriving the rule catches it.
    """

    plan = r.plan_reservation(make_rows(), reservation_spec)
    record = r.build_reservation_record(plan)
    swapped = {
        r.ROLE_SELECT: [
            *record["speakers"][r.ROLE_SELECT][1:],
            record["speakers"][r.ROLE_THRESHOLD][0],
        ],
        r.ROLE_THRESHOLD: [
            *record["speakers"][r.ROLE_THRESHOLD][1:],
            record["speakers"][r.ROLE_SELECT][0],
        ],
    }
    with pytest.raises(ValueError, match="edited after it was frozen"):
        r.verify_frozen_reservation(plan.rows, {**record, "speakers": swapped})


def test_reading_a_committed_record_validates_it(
    tmp_path: Path,
    reservation_spec: r.ReservationSpec,
    make_rows: Callable[..., list[dict[str, Any]]],
) -> None:
    plan = r.plan_reservation(make_rows(), reservation_spec)
    record = r.build_reservation_record(plan)
    path = tmp_path / "dev_reservation_frozen.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")

    loaded, assignment = r.read_frozen_record(path)
    assert loaded["manifest_hash_after"] == plan.manifest_hash_after
    assert assignment == plan.assignment

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        r.read_frozen_record(path)


# --- reading a manifest back ------------------------------------------------


def test_applying_roles_works_on_a_single_pass_iterable(
    reservation_spec: r.ReservationSpec, make_rows: Callable[..., list[dict[str, Any]]]
) -> None:
    """A generator must not validate against the rows and then rewrite none.

    ``apply_development_roles`` reads its input twice, to validate and then to
    rewrite. Iterating a generator twice yields nothing the second time, so the
    function would return an empty manifest with every check "passed" -- and
    streaming the 121k-row manifest is the natural way to call it.
    """

    rows = make_rows()
    assignment = r.assign_development_roles(r.development_speakers(rows), reservation_spec)

    from_list = r.apply_development_roles(rows, assignment)
    from_generator = r.apply_development_roles((row for row in rows), assignment)

    assert len(from_generator) == len(rows)
    assert from_generator == from_list


def test_an_empty_string_role_is_not_read_as_unassigned(
    make_development_rows: Callable[..., list[dict[str, Any]]],
) -> None:
    """``""`` is outside the data contract and must not pass as a null role.

    ``write_manifest_csv`` writes ``""`` for a null role, so a CSV round-trip
    can present it. Reading it as "unassigned" would make a half-written
    reservation look untouched, and therefore re-freezable.
    """

    rows = [{**row, "development_role": ""} for row in make_development_rows()]
    with pytest.raises(ValueError, match="empty string is not a null role"):
        r.extract_role_assignment(rows)


def test_a_development_row_without_a_speaker_is_rejected_when_reading_roles(
    make_row: Callable[..., dict[str, Any]],
) -> None:
    """Speaker-less rows must raise, not collapse into one ``"None"`` speaker."""

    rows = [
        make_row(
            clip_id="LA_D_1",
            speaker_id=None,
            official_split="development",
            development_role=r.ROLE_SELECT,
        ),
        make_row(
            clip_id="LA_D_2",
            speaker_id=None,
            official_split="development",
            development_role=r.ROLE_THRESHOLD,
        ),
    ]
    with pytest.raises(ValueError, match="no speaker_id"):
        r.extract_role_assignment(rows)


def test_mixed_type_speaker_ids_raise_this_modules_own_error(
    reservation_spec: r.ReservationSpec,
) -> None:
    """Validation runs before sorting, so the error is not a bare TypeError."""

    with pytest.raises(ValueError, match="non-empty strings"):
        r.assign_development_roles(["LA_0001", 7], reservation_spec)
    with pytest.raises(ValueError, match="non-empty strings"):
        r.assign_development_roles(["LA_0001", None], reservation_spec)


# --- a fraction the speaker set cannot express ------------------------------


@pytest.mark.parametrize(
    ("fraction", "reserved", "flagged"),
    [
        (0.25, 5, False),  # the committed rule: exact, no divergence
        (0.03, 1, False),  # rounds to one speaker, within half a speaker
        (0.001, 1, True),  # clamped: an achieved 0.05 against a requested 0.001
        (0.999, 19, True),  # clamped at the other end
    ],
)
def test_a_clamped_fraction_is_reported_as_a_coverage_failure(
    reservation_spec: r.ReservationSpec,
    make_rows: Callable[..., list[dict[str, Any]]],
    fraction: float,
    reserved: int,
    flagged: bool,
) -> None:
    """A clamp that moves the achieved fraction must fail, not just be recorded.

    Rounding can move the fraction by at most half a speaker; beyond that the
    ``[1, n - 1]`` clamp is silently delivering a different reservation than the
    one requested, which would otherwise be frozen as a success.
    """

    from dataclasses import replace as dc_replace

    spec = dc_replace(reservation_spec, threshold_fraction=fraction)
    plan = r.plan_reservation(make_rows(), spec)

    assert len(plan.speakers_by_role()[r.ROLE_THRESHOLD]) == reserved
    divergent = [failure for failure in plan.coverage.failures if "cannot be honoured" in failure]
    assert bool(divergent) is flagged
    if flagged:
        assert not plan.coverage.ok
