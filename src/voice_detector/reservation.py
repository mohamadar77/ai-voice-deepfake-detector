"""Speaker-grouped development reservation: ``select`` versus ``threshold``.

The ASVspoof 2019 LA development partition has to serve two jobs that must not
contaminate each other:

* ``select`` (``dev_select``) chooses features, hyperparameters, checkpoints and
  the deployed system;
* ``threshold`` (``dev_threshold``) is untouched by those choices and sets the
  operating threshold *after* candidates are frozen
  (``metrics.threshold_at_target_fpr``).

The reservation is **speaker-grouped**: every clip of a speaker carries the same
role, so no speaker's voice informs both a model choice and its operating point.
Membership is frozen *before* any tuning: once a score has been looked at, the
contamination cannot be undone (see ``AGENTS.md``, "Non-negotiable evaluation
constraints", and the architecture plan's leakage controls 2-4).

**Determinism without library drift.** A speaker's rank is
``sha256("<seed>|<speaker_id>")``; the lowest-ranked speakers are reserved for
``threshold``. SHA-256 is fixed by its specification, so the same seed and
speaker list reproduce the same membership on any machine, Python or NumPy
version — unlike a PRNG stream, whose shuffling algorithm is not guaranteed
stable across releases. The committed record (``reports/``) remains the
authoritative membership; this function is how it was derived.

This module reads no audio and fits nothing, so it runs in fast, dataset-free
tests. The real membership freeze runs on the host against the full manifest,
where it rewrites the ``development_role`` column and therefore changes the
manifest hash.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from voice_detector.manifest import (
    VALID_LABELS,
    compute_manifest_hash,
    git_commit,
    package_version,
)

# Bump when the assignment rule, the coverage contract or the record shape
# changes. Recorded in the frozen record so a membership can never be silently
# re-derived under different rules.
RESERVATION_SCHEMA_VERSION = "dev-reservation-v1"

# The assignment rule itself, named separately from the schema: the record keeps
# both, so a future schema change does not imply a membership change.
ASSIGNMENT_METHOD = "sha256-rank-v1"

ROLE_SELECT = "select"
ROLE_THRESHOLD = "threshold"
RESERVATION_ROLES = (ROLE_SELECT, ROLE_THRESHOLD)
DEVELOPMENT_SPLIT = "development"

# ASVspoof 2019 LA train and development share the attacks A01-A06. Both roles
# must cover all six, or a role cannot speak to the full known-attack set.
KNOWN_ATTACKS = tuple(f"A{index:02d}" for index in range(1, 7))


# ---------------------------------------------------------------------------
# Specification (versioned YAML; never a CLI default)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageRequirements:
    """What a reservation must satisfy before its membership may be frozen."""

    require_attacks: tuple[str, ...]
    min_bonafide_clips_per_threshold_speaker: int
    require_both_labels_per_role: bool


@dataclass(frozen=True)
class ReservationSpec:
    """The frozen reservation rule: fraction, seed and method."""

    version: str
    seed: int
    threshold_fraction: float
    method: str
    coverage: CoverageRequirements

    def as_dict(self) -> dict[str, Any]:
        """Plain, JSON-serializable form for the frozen record."""

        return asdict(self)


def load_reservation_spec(path: Path | str) -> ReservationSpec:
    """Load and validate the reservation specification from versioned YAML.

    Rejects an unknown schema version or assignment method, a fraction outside
    ``(0, 1)``, and an empty attack-coverage requirement, so a malformed or
    silently upgraded config can never produce a "valid" freeze.
    """

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Reservation config must be a mapping: {path}")

    version = str(raw.get("version"))
    if version != RESERVATION_SCHEMA_VERSION:
        raise ValueError(
            f"reservation config version must be {RESERVATION_SCHEMA_VERSION!r}, got {version!r}"
        )
    method = str(raw.get("method"))
    if method != ASSIGNMENT_METHOD:
        raise ValueError(
            f"reservation method must be {ASSIGNMENT_METHOD!r}, got {method!r}"
        )

    seed = raw.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"reservation seed must be an integer, got {seed!r}")

    fraction = raw.get("threshold_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, int | float):
        raise ValueError(f"threshold_fraction must be a number, got {fraction!r}")
    fraction = float(fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"threshold_fraction must be in (0, 1), got {fraction}")

    coverage_raw = raw.get("coverage")
    if not isinstance(coverage_raw, Mapping):
        raise ValueError("reservation config needs a 'coverage' mapping")
    attacks = coverage_raw.get("require_attacks")
    if not isinstance(attacks, list) or not attacks:
        raise ValueError("coverage.require_attacks must be a non-empty list")
    if not all(isinstance(attack, str) and attack for attack in attacks):
        raise ValueError("coverage.require_attacks must contain non-empty strings")
    duplicates = sorted({a for a, n in Counter(attacks).items() if n > 1})
    if duplicates:
        raise ValueError(f"coverage.require_attacks has duplicates: {duplicates}")
    min_bonafide = coverage_raw.get("min_bonafide_clips_per_threshold_speaker")
    if isinstance(min_bonafide, bool) or not isinstance(min_bonafide, int) or min_bonafide < 1:
        raise ValueError(
            "coverage.min_bonafide_clips_per_threshold_speaker must be an integer >= 1, "
            f"got {min_bonafide!r}"
        )
    both_labels = coverage_raw.get("require_both_labels_per_role")
    if not isinstance(both_labels, bool):
        raise ValueError("coverage.require_both_labels_per_role must be a boolean")

    return ReservationSpec(
        version=version,
        seed=seed,
        threshold_fraction=fraction,
        method=method,
        coverage=CoverageRequirements(
            require_attacks=tuple(sorted(attacks)),
            min_bonafide_clips_per_threshold_speaker=min_bonafide,
            require_both_labels_per_role=both_labels,
        ),
    )


# ---------------------------------------------------------------------------
# Deterministic speaker assignment
# ---------------------------------------------------------------------------


def _rank_key(seed: int, speaker_id: str) -> str:
    return hashlib.sha256(f"{seed}|{speaker_id}".encode()).hexdigest()


def n_reserved_speakers(n_speakers: int, threshold_fraction: float) -> int:
    """How many speakers the ``threshold`` role gets, rounded half up.

    Half-up rounding (``floor(x + 0.5)``) is used rather than :func:`round`,
    whose banker's rounding would send a fraction of exactly ``.5`` speakers up
    or down depending on parity. The count is clamped to ``[1, n - 1]`` so
    neither role is ever empty; a clamp that moves the count away from the
    requested fraction is visible in the record's achieved fraction.
    """

    if n_speakers < 2:
        raise ValueError(
            f"a two-role reservation needs at least 2 speakers, got {n_speakers}"
        )
    if not 0.0 < threshold_fraction < 1.0:
        raise ValueError(f"threshold_fraction must be in (0, 1), got {threshold_fraction}")
    reserved = math.floor(threshold_fraction * n_speakers + 0.5)
    return max(1, min(n_speakers - 1, reserved))


def assign_development_roles(
    speaker_ids: Iterable[str], spec: ReservationSpec
) -> dict[str, str]:
    """Map every development speaker to ``select`` or ``threshold``.

    Deterministic for a fixed seed and speaker set, and independent of the input
    order: speakers are ranked by ``sha256("<seed>|<speaker_id>")`` with the
    speaker ID as a total-order tie-break, and the lowest-ranked
    :func:`n_reserved_speakers` are reserved for ``threshold``. The returned
    mapping is ordered by speaker ID.
    """

    unique = sorted(set(speaker_ids))
    if not unique:
        raise ValueError("no development speakers to assign")
    invalid = [speaker for speaker in unique if not isinstance(speaker, str) or not speaker]
    if invalid:
        raise ValueError(f"speaker ids must be non-empty strings, got {invalid!r}")

    reserved_count = n_reserved_speakers(len(unique), spec.threshold_fraction)
    ranked = sorted(unique, key=lambda speaker: (_rank_key(spec.seed, speaker), speaker))
    reserved = set(ranked[:reserved_count])
    return {
        speaker: ROLE_THRESHOLD if speaker in reserved else ROLE_SELECT for speaker in unique
    }


def development_speakers(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Sorted development speaker IDs, raising when any row has no speaker.

    A development row without a speaker cannot be grouped, so it would silently
    break the reservation's whole guarantee rather than degrade it.
    """

    speakers: set[str] = set()
    anonymous = 0
    for row in rows:
        if row["official_split"] != DEVELOPMENT_SPLIT:
            continue
        speaker_id = row.get("speaker_id")
        if not speaker_id:
            anonymous += 1
            continue
        speakers.add(str(speaker_id))
    if anonymous:
        raise ValueError(
            f"{anonymous} development row(s) have no speaker_id; a speaker-grouped "
            "reservation cannot be frozen"
        )
    return sorted(speakers)


def extract_role_assignment(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Read the speaker to role mapping already written in the manifest.

    Raises when one speaker carries more than one ``development_role`` value,
    which also catches a partially written reservation (some clips assigned,
    some still null). Speakers with no role yet are omitted, so an untouched
    manifest yields an empty mapping.
    """

    roles_by_speaker: dict[str, set[str | None]] = defaultdict(set)
    for row in rows:
        if row["official_split"] != DEVELOPMENT_SPLIT:
            continue
        roles_by_speaker[str(row.get("speaker_id"))].add(row.get("development_role") or None)

    conflicted = sorted(speaker for speaker, roles in roles_by_speaker.items() if len(roles) > 1)
    if conflicted:
        raise ValueError(
            f"{len(conflicted)} speaker(s) carry more than one development_role "
            f"(e.g. {conflicted[:5]}); a speaker-grouped reservation is violated"
        )
    assignment: dict[str, str] = {}
    for speaker, roles in sorted(roles_by_speaker.items()):
        role = next(iter(roles))
        if role is not None:
            assignment[speaker] = role
    return assignment


def apply_development_roles(
    rows: Iterable[Mapping[str, Any]], assignment: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Return copies of ``rows`` with ``development_role`` set from ``assignment``.

    Development rows take their speaker's role; every other row is forced back
    to ``None``, so a role can never leak onto a train or evaluation clip. The
    assignment must name exactly the development speakers.
    """

    speakers = set(development_speakers(rows))
    unassigned = sorted(speakers - set(assignment))
    if unassigned:
        raise ValueError(f"no role assigned for development speaker(s): {unassigned}")
    foreign = sorted(set(assignment) - speakers)
    if foreign:
        raise ValueError(
            f"assignment names speaker(s) outside the development split: {foreign}"
        )
    unknown_roles = sorted({role for role in assignment.values() if role not in RESERVATION_ROLES})
    if unknown_roles:
        raise ValueError(f"roles must be one of {list(RESERVATION_ROLES)}, got {unknown_roles}")

    updated: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        new_row["development_role"] = (
            assignment[str(row["speaker_id"])]
            if row["official_split"] == DEVELOPMENT_SPLIT
            else None
        )
        updated.append(new_row)
    return updated


# ---------------------------------------------------------------------------
# Coverage checks (run before the membership may be frozen)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoleCoverage:
    """What one role actually holds, as evidence rather than a pass/fail flag."""

    role: str
    n_speakers: int
    n_clips: int
    n_bonafide: int
    n_spoof: int
    attacks: tuple[str, ...]
    missing_attacks: tuple[str, ...]
    speakers_below_min_bonafide: tuple[str, ...]
    counts_by_label_attack: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CoverageReport:
    """Per-role coverage evidence plus the list of requirement failures."""

    requirements: CoverageRequirements
    roles: tuple[RoleCoverage, ...]
    per_speaker: tuple[dict[str, Any], ...]
    failures: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True when every coverage requirement holds."""

        return not self.failures

    def as_dict(self) -> dict[str, Any]:
        """Plain, JSON-serializable form for the frozen record."""

        return asdict(self)


def _counts_by_label_attack(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    counts = Counter((row["label"], row.get("attack_id")) for row in rows)
    return tuple(
        {"label": label, "attack_id": attack, "clips": clips}
        for (label, attack), clips in sorted(
            counts.items(), key=lambda item: (item[0][0], item[0][1] or "")
        )
    )


def check_reservation_coverage(
    rows: Iterable[Mapping[str, Any]],
    assignment: Mapping[str, str],
    requirements: CoverageRequirements,
) -> CoverageReport:
    """Check that both roles can support the job they are reserved for.

    Requirements, from the architecture plan's leakage controls:

    * every ``threshold`` speaker contributes at least
      ``min_bonafide_clips_per_threshold_speaker`` bona fide clips, since only
      bona fide clips set the operating threshold;
    * when ``require_both_labels_per_role``, each role holds both classes, which
      clip-level EER and the threshold search both need;
    * each role covers every attack in ``require_attacks``.

    Returns evidence and failures instead of raising, so a failing reservation
    can be recorded and the *fraction* adjusted without changing the rule.
    """

    by_role: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_speaker: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["official_split"] != DEVELOPMENT_SPLIT:
            continue
        speaker = str(row["speaker_id"])
        role = assignment.get(speaker)
        if role is None:
            continue
        by_role[role].append(row)
        by_speaker[speaker].append(row)

    failures: list[str] = []
    coverages: list[RoleCoverage] = []
    for role in RESERVATION_ROLES:
        role_rows = by_role.get(role, [])
        role_speakers = sorted(
            speaker for speaker, assigned in assignment.items() if assigned == role
        )
        label_counts = Counter(row["label"] for row in role_rows)
        attacks = tuple(
            sorted({row["attack_id"] for row in role_rows if row.get("attack_id")})
        )
        missing = tuple(
            attack for attack in requirements.require_attacks if attack not in attacks
        )
        below_min: tuple[str, ...] = ()
        if role == ROLE_THRESHOLD:
            minimum = requirements.min_bonafide_clips_per_threshold_speaker
            below_min = tuple(
                speaker
                for speaker in role_speakers
                if sum(1 for row in by_speaker[speaker] if row["label"] == "bonafide") < minimum
            )

        if not role_rows:
            failures.append(f"role {role!r} holds no development clips")
        if requirements.require_both_labels_per_role:
            absent = sorted(label for label in VALID_LABELS if not label_counts.get(label))
            if absent:
                failures.append(f"role {role!r} has no clips for label(s) {absent}")
        if missing:
            failures.append(f"role {role!r} is missing attack(s) {list(missing)}")
        if below_min:
            failures.append(
                f"role {role!r} speaker(s) {list(below_min)} have fewer than "
                f"{requirements.min_bonafide_clips_per_threshold_speaker} bona fide clip(s)"
            )

        coverages.append(
            RoleCoverage(
                role=role,
                n_speakers=len(role_speakers),
                n_clips=len(role_rows),
                n_bonafide=label_counts.get("bonafide", 0),
                n_spoof=label_counts.get("spoof", 0),
                attacks=attacks,
                missing_attacks=missing,
                speakers_below_min_bonafide=below_min,
                counts_by_label_attack=_counts_by_label_attack(role_rows),
            )
        )

    per_speaker = tuple(
        {
            "speaker_id": speaker,
            "development_role": assignment[speaker],
            "n_clips": len(by_speaker[speaker]),
            "n_bonafide": sum(1 for row in by_speaker[speaker] if row["label"] == "bonafide"),
            "n_spoof": sum(1 for row in by_speaker[speaker] if row["label"] == "spoof"),
        }
        for speaker in sorted(assignment)
    )
    return CoverageReport(
        requirements=requirements,
        roles=tuple(coverages),
        per_speaker=per_speaker,
        failures=tuple(failures),
    )


# ---------------------------------------------------------------------------
# Planning and the frozen record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReservationPlan:
    """A derived reservation, before anything is written.

    ``already_frozen`` means the manifest already carries exactly this
    membership, so writing it would be a no-op. ``conflicts_with_existing``
    means the manifest carries a *different* membership: re-freezing it would
    move a protocol boundary that tuning may already depend on, so the caller
    must refuse by default.
    """

    spec: ReservationSpec
    assignment: dict[str, str]
    previous_assignment: dict[str, str]
    coverage: CoverageReport
    rows: list[dict[str, Any]]
    manifest_hash_before: str
    manifest_hash_after: str

    @property
    def already_frozen(self) -> bool:
        """True when the manifest already carries exactly this membership."""

        return self.previous_assignment == self.assignment

    @property
    def conflicts_with_existing(self) -> bool:
        """True when the manifest carries a different, non-empty membership."""

        return bool(self.previous_assignment) and not self.already_frozen

    def speakers_by_role(self) -> dict[str, list[str]]:
        """Sorted speaker IDs per role (the committed membership)."""

        return {
            role: sorted(
                speaker for speaker, assigned in self.assignment.items() if assigned == role
            )
            for role in RESERVATION_ROLES
        }


def plan_reservation(
    rows: Iterable[Mapping[str, Any]], spec: ReservationSpec
) -> ReservationPlan:
    """Derive the reservation for ``rows`` without writing anything.

    Computes the manifest hash before and after the roles are applied; they
    differ because ``compute_manifest_hash`` covers ``development_role`` by
    design, so a feature cache or run record can never cite a manifest whose
    reservation has changed.
    """

    source_rows = [dict(row) for row in rows]
    previous = extract_role_assignment(source_rows)
    assignment = assign_development_roles(development_speakers(source_rows), spec)
    updated = apply_development_roles(source_rows, assignment)
    return ReservationPlan(
        spec=spec,
        assignment=assignment,
        previous_assignment=previous,
        coverage=check_reservation_coverage(updated, assignment, spec.coverage),
        rows=updated,
        manifest_hash_before=compute_manifest_hash(source_rows),
        manifest_hash_after=compute_manifest_hash(updated),
    )


def build_reservation_record(
    plan: ReservationPlan, *, note: str | None = None
) -> dict[str, Any]:
    """Build the small, committed record that freezes the membership.

    This record -- not the manifest, which is ignored by Git -- is the
    authoritative membership. It holds speaker IDs only: no paths, no clip IDs
    and no audio.
    """

    speakers = plan.speakers_by_role()
    n_speakers = len(plan.assignment)
    record: dict[str, Any] = {
        "schema_version": RESERVATION_SCHEMA_VERSION,
        "frozen_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "package_version": package_version(),
        "spec": plan.spec.as_dict(),
        "speakers": speakers,
        "n_speakers": {role: len(ids) for role, ids in speakers.items()},
        "achieved_threshold_fraction": (
            len(speakers[ROLE_THRESHOLD]) / n_speakers if n_speakers else None
        ),
        "manifest_hash_before": plan.manifest_hash_before,
        "manifest_hash_after": plan.manifest_hash_after,
        "coverage": plan.coverage.as_dict(),
        "coverage_ok": plan.coverage.ok,
        "disclosure": (
            f"{len(speakers[ROLE_THRESHOLD])} reserved threshold speaker(s): the operating "
            "threshold rests on few speakers, so its achieved false-positive rate holds on "
            "this reservation and is not a guarantee on other data. The evaluation partition "
            "is never consulted to improve it."
        ),
    }
    if note:
        record["note"] = note
    return record


def verify_frozen_reservation(
    rows: Iterable[Mapping[str, Any]], record: Mapping[str, Any]
) -> None:
    """Raise when a manifest disagrees with the committed frozen record.

    Later slices (features, the LR report, the locked evaluation) call this to
    prove they are reading the manifest whose reservation was frozen, rather
    than trusting a configuration flag.
    """

    manifest_rows = [dict(row) for row in rows]
    expected: dict[str, str] = {}
    speakers = record.get("speakers")
    if not isinstance(speakers, Mapping):
        raise ValueError("frozen record has no 'speakers' mapping")
    for role, ids in speakers.items():
        if role not in RESERVATION_ROLES:
            raise ValueError(f"frozen record names unknown role {role!r}")
        for speaker in ids:
            expected[str(speaker)] = role

    actual = extract_role_assignment(manifest_rows)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        moved = sorted(
            speaker
            for speaker in set(expected) & set(actual)
            if expected[speaker] != actual[speaker]
        )
        raise ValueError(
            "manifest reservation does not match the frozen record "
            f"(missing={missing}, unexpected={unexpected}, moved={moved})"
        )

    recorded_hash = record.get("manifest_hash_after")
    actual_hash = compute_manifest_hash(manifest_rows)
    if recorded_hash != actual_hash:
        raise ValueError(
            f"manifest hash {actual_hash} does not match the frozen record's "
            f"{recorded_hash}; the manifest changed after the reservation was frozen"
        )
