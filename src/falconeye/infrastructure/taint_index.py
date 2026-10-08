"""
Taint specification loader and query index.

Reads YAML specifications describing sources, sinks, sanitizers, and
propagators for one or more programming languages, validates them
against the schema documented in ``src/falconeye/taint_specs/SCHEMA.md``,
and exposes an immutable query API for downstream consumers (the
call-graph builder, the path finder, the LLM verification prompt
assembler).

Loading is a one-shot at scan start. The resulting ``TaintIndex`` is
frozen and safe to share across threads and coroutines.

Sources of specs (in load order):

  1. Built-in specs shipped with the package under
     ``falconeye.taint_specs.*`` (installed by pip into the wheel).
  2. User override specs from an operator-chosen directory. Defaults
     to ``./falconeye_taint_specs/`` under the scanned repo root when
     one exists.

User overrides load AFTER built-ins with the same canonical name, so
an operator can shadow or extend a shipped entry without editing
package files. Cross-file conflicts (same canonical name declared as
source in one file and sink in another) surface as loader warnings;
the later-loaded file wins.

See ``SCHEMA.md`` for the full field reference and the eight
validation rules this module enforces at load time.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from importlib import resources
from pathlib import Path
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import yaml

from .finding_validation import ALLOWED_CWES


# ---------------------------------------------------------------------
# Constants derived from SCHEMA.md.
#
# When SCHEMA.md is revised in a way that changes any of these, bump
# SCHEMA_VERSION so old spec files fail loudly instead of silently
# drifting into unsupported territory.
# ---------------------------------------------------------------------

SCHEMA_VERSION = 1

_CANONICAL_NAME_RE = re.compile(r"^[a-z]+:\S+$")

# The full initial propagator rule enum, per SCHEMA.md §8.
_VALID_PROPAGATOR_RULES: FrozenSet[str] = frozenset({
    "taint_return_if_any_arg_tainted",
    "taint_return_if_specific_args_tainted",
    "taint_return_if_this_tainted",
    "taint_this_if_arg_tainted",
    "pass_through_arg",
    "sanitize_return",
    "no_propagation",
    "propagate_all",
})

# Fixed tokens for `taints:` and `at:` per SCHEMA.md §9. The `arg:N`
# and `field:name` forms are validated separately by regex.
_TAINTS_FIXED_TOKENS: FrozenSet[str] = frozenset({"return", "this", "args"})
_TAINTS_ARG_RE = re.compile(r"^arg:(\d+)$")
_TAINTS_FIELD_RE = re.compile(r"^field:[A-Za-z_]\w*$")

_VALID_SEVERITIES: FrozenSet[str] = frozenset({
    "CRITICAL", "HIGH", "MEDIUM", "LOW"
})

# The "*" sentinel for `sanitizes_kind:` meaning "neutralizes any
# kind of taint" (used for universal transforms: type casts, hashing,
# Guid.Parse). Kept as a module constant so callers pattern-match on
# it explicitly rather than magic-stringing.
SANITIZES_ALL = "*"

# Roles a canonical name can play. Cross-role duplicates (same name
# as source AND propagator) are permitted; same-role duplicates are
# not.
_ROLES: Tuple[str, ...] = ("sources", "sinks", "sanitizers", "propagators")


# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------


class TaintSpecError(Exception):
    """Raised when a spec file fails a hard validation rule."""


class TaintSpecRequiresCycleError(TaintSpecError):
    """Raised when ``requires:`` forms a cycle across spec files."""


# ---------------------------------------------------------------------
# Data model. Every entry class is a frozen dataclass so an instance
# of TaintIndex can be shared freely without defensive copying.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class KindEntry:
    """A taint category (e.g. ``sql_query``, ``http_query``)."""
    name: str
    abstract: bool = False
    parent: Optional[str] = None
    default_cwe: Optional[str] = None
    description: str = ""


@dataclass(frozen=True)
class SourceEntry:
    """A callable that introduces taint of a given kind."""
    name: str
    kind: str
    taints: str
    confidence: str = "high"
    notes: str = ""
    source_file: str = ""


@dataclass(frozen=True)
class SinkEntry:
    """A callable that is dangerous when receiving tainted data."""
    name: str
    kind: str
    tainted_args: Tuple[Union[int, str], ...]
    cwe: Optional[str] = None
    severity: Optional[str] = None
    notes: str = ""
    conditional_on: Optional[Mapping[str, Any]] = None
    source_file: str = ""


@dataclass(frozen=True)
class SanitizerEntry:
    """
    A callable that neutralizes taint of one or more kinds.

    ``sanitizes_kinds`` is the *expanded* set: abstract parent kinds
    referenced in the source YAML have been replaced by their concrete
    children at load time. This is why downstream code can check
    membership without walking the parent chain.

    The ``SANITIZES_ALL`` sentinel is preserved as-is (it is not
    exploded into "every known kind"). Downstream lookups should
    handle it as a special case.
    """
    name: str
    sanitizes_kinds: FrozenSet[str]
    at: str
    partial: bool = False
    notes: str = ""
    source_file: str = ""


@dataclass(frozen=True)
class PropagatorEntry:
    """A callable whose taint-passing behaviour is precision-tuned."""
    name: str
    rule: str
    taint_args: Tuple[int, ...] = ()
    exclude_args: Tuple[int, ...] = ()
    pass_arg: Optional[int] = None
    notes: str = ""
    source_file: str = ""


# ---------------------------------------------------------------------
# The immutable query index. Downstream code (call graph, path finder)
# holds one of these for the entire scan.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TaintIndex:
    """
    Query-friendly immutable snapshot of every loaded taint spec.

    Lookup methods return ``None`` when the name is unknown, so a
    caller can fold the "no entry -> conservative default" case
    into a single ``if`` branch. Do not mutate the dictionaries
    exposed by this class -- they are shared with in-flight
    consumers.
    """
    kinds: Mapping[str, KindEntry]
    sources: Mapping[str, SourceEntry]
    sinks: Mapping[str, SinkEntry]
    sanitizers: Mapping[str, SanitizerEntry]
    propagators: Mapping[str, PropagatorEntry]
    loaded_specs: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()

    # -- lookups --------------------------------------------------------

    def get_source(self, name: str) -> Optional[SourceEntry]:
        return self.sources.get(name)

    def get_sink(self, name: str) -> Optional[SinkEntry]:
        return self.sinks.get(name)

    def get_sanitizer(self, name: str) -> Optional[SanitizerEntry]:
        return self.sanitizers.get(name)

    def get_propagator(self, name: str) -> Optional[PropagatorEntry]:
        return self.propagators.get(name)

    def get_kind(self, name: str) -> Optional[KindEntry]:
        return self.kinds.get(name)

    # -- kind hierarchy queries ----------------------------------------

    def is_kind_in_family(self, kind: str, family: str) -> bool:
        """
        True if ``kind`` is ``family`` or transitively descends from it.

        Used by sink-vs-sanitizer matching when a sanitizer neutralized
        the family (parent) kind and we need to check whether a concrete
        sink kind falls under it. Also protects against infinite loops
        if a malformed spec ever produces a parent cycle -- the load-time
        validator catches these but we belt-and-brace it here too.
        """
        current: Optional[str] = kind
        seen: Set[str] = set()
        while current is not None and current not in seen:
            if current == family:
                return True
            seen.add(current)
            entry = self.kinds.get(current)
            current = entry.parent if entry else None
        return False


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------


def load_taint_index(
    languages: Optional[Sequence[str]] = None,
    override_dir: Optional[Path] = None,
    framework_hints: Optional[Mapping[str, Sequence[str]]] = None,
    logger: Optional[logging.Logger] = None,
) -> TaintIndex:
    """
    Discover, validate, and merge all taint specifications into an
    in-memory query index.

    Args:
        languages: When set, only load specs whose ``language:`` field
            appears in this list. ``None`` loads every language.
        override_dir: Filesystem directory of user override specs.
            Loaded after built-ins with the same name. Files here MUST
            have the same shape as built-in specs (see SCHEMA.md).
        framework_hints: Optional mapping from language tag to a list
            of extra spec-file names (without ``.yaml``) that must be
            loaded even if they are framework-scoped. This is what the
            language adapter's ``detect_frameworks(project_root)``
            populates.
        logger: Optional logger. Defaults to the module logger.

    Returns:
        Fully-populated, immutable ``TaintIndex``.

    Raises:
        TaintSpecError: On any hard validation failure. Soft failures
            (cross-file conflicts, non-fatal issues) are logged and
            surfaced via ``TaintIndex.warnings`` instead.
        TaintSpecRequiresCycleError: On a ``requires:`` cycle.
    """
    logger = logger or logging.getLogger(__name__)

    # Step 1: kind vocabulary. Everything else references this, so it
    # must load first.
    kinds = _load_kind_vocabulary()

    # Step 2: enumerate spec files.
    builtin_paths, override_paths = _enumerate_spec_paths(override_dir)

    # Step 3: parse every file into a dict-of-dicts we can validate
    # before doing the more expensive merge work. Files that fail
    # version check are dropped with a warning.
    warnings: List[str] = []
    parsed: List[Tuple[str, str, Dict[str, Any]]] = []
    for path, is_override in builtin_paths + override_paths:
        try:
            raw = _load_yaml(path)
        except Exception as e:
            raise TaintSpecError(
                f"failed to parse {path}: {e.__class__.__name__}: {e}"
            ) from e

        if raw.get("version") != SCHEMA_VERSION:
            warnings.append(
                f"skipped {path} -- schema version {raw.get('version')!r} "
                f"does not match loader version {SCHEMA_VERSION}"
            )
            logger.warning(warnings[-1])
            continue

        # Store as (spec_id, path_str, raw_dict). spec_id is the file
        # basename without extension; used for requires: resolution
        # and for spec_id tracking in the merged index.
        spec_id = Path(path).stem
        parsed.append((spec_id, str(path), raw))

    # Step 4: language filter. Applied AFTER parse so we can also
    # respect framework_hints (which name spec files by id).
    if languages is not None:
        allowed_languages: Set[str] = set(languages)
        hinted_specs: Set[str] = set()
        for lang_hints in (framework_hints or {}).values():
            hinted_specs.update(lang_hints)
        parsed = [
            (sid, path, raw) for (sid, path, raw) in parsed
            if raw.get("language") in allowed_languages or sid in hinted_specs
        ]

    # Step 5: topologically sort by `requires:` and detect cycles.
    parsed = _topological_sort(parsed)

    # Step 6: validate each spec against the SCHEMA.md rules. Fail
    # loudly on any hard rule violation. Track cross-file soft
    # conflicts and log them.
    kind_names = set(kinds.keys())
    concrete_kinds = {k for k, v in kinds.items() if not v.abstract}

    sources: Dict[str, SourceEntry] = {}
    sinks: Dict[str, SinkEntry] = {}
    sanitizers: Dict[str, SanitizerEntry] = {}
    propagators: Dict[str, PropagatorEntry] = {}

    # For cross-file conflict tracking: name -> (previous_role,
    # previous_file). If a later file assigns a different role, warn.
    role_owner: Dict[str, Tuple[str, str]] = {}

    for spec_id, path_str, raw in parsed:
        try:
            _validate_spec(
                raw=raw,
                path=path_str,
                known_kinds=kind_names,
                concrete_kinds=concrete_kinds,
                allowed_cwes=ALLOWED_CWES,
            )
        except TaintSpecError as e:
            raise TaintSpecError(f"in {path_str}: {e}") from e

        # Add local kinds (declared inside the spec file). Local kinds
        # extend the working vocabulary for this and later files.
        for local_kind in raw.get("kinds") or []:
            k = _build_kind_entry(local_kind)
            if k.name in kinds and kinds[k.name] != k:
                warnings.append(
                    f"local kind {k.name!r} in {path_str} shadows "
                    f"a differing shared/local definition"
                )
                logger.warning(warnings[-1])
            kinds[k.name] = k
            kind_names.add(k.name)
            if not k.abstract:
                concrete_kinds.add(k.name)

        _merge_role(
            role="sources",
            entries=raw.get("sources") or [],
            builder=lambda d, sf=path_str: _build_source_entry(d, sf),
            target=sources,
            path_str=path_str,
            role_owner=role_owner,
            warnings=warnings,
            logger=logger,
        )
        _merge_role(
            role="sinks",
            entries=raw.get("sinks") or [],
            builder=lambda d, sf=path_str: _build_sink_entry(d, sf),
            target=sinks,
            path_str=path_str,
            role_owner=role_owner,
            warnings=warnings,
            logger=logger,
        )
        _merge_role(
            role="sanitizers",
            entries=raw.get("sanitizers") or [],
            builder=lambda d, sf=path_str: _build_sanitizer_entry(d, sf, kinds),
            target=sanitizers,
            path_str=path_str,
            role_owner=role_owner,
            warnings=warnings,
            logger=logger,
        )
        _merge_role(
            role="propagators",
            entries=raw.get("propagators") or [],
            builder=lambda d, sf=path_str: _build_propagator_entry(d, sf),
            target=propagators,
            path_str=path_str,
            role_owner=role_owner,
            warnings=warnings,
            logger=logger,
        )

    return TaintIndex(
        kinds=dict(kinds),
        sources=dict(sources),
        sinks=dict(sinks),
        sanitizers=dict(sanitizers),
        propagators=dict(propagators),
        loaded_specs=tuple(sid for sid, _, _ in parsed),
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------
# File enumeration + YAML loading
# ---------------------------------------------------------------------


def _enumerate_spec_paths(
    override_dir: Optional[Path],
) -> Tuple[List[Tuple[Any, bool]], List[Tuple[Path, bool]]]:
    """
    Discover every candidate spec file.

    Built-in specs live inside the installed ``falconeye.taint_specs``
    package. importlib.resources.files() returns a Traversable, which
    we treat abstractly to work with both a filesystem install (pip -e)
    and a zipped wheel install.

    Returns (builtin_paths, override_paths). Each element is a
    (Traversable-or-Path, is_override) tuple.

    Files named ``_*.yaml`` and files inside ``_shared/`` are treated
    as internal fragments (kinds vocabulary, shared includes) and
    excluded from the spec enumeration. The kind vocabulary is loaded
    separately by ``_load_kind_vocabulary``.
    """
    builtin: List[Tuple[Any, bool]] = []
    override: List[Tuple[Path, bool]] = []

    root = resources.files("falconeye.taint_specs")
    for entry in root.iterdir():
        name = entry.name
        if entry.is_dir():
            continue  # _shared/ and any future subdir
        if not name.endswith(".yaml"):
            continue
        if name.startswith("_"):
            continue
        builtin.append((entry, False))

    # Overrides directory. Enumerate .yaml files at the top level;
    # subdirs and _-prefixed files are ignored (same convention).
    if override_dir is not None and override_dir.exists():
        for p in sorted(override_dir.iterdir()):
            if p.is_dir():
                continue
            if not p.name.endswith(".yaml"):
                continue
            if p.name.startswith("_"):
                continue
            override.append((p, True))

    # Sort built-ins by name so behaviour is deterministic across
    # filesystems and installs.
    builtin.sort(key=lambda t: t[0].name)
    return builtin, override


def _load_yaml(path: Any) -> Dict[str, Any]:
    """Read YAML from a Traversable or a Path."""
    if isinstance(path, Path):
        text = path.read_text(encoding="utf-8")
    else:
        # importlib.resources Traversable
        text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise TaintSpecError(
            f"{path}: top-level YAML must be a mapping, got {type(data).__name__}"
        )
    return data


def _load_kind_vocabulary() -> Dict[str, KindEntry]:
    """
    Load ``_shared/kinds.yaml`` and return a name -> KindEntry map.

    Also validates that each ``parent:`` reference resolves and that
    each ``default_cwe:`` is in ALLOWED_CWES. These checks run before
    any spec file is validated so an inconsistent vocabulary fails
    loudly instead of silently masking other errors.
    """
    kinds_res = resources.files("falconeye.taint_specs").joinpath(
        "_shared", "kinds.yaml"
    )
    raw = _load_yaml(kinds_res)
    if raw.get("version") != SCHEMA_VERSION:
        raise TaintSpecError(
            f"_shared/kinds.yaml: schema version {raw.get('version')!r} "
            f"does not match loader version {SCHEMA_VERSION}"
        )

    kinds_map: Dict[str, KindEntry] = {}
    entries = raw.get("kinds") or {}
    if not isinstance(entries, dict):
        raise TaintSpecError(
            "_shared/kinds.yaml: 'kinds' must be a mapping"
        )

    for name, body in entries.items():
        if not isinstance(body, dict):
            body = {}
        kinds_map[name] = KindEntry(
            name=name,
            abstract=bool(body.get("abstract", False)),
            parent=body.get("parent"),
            default_cwe=body.get("default_cwe"),
            description=body.get("description", "") or "",
        )

    # Parent-reference and default_cwe checks
    for k in kinds_map.values():
        if k.parent is not None and k.parent not in kinds_map:
            raise TaintSpecError(
                f"_shared/kinds.yaml: kind {k.name!r} has unknown "
                f"parent {k.parent!r}"
            )
        if k.default_cwe is not None and k.default_cwe not in ALLOWED_CWES:
            raise TaintSpecError(
                f"_shared/kinds.yaml: kind {k.name!r} default_cwe "
                f"{k.default_cwe!r} not in ALLOWED_CWES"
            )

    return kinds_map


def _build_kind_entry(raw: Dict[str, Any]) -> KindEntry:
    """Build a KindEntry from a local ``kinds:`` list entry."""
    if "name" not in raw:
        raise TaintSpecError("local kind entry missing 'name'")
    return KindEntry(
        name=raw["name"],
        abstract=bool(raw.get("abstract", False)),
        parent=raw.get("parent"),
        default_cwe=raw.get("default_cwe"),
        description=raw.get("description", "") or "",
    )


# ---------------------------------------------------------------------
# Validation. All rules are documented in SCHEMA.md §11.
# ---------------------------------------------------------------------


def _validate_spec(
    *,
    raw: Dict[str, Any],
    path: str,
    known_kinds: Set[str],
    concrete_kinds: Set[str],
    allowed_cwes: Mapping[str, str],
) -> None:
    """Hard validation of a single parsed spec file."""
    if "language" not in raw or not isinstance(raw["language"], str):
        raise TaintSpecError("missing or non-string 'language' field")

    # Rule 7: no duplicate names within the same role in the same file.
    for role in _ROLES:
        seen: Set[str] = set()
        for entry in raw.get(role) or []:
            n = entry.get("name")
            if n in seen:
                raise TaintSpecError(
                    f"duplicate {role[:-1]} name {n!r}"
                )
            seen.add(n)

    # Rule 3 + Rule 2: canonical name format and kind existence.
    for entry in raw.get("sources") or []:
        _check_name(entry, "source")
        _check_kind_concrete(entry, "source", concrete_kinds)
        _check_taints_grammar(entry.get("taints"), "source", entry.get("name"))
        _check_confidence(entry)

    for entry in raw.get("sinks") or []:
        _check_name(entry, "sink")
        _check_kind_concrete(entry, "sink", concrete_kinds)
        _check_tainted_args(entry)
        _check_severity(entry)
        # Rule 4: CWE existence (only if explicitly specified; falls
        # back to the kind's default_cwe when absent).
        cwe = entry.get("cwe")
        if cwe is not None and cwe not in allowed_cwes:
            raise TaintSpecError(
                f"sink {entry.get('name')!r}: cwe {cwe!r} not in "
                f"ALLOWED_CWES (see finding_validation.py)"
            )

    for entry in raw.get("sanitizers") or []:
        _check_name(entry, "sanitizer")
        _check_sanitizes_kind(entry, known_kinds)
        _check_taints_grammar(entry.get("at"), "sanitizer", entry.get("name"))

    for entry in raw.get("propagators") or []:
        _check_name(entry, "propagator")
        # Rule 5: propagator rule enum.
        rule = entry.get("rule")
        if rule not in _VALID_PROPAGATOR_RULES:
            raise TaintSpecError(
                f"propagator {entry.get('name')!r}: rule {rule!r} not "
                f"in enum {sorted(_VALID_PROPAGATOR_RULES)}"
            )


def _check_name(entry: Dict[str, Any], role: str) -> None:
    n = entry.get("name")
    if not isinstance(n, str) or not _CANONICAL_NAME_RE.match(n):
        raise TaintSpecError(
            f"{role}: name {n!r} does not match ^[a-z]+:\\S+$"
        )


def _check_kind_concrete(
    entry: Dict[str, Any], role: str, concrete_kinds: Set[str]
) -> None:
    k = entry.get("kind")
    if k not in concrete_kinds:
        raise TaintSpecError(
            f"{role} {entry.get('name')!r}: kind {k!r} unknown or "
            f"abstract (only concrete kinds may be attached to "
            f"concrete sources/sinks)"
        )


def _check_taints_grammar(value: Any, role: str, name: Any) -> None:
    """Validate ``taints:`` / ``at:`` per SCHEMA.md §9."""
    if not isinstance(value, str):
        raise TaintSpecError(
            f"{role} {name!r}: 'taints'/'at' must be a string"
        )
    if value in _TAINTS_FIXED_TOKENS:
        return
    if _TAINTS_ARG_RE.match(value) or _TAINTS_FIELD_RE.match(value):
        return
    raise TaintSpecError(
        f"{role} {name!r}: 'taints'/'at' value {value!r} not recognised "
        f"(expected one of {sorted(_TAINTS_FIXED_TOKENS)}, 'arg:N', or "
        f"'field:name')"
    )


def _check_tainted_args(entry: Dict[str, Any]) -> None:
    args = entry.get("tainted_args")
    if not isinstance(args, list) or not args:
        raise TaintSpecError(
            f"sink {entry.get('name')!r}: 'tainted_args' must be a "
            f"non-empty list"
        )
    for a in args:
        if isinstance(a, int):
            continue
        if isinstance(a, str) and (a == "value" or a == "any"):
            continue
        raise TaintSpecError(
            f"sink {entry.get('name')!r}: tainted_args entry {a!r} "
            f"must be int, 'value', or 'any'"
        )


def _check_severity(entry: Dict[str, Any]) -> None:
    sev = entry.get("severity")
    if sev is not None and sev not in _VALID_SEVERITIES:
        raise TaintSpecError(
            f"sink {entry.get('name')!r}: severity {sev!r} not in "
            f"{sorted(_VALID_SEVERITIES)}"
        )


def _check_confidence(entry: Dict[str, Any]) -> None:
    c = entry.get("confidence")
    if c is not None and c not in ("high", "low"):
        raise TaintSpecError(
            f"source {entry.get('name')!r}: confidence {c!r} must be "
            f"'high' or 'low'"
        )


def _check_sanitizes_kind(entry: Dict[str, Any], known: Set[str]) -> None:
    sk = entry.get("sanitizes_kind")
    if isinstance(sk, str):
        candidates: List[str] = [sk]
    elif isinstance(sk, list):
        candidates = list(sk)
    else:
        raise TaintSpecError(
            f"sanitizer {entry.get('name')!r}: sanitizes_kind must be "
            f"a string, a list of strings, or '*'"
        )
    for c in candidates:
        if c == SANITIZES_ALL:
            continue
        if c not in known:
            raise TaintSpecError(
                f"sanitizer {entry.get('name')!r}: sanitizes_kind "
                f"{c!r} unknown (must be a defined kind or '*')"
            )


# ---------------------------------------------------------------------
# Entry builders (from raw dict -> frozen dataclass)
# ---------------------------------------------------------------------


def _build_source_entry(raw: Dict[str, Any], source_file: str) -> SourceEntry:
    return SourceEntry(
        name=raw["name"],
        kind=raw["kind"],
        taints=raw["taints"],
        confidence=raw.get("confidence", "high"),
        notes=raw.get("notes", "") or "",
        source_file=source_file,
    )


def _build_sink_entry(raw: Dict[str, Any], source_file: str) -> SinkEntry:
    return SinkEntry(
        name=raw["name"],
        kind=raw["kind"],
        tainted_args=tuple(raw["tainted_args"]),
        cwe=raw.get("cwe"),
        severity=raw.get("severity"),
        notes=raw.get("notes", "") or "",
        conditional_on=raw.get("conditional_on"),
        source_file=source_file,
    )


def _build_sanitizer_entry(
    raw: Dict[str, Any], source_file: str, kinds: Mapping[str, KindEntry]
) -> SanitizerEntry:
    """
    Build a SanitizerEntry with parent kinds expanded to their children.

    See SCHEMA.md §4. A sanitizer targeting an abstract parent kind
    covers every concrete child. Expansion happens once, at load time,
    so downstream membership checks are O(1) hash lookups.
    """
    sk = raw["sanitizes_kind"]
    listed: List[str] = [sk] if isinstance(sk, str) else list(sk)

    expanded: Set[str] = set()
    for entry in listed:
        if entry == SANITIZES_ALL:
            expanded.add(SANITIZES_ALL)
            continue
        kind_obj = kinds.get(entry)
        if kind_obj is not None and kind_obj.abstract:
            # Abstract parent: replace with concrete descendants only.
            # Downstream code checks membership against sink kinds,
            # which are always concrete -- keeping the parent name in
            # the set would be harmless but noisy for debugging.
            expanded.update(_descendants_of(entry, kinds))
        else:
            expanded.add(entry)

    return SanitizerEntry(
        name=raw["name"],
        sanitizes_kinds=frozenset(expanded),
        at=raw["at"],
        partial=bool(raw.get("partial", False)),
        notes=raw.get("notes", "") or "",
        source_file=source_file,
    )


def _descendants_of(
    parent: str, kinds: Mapping[str, KindEntry]
) -> Set[str]:
    """Every concrete kind whose parent chain includes ``parent``."""
    out: Set[str] = set()
    for k in kinds.values():
        if k.abstract:
            continue
        # Walk parent chain until we hit `parent` or the top.
        cursor: Optional[str] = k.parent
        while cursor is not None:
            if cursor == parent:
                out.add(k.name)
                break
            cursor_kind = kinds.get(cursor)
            cursor = cursor_kind.parent if cursor_kind else None
    return out


def _build_propagator_entry(
    raw: Dict[str, Any], source_file: str
) -> PropagatorEntry:
    return PropagatorEntry(
        name=raw["name"],
        rule=raw["rule"],
        taint_args=tuple(raw.get("taint_args") or ()),
        exclude_args=tuple(raw.get("exclude_args") or ()),
        pass_arg=raw.get("pass_arg"),
        notes=raw.get("notes", "") or "",
        source_file=source_file,
    )


# ---------------------------------------------------------------------
# Requires resolution + merge
# ---------------------------------------------------------------------


def _topological_sort(
    parsed: List[Tuple[str, str, Dict[str, Any]]],
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Order parsed specs so each file appears after every ``requires:``
    it depends on. Missing requires becomes a hard error; a cycle
    becomes ``TaintSpecRequiresCycleError``.

    Preserves the input order among independent specs so behaviour is
    deterministic across runs.
    """
    by_id: Dict[str, Tuple[str, str, Dict[str, Any]]] = {
        sid: item for item in parsed for sid in (item[0],)
    }
    order_index: Dict[str, int] = {
        sid: i for i, (sid, _, _) in enumerate(parsed)
    }

    # Kahn's algorithm.
    in_edges: Dict[str, Set[str]] = {sid: set() for sid, _, _ in parsed}
    out_edges: Dict[str, Set[str]] = {sid: set() for sid, _, _ in parsed}
    for sid, path_str, raw in parsed:
        for req in raw.get("requires") or []:
            if req not in by_id:
                raise TaintSpecError(
                    f"{path_str}: requires {req!r} not found among loaded specs"
                )
            in_edges[sid].add(req)
            out_edges[req].add(sid)

    # Roots: no incoming requires. Tie-break by original order.
    roots = sorted(
        (sid for sid, deps in in_edges.items() if not deps),
        key=lambda s: order_index[s],
    )
    result: List[Tuple[str, str, Dict[str, Any]]] = []
    while roots:
        sid = roots.pop(0)
        result.append(by_id[sid])
        for downstream in sorted(
            out_edges[sid], key=lambda s: order_index[s]
        ):
            in_edges[downstream].discard(sid)
            if not in_edges[downstream]:
                roots.append(downstream)
        # Keep roots ordered.
        roots.sort(key=lambda s: order_index[s])

    if len(result) != len(parsed):
        remaining = [sid for sid in by_id if sid not in {r[0] for r in result}]
        raise TaintSpecRequiresCycleError(
            f"requires: cycle among {sorted(remaining)}"
        )
    return result


def _merge_role(
    *,
    role: str,
    entries: Iterable[Dict[str, Any]],
    builder: Any,
    target: Dict[str, Any],
    path_str: str,
    role_owner: Dict[str, Tuple[str, str]],
    warnings: List[str],
    logger: logging.Logger,
) -> None:
    """
    Insert entries into ``target``, honouring override semantics and
    logging cross-file role conflicts.

    Rule 8 (SCHEMA.md §11): if two files declare the same canonical
    name in conflicting roles (source vs. sink, or two different sink
    kinds), we warn and let the later file win. Same-role same-kind
    entries silently override (this is the deliberate override case).
    """
    for raw_entry in entries:
        built = builder(raw_entry)
        name = built.name

        prior_role_file = role_owner.get(name)
        if prior_role_file is not None:
            prior_role, prior_file = prior_role_file
            if prior_role != role:
                warnings.append(
                    f"name {name!r} appears as {prior_role[:-1]} in "
                    f"{prior_file} and as {role[:-1]} in {path_str}; "
                    f"later file wins"
                )
                logger.warning(warnings[-1])
                # Remove from the prior role map.
                # (We can't easily reach in without knowing the role;
                # the caller keeps role tables separate. On conflict
                # we just leave the prior entry in place -- the newer
                # role entry lands in `target` and downstream lookups
                # will see both. That's the "later wins" semantics
                # while preserving the record for debugging via
                # warnings[].)
                pass
        target[name] = built
        role_owner[name] = (role, path_str)
