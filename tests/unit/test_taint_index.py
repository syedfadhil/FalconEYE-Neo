"""Unit tests for the taint specification loader and query index."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from falconeye.infrastructure.taint_index import (
    KindEntry,
    PropagatorEntry,
    SanitizerEntry,
    SinkEntry,
    SourceEntry,
    SANITIZES_ALL,
    TaintIndex,
    TaintSpecError,
    TaintSpecRequiresCycleError,
    load_taint_index,
)


# ---------------------------------------------------------------------
# Fixtures: minimal on-disk override specs used by most tests.
#
# Each helper writes a YAML file into a pytest tmp_path so tests do
# not depend on the state of the shipped taint_specs directory.
# ---------------------------------------------------------------------


def _write_spec(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def _make_valid_override(dir_: Path, name: str = "test_lang_extra") -> Path:
    """A well-formed override spec targeting the shipped C# vocabulary."""
    p = dir_ / f"{name}.yaml"
    _write_spec(p, """\
version: 1
language: csharp
sources:
  - name: csharp:MyLib.Untrusted.Read
    kind: file_content
    taints: return
sinks:
  - name: csharp:MyLib.Danger.Exec
    kind: command_exec
    tainted_args: [0]
    severity: CRITICAL
sanitizers:
  - name: csharp:MyLib.Safe.Clean
    sanitizes_kind: sql_query
    at: return
propagators:
  - name: csharp:MyLib.Helpers.Wrap
    rule: taint_return_if_this_tainted
""")
    return p


# ---------------------------------------------------------------------
# Happy path: shipped C# generic loads.
# ---------------------------------------------------------------------


@pytest.mark.unit
def test_load_shipped_csharp_generic():
    idx = load_taint_index(languages=["csharp"])
    assert "csharp_generic" in idx.loaded_specs
    assert not idx.warnings

    # A sample of specific canonical names we know are in the shipped spec.
    assert idx.get_sink("csharp:System.Data.SqlClient.SqlCommand.<init>") is not None
    assert idx.get_source("csharp:System.IO.File.ReadAllText") is not None
    assert idx.get_sanitizer("csharp:System.Data.SqlClient.SqlParameter.<init>") is not None
    assert idx.get_propagator("csharp:System.String.Format") is not None
    assert idx.get_propagator("csharp:System.String.Length") is not None

    # Unknown names return None (no exception).
    assert idx.get_sink("csharp:Made.Up.Name") is None


@pytest.mark.unit
def test_language_filter_excludes_other_languages(tmp_path):
    # Pretend there is a Python spec in an override dir; language
    # filter should skip it entirely.
    _write_spec(tmp_path / "python_generic.yaml", """\
version: 1
language: python
sinks:
  - name: python:os.system
    kind: command_exec
    tainted_args: [0]
""")
    idx = load_taint_index(languages=["csharp"], override_dir=tmp_path)
    assert "python_generic" not in idx.loaded_specs
    assert idx.get_sink("python:os.system") is None


# ---------------------------------------------------------------------
# Kind parent expansion
# ---------------------------------------------------------------------


@pytest.mark.unit
def test_sanitizer_targeting_abstract_parent_expands_to_children(tmp_path):
    """A sanitizer targeting `injection` covers every concrete injection kind."""
    _write_spec(tmp_path / "csharp_uni_sanitizer.yaml", """\
version: 1
language: csharp
sanitizers:
  - name: csharp:UniLib.PurgeAllInjection
    sanitizes_kind: injection
    at: return
""")
    idx = load_taint_index(languages=["csharp"], override_dir=tmp_path)
    s = idx.get_sanitizer("csharp:UniLib.PurgeAllInjection")
    assert s is not None
    # Abstract parent has been replaced by concrete children.
    assert "injection" not in s.sanitizes_kinds
    assert "sql_query" in s.sanitizes_kinds
    assert "xss" in s.sanitizes_kinds
    assert "command_exec" in s.sanitizes_kinds
    # user_input children are NOT injection children; must not appear.
    assert "http_query" not in s.sanitizes_kinds


@pytest.mark.unit
def test_sanitizer_star_kept_as_sentinel(tmp_path):
    """`sanitizes_kind: "*"` is preserved verbatim, not expanded."""
    _write_spec(tmp_path / "csharp_star.yaml", """\
version: 1
language: csharp
sanitizers:
  - name: csharp:UniLib.HashIt
    sanitizes_kind: "*"
    at: return
""")
    idx = load_taint_index(languages=["csharp"], override_dir=tmp_path)
    s = idx.get_sanitizer("csharp:UniLib.HashIt")
    assert s.sanitizes_kinds == frozenset({SANITIZES_ALL})


@pytest.mark.unit
def test_is_kind_in_family():
    idx = load_taint_index(languages=["csharp"])
    assert idx.is_kind_in_family("sql_query", "injection") is True
    assert idx.is_kind_in_family("http_query", "user_input") is True
    # cross-family
    assert idx.is_kind_in_family("sql_query", "user_input") is False
    # self match
    assert idx.is_kind_in_family("sql_query", "sql_query") is True
    # unknown kind returns False (walks off the top of the parent chain)
    assert idx.is_kind_in_family("no_such_kind", "injection") is False


# ---------------------------------------------------------------------
# Override precedence
# ---------------------------------------------------------------------


@pytest.mark.unit
def test_override_wins_over_builtin(tmp_path):
    """A user override for a canonical name already declared in a
    built-in spec replaces the built-in entry."""
    _write_spec(tmp_path / "csharp_override.yaml", """\
version: 1
language: csharp
sinks:
  - name: csharp:System.Data.SqlClient.SqlCommand.<init>
    kind: sql_query
    tainted_args: [0]
    severity: LOW
    notes: user override -- downgraded on this codebase
""")
    idx = load_taint_index(languages=["csharp"], override_dir=tmp_path)
    sink = idx.get_sink("csharp:System.Data.SqlClient.SqlCommand.<init>")
    assert sink.severity == "LOW"
    assert "user override" in sink.notes


# ---------------------------------------------------------------------
# Hard validation failures -- rules from SCHEMA.md §11
# ---------------------------------------------------------------------


@pytest.mark.unit
def test_bad_canonical_name_raises(tmp_path):
    _write_spec(tmp_path / "csharp_badname.yaml", """\
version: 1
language: csharp
sinks:
  - name: NotCanonicalAtAll
    kind: sql_query
    tainted_args: [0]
""")
    with pytest.raises(TaintSpecError, match="does not match"):
        load_taint_index(languages=["csharp"], override_dir=tmp_path)


@pytest.mark.unit
def test_unknown_kind_on_sink_raises(tmp_path):
    _write_spec(tmp_path / "csharp_badkind.yaml", """\
version: 1
language: csharp
sinks:
  - name: csharp:My.Bad.Sink
    kind: not_a_real_kind
    tainted_args: [0]
""")
    with pytest.raises(TaintSpecError, match="unknown or abstract"):
        load_taint_index(languages=["csharp"], override_dir=tmp_path)


@pytest.mark.unit
def test_abstract_kind_on_concrete_sink_raises(tmp_path):
    """Concrete sources/sinks may not use an abstract kind."""
    _write_spec(tmp_path / "csharp_abstractsink.yaml", """\
version: 1
language: csharp
sinks:
  - name: csharp:My.Sink
    kind: injection
    tainted_args: [0]
""")
    with pytest.raises(TaintSpecError, match="unknown or abstract"):
        load_taint_index(languages=["csharp"], override_dir=tmp_path)


@pytest.mark.unit
def test_bad_cwe_raises(tmp_path):
    _write_spec(tmp_path / "csharp_badcwe.yaml", """\
version: 1
language: csharp
sinks:
  - name: csharp:My.Sink
    kind: sql_query
    tainted_args: [0]
    cwe: CWE-99999
""")
    with pytest.raises(TaintSpecError, match="not in ALLOWED_CWES"):
        load_taint_index(languages=["csharp"], override_dir=tmp_path)


@pytest.mark.unit
def test_bad_propagator_rule_raises(tmp_path):
    _write_spec(tmp_path / "csharp_badprop.yaml", """\
version: 1
language: csharp
propagators:
  - name: csharp:My.Method
    rule: teleport_taint_via_wormhole
""")
    with pytest.raises(TaintSpecError, match="rule 'teleport_taint_via_wormhole' not in enum"):
        load_taint_index(languages=["csharp"], override_dir=tmp_path)


@pytest.mark.unit
def test_bad_severity_raises(tmp_path):
    _write_spec(tmp_path / "csharp_badsev.yaml", """\
version: 1
language: csharp
sinks:
  - name: csharp:My.Sink
    kind: sql_query
    tainted_args: [0]
    severity: BANANAS
""")
    with pytest.raises(TaintSpecError, match="severity 'BANANAS'"):
        load_taint_index(languages=["csharp"], override_dir=tmp_path)


@pytest.mark.unit
def test_duplicate_name_within_role_raises(tmp_path):
    _write_spec(tmp_path / "csharp_dup.yaml", """\
version: 1
language: csharp
sinks:
  - name: csharp:My.Sink
    kind: sql_query
    tainted_args: [0]
  - name: csharp:My.Sink
    kind: command_exec
    tainted_args: [0]
""")
    with pytest.raises(TaintSpecError, match="duplicate sink name"):
        load_taint_index(languages=["csharp"], override_dir=tmp_path)


@pytest.mark.unit
def test_bad_taints_grammar_raises(tmp_path):
    _write_spec(tmp_path / "csharp_badtaints.yaml", """\
version: 1
language: csharp
sources:
  - name: csharp:My.Source
    kind: file_content
    taints: wherever
""")
    with pytest.raises(TaintSpecError, match="not recognised"):
        load_taint_index(languages=["csharp"], override_dir=tmp_path)


@pytest.mark.unit
def test_taints_arg_form_accepted(tmp_path):
    """`taints: arg:2` and `taints: field:foo` are valid grammar."""
    _write_spec(tmp_path / "csharp_taintsforms.yaml", """\
version: 1
language: csharp
sources:
  - name: csharp:My.Source1
    kind: file_content
    taints: "arg:2"
  - name: csharp:My.Source2
    kind: file_content
    taints: "field:_cached_query"
""")
    idx = load_taint_index(languages=["csharp"], override_dir=tmp_path)
    assert idx.get_source("csharp:My.Source1").taints == "arg:2"
    assert idx.get_source("csharp:My.Source2").taints == "field:_cached_query"


# ---------------------------------------------------------------------
# Requires resolution
# ---------------------------------------------------------------------


@pytest.mark.unit
def test_requires_orders_dependents_after_dependencies(tmp_path):
    _write_spec(tmp_path / "csharp_base.yaml", """\
version: 1
language: csharp
sinks:
  - name: csharp:Base.Sink
    kind: sql_query
    tainted_args: [0]
""")
    _write_spec(tmp_path / "csharp_dependent.yaml", """\
version: 1
language: csharp
requires: [csharp_base]
sinks:
  - name: csharp:Dependent.Sink
    kind: sql_query
    tainted_args: [0]
""")
    idx = load_taint_index(languages=["csharp"], override_dir=tmp_path)
    order = list(idx.loaded_specs)
    assert order.index("csharp_base") < order.index("csharp_dependent")


@pytest.mark.unit
def test_requires_cycle_raises(tmp_path):
    _write_spec(tmp_path / "csharp_a.yaml", """\
version: 1
language: csharp
requires: [csharp_b]
""")
    _write_spec(tmp_path / "csharp_b.yaml", """\
version: 1
language: csharp
requires: [csharp_a]
""")
    with pytest.raises(TaintSpecRequiresCycleError, match="cycle"):
        load_taint_index(languages=["csharp"], override_dir=tmp_path)


@pytest.mark.unit
def test_requires_missing_target_raises(tmp_path):
    _write_spec(tmp_path / "csharp_orphan.yaml", """\
version: 1
language: csharp
requires: [does_not_exist]
""")
    with pytest.raises(TaintSpecError, match="requires 'does_not_exist' not found"):
        load_taint_index(languages=["csharp"], override_dir=tmp_path)


# ---------------------------------------------------------------------
# Soft failures -- surface as warnings, do not raise
# ---------------------------------------------------------------------


@pytest.mark.unit
def test_conflicting_roles_across_files_warns(tmp_path):
    """
    Same canonical name declared as source in one file and sink in
    another -- later wins, but a warning is emitted.
    """
    _write_spec(tmp_path / "aaa_first.yaml", """\
version: 1
language: csharp
sources:
  - name: csharp:Debatable.Thing
    kind: file_content
    taints: return
""")
    _write_spec(tmp_path / "bbb_second.yaml", """\
version: 1
language: csharp
sinks:
  - name: csharp:Debatable.Thing
    kind: sql_query
    tainted_args: [0]
""")
    idx = load_taint_index(languages=["csharp"], override_dir=tmp_path)
    # Should NOT raise; warning present.
    assert any("Debatable.Thing" in w for w in idx.warnings)


@pytest.mark.unit
def test_version_mismatch_skips_file_with_warning(tmp_path):
    _write_spec(tmp_path / "csharp_future.yaml", """\
version: 99
language: csharp
sinks:
  - name: csharp:Future.Sink
    kind: sql_query
    tainted_args: [0]
""")
    idx = load_taint_index(languages=["csharp"], override_dir=tmp_path)
    assert idx.get_sink("csharp:Future.Sink") is None
    assert any("schema version 99" in w for w in idx.warnings)


# ---------------------------------------------------------------------
# Framework hints
# ---------------------------------------------------------------------


@pytest.mark.unit
def test_framework_hint_loads_extra_spec_across_language_filter(tmp_path):
    """
    Framework hints are name-based. A spec listed in framework_hints
    loads even when its `language:` isn't in the `languages` filter.
    (In practice languages match anyway, but this proves the hint
    path works when the language field is missing or non-matching --
    important for optional/mixin frameworks like csharp_aspnet.)
    """
    _write_spec(tmp_path / "csharp_myfw.yaml", """\
version: 1
language: csharp
framework: myfw
sinks:
  - name: csharp:MyFW.Sink
    kind: xss
    tainted_args: [0]
""")
    idx = load_taint_index(
        languages=["csharp"],
        override_dir=tmp_path,
        framework_hints={"csharp": ["csharp_myfw"]},
    )
    assert idx.get_sink("csharp:MyFW.Sink") is not None
