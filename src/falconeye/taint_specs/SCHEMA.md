# FalconEYE taint specification format

Version: 1

This directory holds the built-in taint specifications shipped with
FalconEYE. Each spec file declares the sources, sinks, sanitizers, and
propagators for one language, or for one framework layered on top of a
language. The taint analyzer reads these specs plus a per-language
adapter and produces candidate source→sink flows for LLM verification.

This document defines the file format so a contributor can add support
for a new language or framework without reading the loader source.

- [1. Directory layout](#1-directory-layout)
- [2. Canonical names](#2-canonical-names)
- [3. Spec file structure](#3-spec-file-structure)
- [4. Kinds](#4-kinds)
- [5. Sources](#5-sources)
- [6. Sinks](#6-sinks)
- [7. Sanitizers](#7-sanitizers)
- [8. Propagators](#8-propagators)
- [9. Grammar reference](#9-grammar-reference)
- [10. Framework auto-detection](#10-framework-auto-detection)
- [11. Loader validation rules](#11-loader-validation-rules)
- [12. Adding a new language](#12-adding-a-new-language)


## 1. Directory layout

```
src/falconeye/taint_specs/
    SCHEMA.md                    # this file
    _shared/
        kinds.yaml               # canonical kind vocabulary
    csharp_generic.yaml          # stdlib: SqlClient, Process, File, ...
    csharp_aspnet.yaml           # HttpRequest.Query, controller binding
    csharp_efcore.yaml           # FromSqlRaw, DbSet<T>
    python_generic.yaml
    python_flask.yaml
    ...
```

Built-in specs ship with the wheel. Users can supply additional or
overriding specs by dropping YAML files into `./falconeye_taint_specs/`
at the scanned repository root. Override files are loaded *after*
built-ins with the same language tag, so a user file can add sinks
that the shipped spec missed. Contradictory entries (same canonical
name declared as source in one file and sink in another) surface as a
loader warning; the user file wins.

Files starting with `_` are treated as internal (kinds, shared
fragments) and are not enumerated as normal specs.


## 2. Canonical names

Every entity referenced anywhere in a spec file uses a canonical name
of the form:

```
<lang>:<fully_qualified_symbol>
```

Examples:

```
csharp:System.Data.SqlClient.SqlCommand.ExecuteReader
csharp:System.Data.SqlClient.SqlCommand.<init>
csharp:System.Web.HttpUtility.HtmlEncode
python:flask.request.args.__getitem__
python:sqlite3.Cursor.execute
java:java.sql.Statement.executeQuery
```

Conventions:

- The `<lang>:` prefix is mandatory. It permits multiple languages to
  live in the same graph and makes future cross-language flows
  representable without schema changes.
- Constructors are named `<init>`. This matches CIL / JVM bytecode
  conventions and disambiguates from the type name.
- Property accessors are named `get_X` and `set_X` for a property `X`,
  matching CIL / property-descriptor conventions across languages.
  Doc strings may refer to the underlying property name for clarity;
  the graph uses the accessor form.
- Operator overloads are named `op_Addition`, `op_Equality`, etc.,
  following the CIL naming convention. This makes the C# `+` operator
  on strings expressible as a propagator.
- Overloads share a canonical name. A spec entry for
  `csharp:System.Data.SqlClient.SqlCommand.<init>` matches every
  constructor overload; the `tainted_args` field selects which
  positional arg is the concern.

Language adapters own the mapping from raw AST callees to canonical
names, including handling of `using`, imports, base-class lookup, and
generic instantiations. If a canonical name cannot be produced (the
callee resolves to an external type not in any loaded spec), that
call site is treated as an opaque propagator by default (see §8).


## 3. Spec file structure

```yaml
version: 1                   # required; must be 1
language: csharp             # required; one canonical language tag
framework: aspnetcore        # optional; auto-loaded when detected
requires: [csharp_generic]   # optional; other specs that must be loaded

sources:     [...]           # optional; may be empty
sinks:       [...]           # optional; may be empty
sanitizers:  [...]           # optional; may be empty
propagators: [...]           # optional; may be empty
kinds:       [...]           # optional; local kind definitions
```

All sections are optional. A pure-sanitizer spec is valid; a
propagator-only precision-tuning spec is valid. Only `version` and
`language` are required.

`framework:` is a hint used by auto-detection (see §10). If absent,
the spec is treated as language-generic and loaded whenever the
language is enabled.

`requires:` lists other spec files by name (without `.yaml`) that
must be loaded before this one. The loader resolves the dependency
graph and errors on cycles. Requirement is by name, not by content,
so a spec can require `csharp_generic` even if the user overrides it.


## 4. Kinds

A *kind* is a category of taint, e.g. `sql_query` (data that will be
interpreted as SQL) or `http_query` (data pulled from an HTTP query
string). Kinds are the vocabulary that links sources to sinks and
that sanitizers target.

The canonical kind vocabulary lives in `_shared/kinds.yaml`. Each
kind has an optional parent, which lets sanitizers cover a whole
family:

```yaml
kinds:
  # source families
  user_input:      {abstract: true}
  http_query:      {parent: user_input}
  http_form:       {parent: user_input}
  file_content:    {parent: user_input}
  env_var:         {parent: user_input}
  # sink families
  injection:       {abstract: true}
  sql_query:       {parent: injection,  default_cwe: CWE-89}
  command_exec:    {parent: injection,  default_cwe: CWE-78}
  xss:             {parent: injection,  default_cwe: CWE-79}
  # standalone kinds
  deserialization: {default_cwe: CWE-502}
  ssrf:            {default_cwe: CWE-918}
```

`abstract: true` means the kind cannot be attached to a concrete
source or sink — only sanitizers may target it (to clean an entire
family). This prevents confusion between "user_input" as an umbrella
concept and specific input kinds.

`default_cwe:` is the CWE assigned to findings on this kind when a
sink entry does not override it explicitly. Values must appear in
the `ALLOWED_CWES` list in `infrastructure/finding_validation.py`.

A spec file may define local kinds under a top-level `kinds:` key.
Local kinds are namespaced to the file's language and are only
usable within that file. Use this sparingly — cross-language kinds
belong in `_shared/kinds.yaml`.


## 5. Sources

A source is a callable whose return value (or side-effect on `this`
or a specific arg) introduces taint. Format:

```yaml
sources:
  - name: csharp:System.IO.File.ReadAllText
    kind: file_content
    taints: return
    notes: >
      Entire file contents returned as a string. Path arg is not a
      source (the caller controls it) but the resulting string is
      whatever was on disk.
```

Fields:

| Field | Required | Description |
|---|---|---|
| `name` | yes | Canonical name of the source callable (see §2). |
| `kind` | yes | A concrete (non-abstract) kind from the kind vocabulary. |
| `taints` | yes | What becomes tainted. See §9. |
| `notes` | no | Human-readable rationale. Not used at analysis time. |
| `confidence` | no | `high` (default) or `low`. `low` sources produce lower-confidence flow candidates that the LLM verifier can more easily reject. |

`taints:` is one of the grammar values in §9. The most common values
for sources are `return` (return value is tainted) and `this` (the
receiver becomes tainted, e.g. `new StreamReader(userPath)`).


## 6. Sinks

A sink is a callable that mishandles tainted data if any of its
tainted args carry taint of the matching kind. Format:

```yaml
sinks:
  - name: csharp:System.Data.SqlClient.SqlCommand.<init>
    kind: sql_query
    tainted_args: [0]
    cwe: CWE-89
    severity: HIGH
    notes: >
      Constructor overload SqlCommand(string, SqlConnection). Arg 0
      is the SQL text; if tainted with sql_query, this is a classic
      SQL injection.
```

Fields:

| Field | Required | Description |
|---|---|---|
| `name` | yes | Canonical name of the sink callable. |
| `kind` | yes | The taint kind that is dangerous at this sink. |
| `tainted_args` | yes | List of positional arg indices (0-based) or named-arg strings (`value` for property setters). See §9. |
| `cwe` | no | CWE ID. Defaults to the kind's `default_cwe`. Must appear in `ALLOWED_CWES`. |
| `severity` | no | `CRITICAL`, `HIGH`, `MEDIUM`, or `LOW`. Defaults derived from CWE severity table. |
| `notes` | no | Human-readable. |
| `conditional_on` | no | Extra guard (currently only `arg_literal: {N: <value>}`). See below. |

`conditional_on:` handles sinks that are dangerous only in specific
configurations. Example: `XmlReader.Create` is only an XXE sink when
called without a hardened `XmlReaderSettings`. Currently supported:

```yaml
conditional_on:
  arg_literal: {2: null}    # dangerous when arg 2 is literal null
```

Framework specs should avoid `conditional_on` when possible — it
adds precision-cost. Prefer restructuring the sink to a more
specific canonical name if the framework provides one.


## 7. Sanitizers

A sanitizer neutralizes taint of a specific kind or family. Format:

```yaml
sanitizers:
  - name: csharp:System.Data.SqlClient.SqlParameter.<init>
    sanitizes_kind: sql_query
    at: return
    notes: >
      Standard parameterization pattern. Anything wrapped in a
      SqlParameter is safe for a downstream SqlCommand sink even
      if the underlying string was tainted.
```

Fields:

| Field | Required | Description |
|---|---|---|
| `name` | yes | Canonical name of the sanitizer callable. |
| `sanitizes_kind` | yes | Single kind, list of kinds, or `"*"` for all kinds. Parent kinds are expanded to their children. |
| `at` | yes | What gets sanitized. See §9. Most commonly `return`. |
| `notes` | no | Human-readable. |
| `partial` | no | `true` if the sanitizer reduces but does not eliminate taint. Findings passing through partial sanitizers are downgraded, not dropped. Default `false`. |

**Kind-based matching** is deliberate. A sanitizer that neutralizes
`sql_query` does NOT neutralize `xss` — an SQL parameterization
routine will not stop an XSS attack. Listing multiple kinds in
`sanitizes_kind` is the right way to declare multi-purpose
sanitizers (e.g. `HtmlEncode` sanitizes both `xss` and
`html_injection`).

`"*"` should be reserved for genuinely universal transformations —
type casts (`Convert.ToInt32`), hashing, `Guid.Parse`. Overuse of
`"*"` reintroduces precision problems the kind system was designed
to prevent.


## 8. Propagators

A propagator moves taint from one location to another without
introducing or removing it. Most methods propagate; only "interesting"
propagators need to be declared explicitly. When a call site's callee
has no propagator entry AND no source/sink/sanitizer entry, the
loader applies a **default conservative propagation rule**:

> Any tainted arg, or a tainted `this`, taints all reference-type
> outputs of the call (return value and out-params). Value-type
> outputs (int, bool, structs, enums) are not tainted.

This default favors recall over precision. Explicit propagator
entries are precision improvements — declaring `String.Length` as
`no_propagation` prevents a `return int` from being flagged.

Format:

```yaml
propagators:
  - name: csharp:System.String.Concat
    rule: taint_return_if_any_arg_tainted

  - name: csharp:System.String.Format
    rule: taint_return_if_any_arg_tainted
    exclude_args: [0]    # arg 0 is the format string, not user data

  - name: csharp:System.String.Length
    rule: no_propagation

  - name: csharp:System.Convert.ToInt32
    rule: sanitize_return
```

Fields:

| Field | Required | Description |
|---|---|---|
| `name` | yes | Canonical name of the callable. |
| `rule` | yes | One of the enum values below. |
| `taint_args` | no | List of arg indices. Used with `taint_return_if_specific_args_tainted`. |
| `exclude_args` | no | List of arg indices. Used with `taint_return_if_any_arg_tainted` to declare known-safe args. |
| `pass_arg` | no | Arg index. Used with `pass_through_arg`. |

### Propagator rule enum

The full initial set. Additions require a schema version bump.

| Rule | Meaning |
|---|---|
| `taint_return_if_any_arg_tainted` | Any tainted arg → tainted return. Use with `exclude_args` for known-safe positions (format strings, encoding parameters). |
| `taint_return_if_specific_args_tainted` | Only specific arg positions taint the return. Requires `taint_args`. |
| `taint_return_if_this_tainted` | Instance methods where the receiver's state produces the output. e.g. `StringBuilder.ToString`, `String.Substring`. |
| `taint_this_if_arg_tainted` | Mutators. Arg taints the receiver. e.g. `StringBuilder.Append`, `List.Add`. |
| `pass_through_arg` | Return value IS the input arg (identity-like). e.g. `Convert.ChangeType(x, t)` where `x` becomes the return. Requires `pass_arg`. |
| `sanitize_return` | Type cast or well-known neutralizing transform. Return is untainted regardless of input taint. Prefer sanitizer entries over this rule when the neutralization is kind-specific. |
| `no_propagation` | Return never carries taint from args or `this`. Use for value-type accessors (`String.Length`, `Array.Length`, `Count`). |
| `propagate_all` | Explicit spelling of the conservative default. Rarely needed; useful in override specs to shadow a more specific rule. |


## 9. Grammar reference

Small controlled vocabulary shared across `taints:`, `at:`, and
`tainted_args:`.

### `taints:` and `at:` values

| Value | Meaning |
|---|---|
| `return` | The method's return value. |
| `this` | The receiver object (the instance the method is called on). |
| `args` | All args become tainted. Rare; used for out-params. |
| `arg:N` | Positional arg `N` (0-based). |
| `field:name` | A named field on `this`. Use for cache-like sinks (`obj.query = tainted`). |

### `tainted_args:` values

A list. Elements are:

| Element | Meaning |
|---|---|
| Integer `N` | Positional arg `N` (0-based). |
| String `"value"` | The `value` keyword arg (used for property setters). |
| String `"any"` | Any positional arg. Use when overloads have different arities. |

### `sanitizes_kind:` values

Single kind, list of kinds, or `"*"`. Parent kinds expand to their
children at load time. Local kinds are only usable within the
declaring spec file.


## 10. Framework auto-detection

Each language adapter exposes a `detect_frameworks(project_root) ->
list[str]` method. The returned list names spec files (without
`.yaml`) that must be loaded in addition to the language-generic
spec. Example (C#):

```python
def detect_frameworks(self, project_root: Path) -> list[str]:
    specs = []
    for csproj in project_root.rglob("*.csproj"):
        text = csproj.read_text(errors="ignore")
        if "Microsoft.AspNetCore" in text:
            specs.append("csharp_aspnet")
        if "Microsoft.EntityFrameworkCore" in text:
            specs.append("csharp_efcore")
        if re.search(r"\bDapper\b", text):
            specs.append("csharp_dapper")
    return sorted(set(specs))
```

Detection is a one-shot at scan start. To disable and pin the spec
set explicitly, set `framework_autodetect: false` under the
language's per-language config block in `config.yaml` and list
`specs:` manually.

Language-generic specs (e.g. `csharp_generic.yaml`) are always
loaded when the language is enabled, regardless of detection.


## 11. Loader validation rules

The loader runs these checks at startup and refuses to start if any
fail:

1. **Schema version** — `version: 1` present at top level of every
   spec file. Files with mismatched versions are skipped with a log
   warning.

2. **Kind existence** — every `kind:` and `sanitizes_kind:` value
   resolves to a concrete kind (either in `_shared/kinds.yaml` or in
   the file's local `kinds:` section). Abstract kinds are only
   permitted in `sanitizes_kind:`.

3. **Canonical name format** — every `name:` value matches
   `^[a-z]+:\S+$`. Names must start with a known language tag.

4. **CWE existence** — every explicit `cwe:` value appears in the
   `ALLOWED_CWES` list in `infrastructure/finding_validation.py`.
   Missing CWE assignments fall back to the kind's `default_cwe`.

5. **Propagator rule enum** — every `rule:` value is one of the
   enum entries in §8.

6. **Requires resolution** — every entry in `requires:` names an
   existing spec file (built-in or user override). Cycles are
   rejected.

7. **Duplicate names within a spec** — a callable may not appear
   twice as the same role (source / sink / sanitizer / propagator)
   in the same file. Same-role duplicates surface as a spec-file
   error. Cross-role duplicates (source AND propagator, for
   instance) are permitted and represent a callable that both
   introduces taint and passes it through.

8. **Cross-file conflicts** — if two loaded spec files declare the
   same canonical name in *conflicting* roles (source vs. sink,
   or two different sink kinds), the loader emits a warning and
   the later-loaded file wins. User override files always load
   last.


## 12. Adding a new language

Concrete checklist for a contributor adding Python taint support:

1. **Adapter.** Implement the language adapter interface (defined
   in `src/falconeye/domain/services/taint_adapter.py`) for Python:
   `extract_functions`, `extract_calls`, `resolve_call`,
   `detect_frameworks`. Focus on 80% name-resolution coverage;
   the LLM verification layer handles the tail. Register the
   adapter in the plugin registry.

2. **Generic spec.** Write `python_generic.yaml` covering the
   standard library. Baseline set to aim for:
   - Sources: `os.environ.__getitem__`, `sys.argv.__getitem__`,
     `input`, `open(...).read`, `sys.stdin.read`, `subprocess.*.stdout`.
   - Sinks: `sqlite3.Cursor.execute`, `os.system`, `subprocess.Popen`,
     `pickle.loads`, `eval`, `exec`, `open` (path arg), `urllib.request.urlopen`.
   - Sanitizers: `int`, `float`, `bool`, `uuid.UUID`, `html.escape`,
     `urllib.parse.quote`, `os.path.basename`.
   - Propagators: `str.format`, `str.__add__`, `str.join`, `str.replace`,
     `f-string` (handled at the adapter level, not here).

3. **Framework specs.** Add one YAML per web framework. Start with
   the most-used: `python_flask.yaml`, `python_django.yaml`,
   `python_fastapi.yaml`.

4. **Kind additions.** If the language surfaces a new sink category
   (e.g. `template_injection` for Jinja2), add it to
   `_shared/kinds.yaml` with a documented parent and default CWE.
   Do not add language-local kinds for cross-language concepts.

5. **Framework detection.** Implement `detect_frameworks` on the
   adapter. Read `requirements.txt`, `pyproject.toml`, or
   `setup.py` and return the framework tags that apply.

6. **Test corpus.** Add a small vulnerable-sample repo under
   `tests/e2e/python_vulnerable/` and a test that scans it and
   asserts on the expected findings. Match the layout used by the
   existing C# test corpus.

For a full worked example, see `csharp_generic.yaml` alongside
this file.
