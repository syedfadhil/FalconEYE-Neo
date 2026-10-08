"""Unit tests for the C# taint adapter and adapter registry."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from falconeye.domain.models.taint import (
    FileResolutionContext,
    FunctionDef,
    Location,
    MethodResolutionContext,
    UnresolvedCall,
)
from falconeye.domain.services.taint_adapter import TaintLanguageAdapter
from falconeye.infrastructure.taint_adapters.adapter_registry import AdapterRegistry
from falconeye.infrastructure.taint_adapters.csharp import CSharpTaintAdapter


# ---------------------------------------------------------------------
# C# adapter -- function extraction
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def csharp_adapter():
    return CSharpTaintAdapter()


def _names(fns):
    """Just the canonical names, for compact assertions."""
    return [fn.canonical_name for fn in fns]


@pytest.mark.unit
def test_extract_method_in_block_namespace(csharp_adapter):
    src = textwrap.dedent("""
    namespace MyApp.Web {
        public class UserController {
            public string GetById(int id) { return null; }
        }
    }
    """)
    fns = csharp_adapter.extract_functions(src, "UserController.cs")
    assert "csharp:MyApp.Web.UserController.GetById" in _names(fns)


@pytest.mark.unit
def test_extract_constructor_uses_init(csharp_adapter):
    src = textwrap.dedent("""
    namespace X {
        public class Y {
            public Y(int a) { }
            public Y() { }
        }
    }
    """)
    fns = csharp_adapter.extract_functions(src, "Y.cs")
    ctors = [f for f in fns if f.is_constructor]
    assert len(ctors) == 2
    assert all(f.canonical_name == "csharp:X.Y.<init>" for f in ctors)
    # Parameters carried through
    assert ctors[0].parameters == ("a",)
    assert ctors[1].parameters == ()


@pytest.mark.unit
def test_file_scoped_namespace(csharp_adapter):
    """C# 10+ file-scoped namespace applies to sibling declarations."""
    src = textwrap.dedent("""
    namespace MyApp.Domain;

    public class User {
        public string GetName() { return null; }
    }
    """)
    fns = csharp_adapter.extract_functions(src, "User.cs")
    assert "csharp:MyApp.Domain.User.GetName" in _names(fns)


@pytest.mark.unit
def test_nested_namespaces_and_classes(csharp_adapter):
    src = textwrap.dedent("""
    namespace A.B {
        namespace C {
            public class D {
                public class E {
                    public void F() { }
                }
                public void G() { }
            }
        }
    }
    """)
    fns = _names(csharp_adapter.extract_functions(src, "N.cs"))
    assert "csharp:A.B.C.D.E.F" in fns
    assert "csharp:A.B.C.D.G" in fns


@pytest.mark.unit
def test_property_accessors_emit_get_set_pair(csharp_adapter):
    src = textwrap.dedent("""
    namespace X;
    public class Y {
        public string Name { get; set; }
    }
    """)
    fns = csharp_adapter.extract_functions(src, "Y.cs")
    names = _names(fns)
    assert "csharp:X.Y.get_Name" in names
    assert "csharp:X.Y.set_Name" in names

    setter = next(f for f in fns if f.canonical_name == "csharp:X.Y.set_Name")
    assert setter.parameters == ("value",)
    assert setter.accessor == "set"


@pytest.mark.unit
def test_init_accessor_folds_to_set(csharp_adapter):
    """C# `init` accessors compile to setters; the adapter treats them as such."""
    src = textwrap.dedent("""
    namespace X;
    public class Y {
        public int Age { get; init; }
    }
    """)
    fns = csharp_adapter.extract_functions(src, "Y.cs")
    names = _names(fns)
    assert "csharp:X.Y.get_Age" in names
    assert "csharp:X.Y.set_Age" in names
    setter = next(f for f in fns if f.canonical_name == "csharp:X.Y.set_Age")
    assert setter.accessor == "set"


@pytest.mark.unit
def test_static_modifier_detected(csharp_adapter):
    src = textwrap.dedent("""
    namespace X;
    public class Y {
        public static Y Create() { return null; }
        public Y Instance() { return this; }
    }
    """)
    fns = csharp_adapter.extract_functions(src, "Y.cs")
    static_fn = next(f for f in fns if f.canonical_name == "csharp:X.Y.Create")
    instance_fn = next(f for f in fns if f.canonical_name == "csharp:X.Y.Instance")
    assert static_fn.is_static is True
    assert instance_fn.is_static is False


@pytest.mark.unit
def test_interface_methods_extracted(csharp_adapter):
    src = textwrap.dedent("""
    namespace X;
    public interface IFoo {
        void Bar();
        string Baz(int n);
    }
    """)
    fns = _names(csharp_adapter.extract_functions(src, "IFoo.cs"))
    assert "csharp:X.IFoo.Bar" in fns
    assert "csharp:X.IFoo.Baz" in fns


@pytest.mark.unit
def test_struct_constructor(csharp_adapter):
    src = textwrap.dedent("""
    namespace X;
    public struct Point {
        public int X, Y;
        public Point(int x, int y) { X = x; Y = y; }
    }
    """)
    fns = csharp_adapter.extract_functions(src, "Point.cs")
    ctor = next(f for f in fns if f.is_constructor)
    assert ctor.canonical_name == "csharp:X.Point.<init>"
    assert ctor.parameters == ("x", "y")


@pytest.mark.unit
def test_top_level_statements_produce_nothing(csharp_adapter):
    """Top-level statements have no enclosing type; adapter drops them."""
    src = 'System.Console.WriteLine("hello");\n'
    assert csharp_adapter.extract_functions(src, "top.cs") == []


@pytest.mark.unit
def test_empty_source_returns_empty(csharp_adapter):
    assert csharp_adapter.extract_functions("", "empty.cs") == []


@pytest.mark.unit
def test_local_functions_are_skipped(csharp_adapter):
    """Local functions have no stable canonical name; skipped for now."""
    src = textwrap.dedent("""
    namespace X;
    public class Y {
        public void Outer() {
            void Local() { }
            Local();
        }
    }
    """)
    fns = _names(csharp_adapter.extract_functions(src, "Y.cs"))
    assert "csharp:X.Y.Outer" in fns
    # Local() has no canonical name; must not appear.
    assert not any("Local" in n for n in fns)


@pytest.mark.unit
def test_extracted_function_carries_location(csharp_adapter):
    src = textwrap.dedent("""
    namespace X;
    public class Y {
        public void Foo() { }
    }
    """)
    fns = csharp_adapter.extract_functions(src, "path/to/Y.cs")
    foo = next(f for f in fns if f.canonical_name == "csharp:X.Y.Foo")
    assert isinstance(foo.location, Location)
    assert foo.location.file_path == "path/to/Y.cs"
    assert foo.location.line_start >= 1
    assert foo.location.line_end >= foo.location.line_start


# ---------------------------------------------------------------------
# C# adapter -- framework detection
# ---------------------------------------------------------------------


@pytest.mark.unit
def test_framework_detection_aspnet(csharp_adapter, tmp_path):
    (tmp_path / "MyApp.csproj").write_text(textwrap.dedent("""\
        <Project Sdk="Microsoft.NET.Sdk.Web">
          <ItemGroup>
            <PackageReference Include="Microsoft.AspNetCore.OpenApi" Version="8.0.0" />
          </ItemGroup>
        </Project>
    """))
    assert csharp_adapter.detect_frameworks(tmp_path) == ["csharp_aspnet"]


@pytest.mark.unit
def test_framework_detection_multiple(csharp_adapter, tmp_path):
    (tmp_path / "MyApp.csproj").write_text(textwrap.dedent("""\
        <Project>
          <ItemGroup>
            <PackageReference Include="Microsoft.AspNetCore.Mvc" />
            <PackageReference Include="Microsoft.EntityFrameworkCore.SqlServer" />
            <PackageReference Include="Dapper" />
            <PackageReference Include="Newtonsoft.Json" />
          </ItemGroup>
        </Project>
    """))
    result = csharp_adapter.detect_frameworks(tmp_path)
    assert result == ["csharp_aspnet", "csharp_dapper", "csharp_efcore"]


@pytest.mark.unit
def test_framework_detection_nested_projects(csharp_adapter, tmp_path):
    """rglob picks up .csproj files in nested directories."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "Api.csproj").write_text(
        '<Project><ItemGroup><PackageReference Include="Microsoft.AspNetCore.Mvc" /></ItemGroup></Project>'
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "Tests.csproj").write_text(
        '<Project><ItemGroup><PackageReference Include="Microsoft.EntityFrameworkCore" /></ItemGroup></Project>'
    )
    result = csharp_adapter.detect_frameworks(tmp_path)
    assert "csharp_aspnet" in result
    assert "csharp_efcore" in result


@pytest.mark.unit
def test_framework_detection_empty_repo(csharp_adapter, tmp_path):
    assert csharp_adapter.detect_frameworks(tmp_path) == []


@pytest.mark.unit
def test_framework_detection_missing_path(csharp_adapter):
    assert csharp_adapter.detect_frameworks(Path("/does/not/exist/anywhere")) == []


# ---------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------


@pytest.mark.unit
def test_registry_register_and_get():
    reg = AdapterRegistry()
    adapter = CSharpTaintAdapter()
    reg.register(adapter)
    assert reg.get("csharp") is adapter
    assert reg.get("python") is None
    assert "csharp" in reg
    assert len(reg) == 1


@pytest.mark.unit
def test_registry_rejects_duplicate_language():
    reg = AdapterRegistry()
    reg.register(CSharpTaintAdapter())
    with pytest.raises(ValueError, match="already registered"):
        reg.register(CSharpTaintAdapter())


@pytest.mark.unit
def test_registry_rejects_empty_language():
    class Broken(TaintLanguageAdapter):
        language = ""
        def extract_functions(self, source, file_path): return []
        def detect_frameworks(self, project_root): return []
    reg = AdapterRegistry()
    with pytest.raises(ValueError, match="empty `language`"):
        reg.register(Broken())


@pytest.mark.unit
def test_registry_languages_iterator():
    reg = AdapterRegistry()
    reg.register(CSharpTaintAdapter())
    assert list(reg.languages()) == ["csharp"]


# ---------------------------------------------------------------------
# Config wiring: DIContainer builds taint services when opted in
# ---------------------------------------------------------------------


@pytest.mark.unit
def test_config_per_language_taint_config_defaults():
    from falconeye.infrastructure.config.config_models import (
        LanguageTaintConfig, LanguageSettings, LanguagesConfig,
    )
    # Defaults: taint disabled everywhere; per_language empty dict.
    lc = LanguagesConfig()
    assert lc.per_language == {}
    ltc = LanguageTaintConfig()
    assert ltc.enabled is False
    assert ltc.max_hops == 5
    assert ltc.framework_autodetect is True
    assert ltc.specs == []


@pytest.mark.unit
def test_config_extra_forbid_on_taint_config():
    """Typos in taint config must raise, not silently drift."""
    from falconeye.infrastructure.config.config_models import LanguageTaintConfig
    with pytest.raises(Exception):
        # extra="forbid" -> unknown field rejected
        LanguageTaintConfig(enabld=True)  # typo


@pytest.mark.unit
def test_config_max_hops_bounded():
    from falconeye.infrastructure.config.config_models import LanguageTaintConfig
    with pytest.raises(Exception):
        LanguageTaintConfig(enabled=True, max_hops=0)  # ge=1
    with pytest.raises(Exception):
        LanguageTaintConfig(enabled=True, max_hops=100)  # le=20

# ---------------------------------------------------------------------
# C# adapter -- extract_calls
# ---------------------------------------------------------------------


def _calls_by_flag(calls):
    """Group calls by their (is_constructor, is_property_set, ordinary) role."""
    ordinary = [c for c in calls if not c.is_constructor and not c.is_property_set]
    ctors    = [c for c in calls if c.is_constructor]
    setters  = [c for c in calls if c.is_property_set]
    return ordinary, ctors, setters


@pytest.mark.unit
def test_extract_calls_ordinary_invocation(csharp_adapter):
    src = textwrap.dedent("""
    namespace X;
    public class Y {
        public void M() { obj.Foo(1, 2); }
    }
    """)
    calls = csharp_adapter.extract_calls(src, "Y.cs")
    ordinary, ctors, setters = _calls_by_flag(calls)
    assert len(ordinary) == 1 and not ctors and not setters
    assert ordinary[0].callee_text == "obj.Foo"
    assert ordinary[0].positional_arg_count == 2
    assert ordinary[0].in_function == "csharp:X.Y.M"


@pytest.mark.unit
def test_extract_calls_namespace_qualified_static(csharp_adapter):
    """Fully-qualified static calls are emitted with the full dotted
    callee text -- the resolver strips using-shortened forms in a
    later patch, but the raw name is what the extractor produces.
    """
    src = textwrap.dedent("""
    namespace X;
    public class Y {
        public string M(string p) { return System.IO.File.ReadAllText(p); }
    }
    """)
    calls = csharp_adapter.extract_calls(src, "Y.cs")
    ordinary, _, _ = _calls_by_flag(calls)
    assert any(c.callee_text == "System.IO.File.ReadAllText" for c in ordinary)


@pytest.mark.unit
def test_extract_calls_bare_call(csharp_adapter):
    """Bare `Helper()` (implicit this.Helper or static Helper) -- the
    resolver disambiguates later; the extractor emits the raw name."""
    src = textwrap.dedent("""
    namespace X;
    public class Y {
        public void M() { Helper(); }
    }
    """)
    calls = csharp_adapter.extract_calls(src, "Y.cs")
    ordinary, _, _ = _calls_by_flag(calls)
    assert len(ordinary) == 1
    assert ordinary[0].callee_text == "Helper"
    assert ordinary[0].positional_arg_count == 0


@pytest.mark.unit
def test_extract_calls_constructor(csharp_adapter):
    """`new SqlCommand(q, conn)` -> is_constructor=True, callee_text is
    just the type name (resolver appends `.<init>`).
    """
    src = textwrap.dedent("""
    namespace X;
    public class Y {
        public void M() { var cmd = new SqlCommand(q, conn); }
    }
    """)
    calls = csharp_adapter.extract_calls(src, "Y.cs")
    _, ctors, _ = _calls_by_flag(calls)
    assert len(ctors) == 1
    assert ctors[0].callee_text == "SqlCommand"
    assert ctors[0].positional_arg_count == 2
    assert ctors[0].is_constructor is True
    assert ctors[0].is_property_set is False
    assert ctors[0].in_function == "csharp:X.Y.M"


@pytest.mark.unit
def test_extract_calls_property_setter(csharp_adapter):
    """`obj.Name = value` -> is_property_set=True, positional_arg_count=1
    (RHS is treated as the single value argument, matching the
    `tainted_args: [value]` shape in the taint spec YAMLs)."""
    src = textwrap.dedent("""
    namespace X;
    public class Y {
        public void M() { cmd.CommandText = q; }
    }
    """)
    calls = csharp_adapter.extract_calls(src, "Y.cs")
    _, _, setters = _calls_by_flag(calls)
    assert len(setters) == 1
    assert setters[0].callee_text == "cmd.CommandText"
    assert setters[0].positional_arg_count == 1
    assert setters[0].is_property_set is True
    assert setters[0].is_constructor is False


@pytest.mark.unit
def test_extract_calls_ignores_plain_identifier_assignment(csharp_adapter):
    """`x = y` (identifier LHS, not member access) is NOT a property
    setter and must not be emitted.
    """
    src = textwrap.dedent("""
    namespace X;
    public class Y {
        public void M() { int x = 3; x = 4; }
    }
    """)
    calls = csharp_adapter.extract_calls(src, "Y.cs")
    assert calls == []


@pytest.mark.unit
def test_extract_calls_chained(csharp_adapter):
    """foo.Bar().Baz() emits both calls (outer + inner) with correct
    positional argument counts. Chained calls are the typical
    pattern in LINQ / fluent APIs -- both must appear in the graph."""
    src = textwrap.dedent("""
    namespace X;
    public class Y {
        public void M() { foo.Bar(1).Baz(2, 3); }
    }
    """)
    calls = csharp_adapter.extract_calls(src, "Y.cs")
    ordinary, _, _ = _calls_by_flag(calls)
    callees = [(c.callee_text, c.positional_arg_count) for c in ordinary]
    assert ("foo.Bar", 1) in callees
    assert ("foo.Bar(1).Baz", 2) in callees


@pytest.mark.unit
def test_extract_calls_in_constructor_body(csharp_adapter):
    """Calls inside a constructor attribute to `<init>` in_function."""
    src = textwrap.dedent("""
    namespace X;
    public class Y {
        public Y(string s) { InitializeFrom(s); }
    }
    """)
    calls = csharp_adapter.extract_calls(src, "Y.cs")
    ordinary, _, _ = _calls_by_flag(calls)
    assert len(ordinary) == 1
    assert ordinary[0].callee_text == "InitializeFrom"
    assert ordinary[0].in_function == "csharp:X.Y.<init>"


@pytest.mark.unit
def test_extract_calls_in_property_accessor(csharp_adapter):
    """A call inside a property setter body attributes to set_X;
    a call inside a getter attributes to get_X."""
    src = textwrap.dedent("""
    namespace X;
    public class Y {
        private string _name;
        public string Name {
            get { return Normalize(_name); }
            set { _name = Sanitize(value); }
        }
    }
    """)
    calls = csharp_adapter.extract_calls(src, "Y.cs")
    by_fn = {c.callee_text: c.in_function for c in calls if not c.is_property_set}
    assert by_fn.get("Normalize") == "csharp:X.Y.get_Name"
    assert by_fn.get("Sanitize") == "csharp:X.Y.set_Name"


@pytest.mark.unit
def test_extract_calls_drops_calls_outside_functions(csharp_adapter):
    """Top-level statements have no enclosing function; their calls
    are dropped (they cannot participate in a call-graph edge)."""
    src = 'System.Console.WriteLine("hello");\n'
    assert csharp_adapter.extract_calls(src, "top.cs") == []


@pytest.mark.unit
def test_extract_calls_end_to_end_sql_injection_shape(csharp_adapter):
    """The canonical shape the csharp_generic.yaml specs target:
    a SqlCommand constructor AND a CommandText setter in the same
    method, both attributed to the enclosing GetById."""
    src = textwrap.dedent("""
    namespace MyApp.Web;
    public class UserController {
        public string GetById(int id) {
            var query = "SELECT * FROM users WHERE id=" + id;
            var cmd = new SqlCommand(query, _conn);
            cmd.CommandText = query;
            return cmd.ExecuteReader().ToString();
        }
    }
    """)
    calls = csharp_adapter.extract_calls(src, "UserController.cs")
    ordinary, ctors, setters = _calls_by_flag(calls)
    # Constructor sink
    assert any(c.callee_text == "SqlCommand" and c.positional_arg_count == 2
               for c in ctors)
    # Property setter sink
    assert any(c.callee_text == "cmd.CommandText" for c in setters)
    # All calls attribute to the enclosing method
    expected_fn = "csharp:MyApp.Web.UserController.GetById"
    assert all(c.in_function == expected_fn for c in calls)


@pytest.mark.unit
def test_extract_calls_default_returns_empty_on_base_class():
    """The base TaintLanguageAdapter provides a concrete extract_calls
    that returns [] -- this is what enables partial adapters to be
    valid contributors while still growing the interface."""
    class Minimal(TaintLanguageAdapter):
        language = "made_up"
        def extract_functions(self, source, file_path): return []
        def detect_frameworks(self, project_root): return []
    m = Minimal()
    assert m.extract_calls("anything", "any.file") == []


@pytest.mark.unit
def test_extract_calls_carries_location(csharp_adapter):
    src = textwrap.dedent("""
    namespace X;
    public class Y {
        public void M() {
            foo();
        }
    }
    """)
    calls = csharp_adapter.extract_calls(src, "path/to/Y.cs")
    assert calls[0].location.file_path == "path/to/Y.cs"
    assert calls[0].location.line_start >= 1


@pytest.mark.unit
def test_extract_calls_empty_source(csharp_adapter):
    assert csharp_adapter.extract_calls("", "empty.cs") == []


# ---------------------------------------------------------------------
# C# adapter -- build_resolution_context
# ---------------------------------------------------------------------


@pytest.mark.unit
def test_context_captures_usings(csharp_adapter):
    src = textwrap.dedent('''
    using System.Data.SqlClient;
    using System.IO;
    using static System.Console;
    namespace X;
    public class Y {
        public void M() { }
    }
    ''')
    ctx = csharp_adapter.build_resolution_context(src, 'Y.cs')
    assert 'System.Data.SqlClient' in ctx.usings
    assert 'System.IO' in ctx.usings
    assert 'System.Console' in ctx.static_usings
    assert ctx.file_namespace == 'X'


@pytest.mark.unit
def test_context_using_alias_ignored(csharp_adapter):
    """Aliased usings are silently dropped (deferred to follow-up)."""
    src = textwrap.dedent('''
    using MyList = System.Collections.Generic.List<int>;
    namespace X;
    public class Y { public void M() { } }
    ''')
    ctx = csharp_adapter.build_resolution_context(src, 'Y.cs')
    # Neither the alias NOR the target should end up in usings.
    assert 'MyList' not in ctx.usings
    assert 'System.Collections.Generic.List' not in ctx.usings


@pytest.mark.unit
def test_context_captures_variable_types_explicit(csharp_adapter):
    src = textwrap.dedent('''
    namespace X;
    public class Y {
        public void M() {
            SqlCommand cmd = null;
            string query = null;
        }
    }
    ''')
    ctx = csharp_adapter.build_resolution_context(src, 'Y.cs')
    mctx = ctx.method_contexts['csharp:X.Y.M']
    assert mctx.variable_types['cmd'] == 'SqlCommand'
    assert mctx.variable_types['query'] == 'string'


@pytest.mark.unit
def test_context_captures_var_inference(csharp_adapter):
    """`var x = new Type(...)` infers x's type as Type."""
    src = textwrap.dedent('''
    namespace X;
    public class Y {
        public void M() {
            var cmd = new SqlCommand(q);
        }
    }
    ''')
    ctx = csharp_adapter.build_resolution_context(src, 'Y.cs')
    mctx = ctx.method_contexts['csharp:X.Y.M']
    assert mctx.variable_types['cmd'] == 'SqlCommand'


@pytest.mark.unit
def test_context_var_with_nonliteral_skipped(csharp_adapter):
    """`var x = Foo()` cannot be inferred without flow types; skipped."""
    src = textwrap.dedent('''
    namespace X;
    public class Y {
        public void M() { var cmd = GetIt(); }
    }
    ''')
    ctx = csharp_adapter.build_resolution_context(src, 'Y.cs')
    mctx = ctx.method_contexts['csharp:X.Y.M']
    assert 'cmd' not in mctx.variable_types


@pytest.mark.unit
def test_context_captures_parameters(csharp_adapter):
    src = textwrap.dedent('''
    namespace X;
    public class Y {
        public void M(SqlCommand cmd, string q) { }
    }
    ''')
    ctx = csharp_adapter.build_resolution_context(src, 'Y.cs')
    mctx = ctx.method_contexts['csharp:X.Y.M']
    assert mctx.variable_types['cmd'] == 'SqlCommand'
    assert mctx.variable_types['q'] == 'string'


@pytest.mark.unit
def test_context_captures_class_fields(csharp_adapter):
    src = textwrap.dedent('''
    namespace X;
    public class Y {
        private SqlCommand _cmd;
        public string _q;
    }
    ''')
    ctx = csharp_adapter.build_resolution_context(src, 'Y.cs')
    fields = ctx.class_fields['X.Y']
    assert fields['_cmd'] == 'SqlCommand'
    assert fields['_q'] == 'string'


@pytest.mark.unit
def test_context_property_setter_value_type(csharp_adapter):
    """Inside a property setter body, `value` has the property's type."""
    src = textwrap.dedent('''
    namespace X;
    public class Y {
        public SqlCommand Cmd {
            get { return null; }
            set { _cmd = value; }
        }
    }
    ''')
    ctx = csharp_adapter.build_resolution_context(src, 'Y.cs')
    setter = ctx.method_contexts['csharp:X.Y.set_Cmd']
    assert setter.variable_types['value'] == 'SqlCommand'


# ---------------------------------------------------------------------
# C# adapter -- resolve_call
# ---------------------------------------------------------------------


@pytest.fixture(scope='module')
def loaded_taint_index():
    """The real shipped csharp_generic index -- what production uses."""
    from falconeye.infrastructure.taint_index import load_taint_index
    return load_taint_index(languages=['csharp'])


def _resolve_one(adapter, src, call_index=0, *, taint_index):
    ctx = adapter.build_resolution_context(src, 'test.cs')
    calls = adapter.extract_calls(src, 'test.cs')
    call = calls[call_index]
    return adapter.resolve_call(call, ctx, taint_index), call


@pytest.mark.unit
def test_resolve_constructor_via_using(csharp_adapter, loaded_taint_index):
    src = textwrap.dedent('''
    using System.Data.SqlClient;
    namespace X;
    public class Y {
        public void M() { var cmd = new SqlCommand(q, conn); }
    }
    ''')
    resolved, _ = _resolve_one(csharp_adapter, src, taint_index=loaded_taint_index)
    assert resolved == 'csharp:System.Data.SqlClient.SqlCommand.<init>'
    assert loaded_taint_index.get_sink(resolved) is not None


@pytest.mark.unit
def test_resolve_property_setter_via_local_var(csharp_adapter, loaded_taint_index):
    """Local var typed via var/new, property setter resolves through it."""
    src = textwrap.dedent('''
    using System.Data.SqlClient;
    namespace X;
    public class Y {
        public void M(string q) {
            var cmd = new SqlCommand(q, _conn);
            cmd.CommandText = q;
        }
    }
    ''')
    ctx = csharp_adapter.build_resolution_context(src, 'test.cs')
    calls = csharp_adapter.extract_calls(src, 'test.cs')
    prop_call = next(c for c in calls if c.is_property_set)
    resolved = csharp_adapter.resolve_call(prop_call, ctx, loaded_taint_index)
    assert resolved == 'csharp:System.Data.SqlClient.SqlCommand.set_CommandText'
    assert loaded_taint_index.get_sink(resolved) is not None


@pytest.mark.unit
def test_resolve_property_setter_via_class_field(csharp_adapter, loaded_taint_index):
    """Class field typed explicitly, setter resolves through field type."""
    src = textwrap.dedent('''
    using System.Data.SqlClient;
    namespace X;
    public class Y {
        private SqlCommand _cmd;
        public void M(string q) {
            _cmd.CommandText = q;
        }
    }
    ''')
    ctx = csharp_adapter.build_resolution_context(src, 'test.cs')
    calls = csharp_adapter.extract_calls(src, 'test.cs')
    prop_call = next(c for c in calls if c.is_property_set)
    resolved = csharp_adapter.resolve_call(prop_call, ctx, loaded_taint_index)
    assert resolved == 'csharp:System.Data.SqlClient.SqlCommand.set_CommandText'
    assert loaded_taint_index.get_sink(resolved) is not None


@pytest.mark.unit
def test_resolve_fully_qualified_source(csharp_adapter, loaded_taint_index):
    src = textwrap.dedent('''
    namespace X;
    public class Y {
        public void M(string p) { System.IO.File.ReadAllText(p); }
    }
    ''')
    resolved, _ = _resolve_one(csharp_adapter, src, taint_index=loaded_taint_index)
    assert resolved == 'csharp:System.IO.File.ReadAllText'
    assert loaded_taint_index.get_source(resolved) is not None


@pytest.mark.unit
def test_resolve_namespace_shortened_via_using(csharp_adapter, loaded_taint_index):
    src = textwrap.dedent('''
    using System.IO;
    namespace X;
    public class Y {
        public void M(string p) { File.ReadAllText(p); }
    }
    ''')
    resolved, _ = _resolve_one(csharp_adapter, src, taint_index=loaded_taint_index)
    assert resolved == 'csharp:System.IO.File.ReadAllText'
    assert loaded_taint_index.get_source(resolved) is not None


@pytest.mark.unit
def test_resolve_bare_call_via_static_using(csharp_adapter, loaded_taint_index):
    """`using static X.Type;` + bare call resolves to X.Type.Method."""
    # ReadLine is a shipped stdin source, so this doubles as an
    # end-to-end spec-match check.
    src = textwrap.dedent('''
    using static System.Console;
    namespace X;
    public class Y {
        public string M() { return ReadLine(); }
    }
    ''')
    resolved, _ = _resolve_one(csharp_adapter, src, taint_index=loaded_taint_index)
    assert resolved == 'csharp:System.Console.ReadLine'
    assert loaded_taint_index.get_source(resolved) is not None


@pytest.mark.unit
def test_resolve_parameter_typed_receiver(csharp_adapter, loaded_taint_index):
    """Method parameter with declared type resolves member access."""
    src = textwrap.dedent('''
    using System.Data.SqlClient;
    namespace X;
    public class Y {
        public void M(SqlCommand cmd, string q) {
            cmd.CommandText = q;
        }
    }
    ''')
    ctx = csharp_adapter.build_resolution_context(src, 'test.cs')
    calls = csharp_adapter.extract_calls(src, 'test.cs')
    prop_call = next(c for c in calls if c.is_property_set)
    resolved = csharp_adapter.resolve_call(prop_call, ctx, loaded_taint_index)
    assert resolved == 'csharp:System.Data.SqlClient.SqlCommand.set_CommandText'


@pytest.mark.unit
def test_resolve_this_call(csharp_adapter, loaded_taint_index):
    src = textwrap.dedent('''
    namespace X;
    public class Y {
        public void M() { this.Helper(); }
        private void Helper() { }
    }
    ''')
    resolved, _ = _resolve_one(csharp_adapter, src, taint_index=loaded_taint_index)
    # No spec match for user-defined; falls back to best-guess candidate
    # which is the this-scope resolution.
    assert resolved == 'csharp:X.Y.Helper'


@pytest.mark.unit
def test_resolve_unresolvable_returns_none(csharp_adapter, loaded_taint_index):
    """A call with no receiver in scope and no static usings returns None."""
    # The chained-call outer part -- `foo.Bar().Baz` -- has receiver
    # `foo.Bar()` which is not a variable or type. Currently the
    # resolver treats `foo.Bar()` as a type path and tries to expand
    # via usings; the resulting candidate almost certainly won't
    # match the spec.
    src = textwrap.dedent('''
    namespace X;
    public class Y {
        public void M() { foo.Bar().Baz(); }
    }
    ''')
    ctx = csharp_adapter.build_resolution_context(src, 'test.cs')
    calls = csharp_adapter.extract_calls(src, 'test.cs')
    # Inner call `foo.Bar` -- receiver foo not in scope -> best guess
    # but no spec match.
    inner = next(c for c in calls if c.callee_text == 'foo.Bar')
    resolved_inner = csharp_adapter.resolve_call(inner, ctx, loaded_taint_index)
    assert loaded_taint_index.get_source(resolved_inner) is None
    assert loaded_taint_index.get_sink(resolved_inner) is None


@pytest.mark.unit
def test_resolve_prefers_spec_match_over_first_candidate(csharp_adapter, loaded_taint_index):
    """When multiple candidates exist (bare type + multiple usings),
    the resolver picks the one that hits a spec entry.
    """
    src = textwrap.dedent('''
    using SomeUnusedNamespace;
    using System.Data.SqlClient;
    using AnotherUnusedNamespace;
    namespace X;
    public class Y {
        public void M() { var c = new SqlCommand(q, conn); }
    }
    ''')
    resolved, _ = _resolve_one(csharp_adapter, src, taint_index=loaded_taint_index)
    # Should pick the SqlClient version, not the first using in order.
    assert resolved == 'csharp:System.Data.SqlClient.SqlCommand.<init>'


@pytest.mark.unit
def test_resolve_base_adapter_returns_none():
    """Default resolve_call on TaintLanguageAdapter returns None."""
    from falconeye.infrastructure.taint_index import load_taint_index
    class Minimal(TaintLanguageAdapter):
        language = 'made_up'
        def extract_functions(self, source, file_path): return []
        def detect_frameworks(self, project_root): return []
    m = Minimal()
    idx = load_taint_index(languages=['csharp'])
    call = UnresolvedCall(
        callee_text='foo', location=Location('x', 1, 1),
        in_function='made_up:X.Y.M',
    )
    assert m.resolve_call(call, FileResolutionContext(file_path='x'), idx) is None


@pytest.mark.unit
def test_resolve_end_to_end_sql_injection_matches_spec(csharp_adapter, loaded_taint_index):
    """The exact scenario csharp_generic.yaml is designed to catch:
    a constructor sink + property setter sink both resolve to entries
    the taint index recognizes.
    """
    src = textwrap.dedent('''
    using System.Data.SqlClient;
    namespace MyApp.Web;
    public class UserController {
        public string GetById(int id) {
            var query = "SELECT * FROM users WHERE id=" + id;
            var cmd = new SqlCommand(query, _conn);
            cmd.CommandText = query;
            return cmd.ExecuteReader().ToString();
        }
    }
    ''')
    ctx = csharp_adapter.build_resolution_context(src, 'UC.cs')
    calls = csharp_adapter.extract_calls(src, 'UC.cs')
    resolved = [csharp_adapter.resolve_call(c, ctx, loaded_taint_index) for c in calls]
    sinks_hit = [r for r in resolved if r and loaded_taint_index.get_sink(r)]
    assert 'csharp:System.Data.SqlClient.SqlCommand.<init>' in sinks_hit
    assert 'csharp:System.Data.SqlClient.SqlCommand.set_CommandText' in sinks_hit

