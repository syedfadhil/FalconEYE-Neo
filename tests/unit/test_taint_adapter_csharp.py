"""Unit tests for the C# taint adapter and adapter registry."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from falconeye.domain.models.taint import FunctionDef, Location
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
