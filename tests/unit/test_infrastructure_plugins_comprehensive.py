"""Unit tests for built-in language plugins and plugin registry."""

import pytest

from falconeye.infrastructure.plugins.c_cpp_plugin import CCppPlugin
from falconeye.infrastructure.plugins.csharp_plugin import CSharpPlugin
from falconeye.infrastructure.plugins.dart_plugin import DartPlugin
from falconeye.infrastructure.plugins.go_plugin import GoPlugin
from falconeye.infrastructure.plugins.java_plugin import JavaPlugin
from falconeye.infrastructure.plugins.javascript_plugin import JavaScriptPlugin
from falconeye.infrastructure.plugins.php_plugin import PHPPlugin
from falconeye.infrastructure.plugins.plugin_registry import PluginRegistry
from falconeye.infrastructure.plugins.python_plugin import PythonPlugin
from falconeye.infrastructure.plugins.ruby_plugin import RubyPlugin
from falconeye.infrastructure.plugins.rust_plugin import RustPlugin


PLUGIN_CASES = [
    (PythonPlugin, "python", [".py", ".pyw"], "def login(user):\n    return user\n", "Python"),
    (JavaScriptPlugin, "javascript", [".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"], "function login(user) { return user; }", "JavaScript"),
    (GoPlugin, "go", [".go"], "package main\nfunc login(user string) string { return user }", "Go"),
    (RustPlugin, "rust", [".rs"], "fn login(user: &str) -> &str { user }", "Rust"),
    (CCppPlugin, "c_cpp", [".c", ".cpp", ".h"], "int login(char *user) { return 0; }", "C/C++"),
    (JavaPlugin, "java", [".java"], "class App { String login(String u) { return u; } }", "Java"),
    (CSharpPlugin, "csharp", [".cs", ".csx", ".cshtml", ".razor"], "class App { string Login(string u) => u; }", "C#"),
    (PHPPlugin, "php", [".php", ".phtml"], "<?php function login($u) { return $u; }", "PHP"),
    (RubyPlugin, "ruby", [".rb", ".rake", "Gemfile", "Rakefile"], "def login(user)\n  user\nend", "Ruby"),
    (DartPlugin, "dart", [".dart"], "String login(String user) => user;", "Dart"),
]


@pytest.mark.parametrize("plugin_cls,language,extensions,snippet,prompt_keyword", PLUGIN_CASES)
def test_language_plugins_report_language_extensions_prompts_and_chunking(plugin_cls, language, extensions, snippet, prompt_keyword):
    plugin = plugin_cls()

    assert plugin.language_name == language
    for ext in extensions:
        assert ext in plugin.file_extensions

    system_prompt = plugin.get_system_prompt()
    validation_prompt = plugin.get_validation_prompt()
    categories = plugin.get_vulnerability_categories()
    chunking = plugin.get_chunking_strategy()

    assert prompt_keyword.lower().split("/")[0] in system_prompt.lower()
    assert "security" in system_prompt.lower()
    assert "false positive" in validation_prompt.lower() or "validate" in validation_prompt.lower()
    assert categories and all(isinstance(category, str) for category in categories)
    assert chunking["chunk_size"] > 0
    assert 0 <= chunking["chunk_overlap"] < chunking["chunk_size"]

    # Representative fixture snippets should be non-empty source that can be carried
    # through plugin-guided chunk context without external files.
    fixture_chunk = snippet[: chunking["chunk_size"] * 20]
    assert fixture_chunk.strip()
    assert "login" in fixture_chunk.lower()


@pytest.mark.parametrize("plugin_cls,language,extensions,_,__", PLUGIN_CASES)
def test_each_plugin_is_discoverable_by_registry_language_and_extension(plugin_cls, language, extensions, _, __):
    registry = PluginRegistry()
    registry.load_all_plugins()

    plugin = registry.get_plugin(language)
    assert isinstance(plugin, plugin_cls)
    assert language in registry.get_supported_languages()

    for ext in extensions:
        assert registry.is_extension_supported(ext)
        assert registry.get_plugin_by_extension(ext).language_name == language


def test_plugin_registry_lists_all_builtins_and_rejects_unknowns():
    registry = PluginRegistry()
    registry.load_all_plugins()

    languages = set(registry.get_supported_languages())
    assert languages == {"python", "javascript", "go", "rust", "c_cpp", "java", "dart", "php", "ruby", "csharp"}
    assert len(registry.get_all_plugins()) == 10
    assert registry.get_plugin("unknown") is None
    assert registry.get_plugin_by_extension(".wat") is None
    assert registry.is_language_supported("python") is True
    assert registry.is_language_supported("unknown") is False
    assert "PluginRegistry" in repr(registry)
