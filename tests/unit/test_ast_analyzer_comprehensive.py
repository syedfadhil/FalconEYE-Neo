"""Unit tests for enhanced AST analyzer."""

from unittest.mock import MagicMock

from falconeye.infrastructure.ast import ast_analyzer as ast_module
from falconeye.infrastructure.ast.ast_analyzer import EnhancedASTAnalyzer


def names(items):
    return {item.name for item in items}


def modules(imports):
    return {item.module for item in imports}


def test_ast_analyzer_extracts_python_functions_classes_imports_and_calls():
    code = """import os
from pathlib import Path

class Service:
    def method(self, value):
        return helper(value)

def helper(value):
    print(os.getcwd())
    return Path(value)
"""
    metadata = EnhancedASTAnalyzer().analyze_file("app.py", code)

    assert metadata.language == "python"
    assert {"method", "helper"}.issubset(names(metadata.functions))
    assert "Service" in names(metadata.classes)
    assert {"os", "pathlib"}.issubset(modules(metadata.imports))
    assert {"helper", "print", "os.getcwd", "Path"}.intersection({c.function for c in metadata.calls})
    assert metadata.complexity_score > 0


def test_ast_analyzer_extracts_javascript_functions_classes_and_imports():
    code = """import express from 'express';
class Controller {
  handle(req) { return sanitize(req.body.name); }
}
function sanitize(value) { return value.trim(); }
const app = express();
"""
    metadata = EnhancedASTAnalyzer().analyze_file("app.js", code)

    assert metadata.language == "javascript"
    assert "sanitize" in names(metadata.functions)
    assert "Controller" in names(metadata.classes)
    assert any("express" in imp.statement for imp in metadata.imports)


def test_ast_analyzer_extracts_go_functions_and_imports():
    code = """package main

import "fmt"

type Server struct {}

func Handle(name string) string {
    fmt.Println(name)
    return name
}
"""
    metadata = EnhancedASTAnalyzer().analyze_file("main.go", code)

    assert metadata.language == "go"
    assert "Handle" in names(metadata.functions)
    assert any("fmt" in imp.statement for imp in metadata.imports)


def test_ast_analyzer_handles_unsupported_extension_gracefully():
    metadata = EnhancedASTAnalyzer().analyze_file("README.md", "# docs")

    assert metadata.language == "unknown"
    assert metadata.functions == []
    assert metadata.classes == []
    assert metadata.imports == []


def test_ast_analyzer_handles_unparseable_and_binary_like_content_gracefully():
    analyzer = EnhancedASTAnalyzer()

    bad_python = analyzer.analyze_file("broken.py", "def broken(:\n  pass")
    binary_like = analyzer.analyze_file("blob.py", "\x00\x01\x02not really python")

    assert bad_python.language == "python"
    assert isinstance(bad_python.functions, list)
    assert binary_like.language == "python"
    assert isinstance(binary_like.imports, list)


def test_ast_analyzer_logs_and_extracts_python_with_tree_sitter_unavailable(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(ast_module.FalconEyeLogger, "get_instance", lambda: logger)
    monkeypatch.setattr(ast_module, "tree_sitter_language_pack", None)

    metadata = EnhancedASTAnalyzer().analyze_file(
        "service.py",
        """import os
from .helpers import clean

class Service:
    async def handle(self, value):
        return clean(os.getenv(value))
""",
    )

    logger.warning.assert_called_once()
    assert metadata.language == "python"
    assert "Service" in names(metadata.classes)
    assert "handle" in names(metadata.functions)
    assert {"os", "helpers"}.issubset(modules(metadata.imports))
    assert {"clean", "os.getenv"}.issubset({call.function for call in metadata.calls})


def test_ast_analyzer_fallback_extracts_javascript_arrow_functions(monkeypatch):
    monkeypatch.setattr(ast_module, "tree_sitter_language_pack", None)

    metadata = EnhancedASTAnalyzer().analyze_file(
        "app.js",
        """import express from 'express';
class Controller {}
async function load() { return true; }
const sanitize = (value) => value.trim();
""",
    )

    assert metadata.language == "javascript"
    assert {"load", "sanitize"}.issubset(names(metadata.functions))
    assert "Controller" in names(metadata.classes)
    assert "express" in modules(metadata.imports)


def test_ast_analyzer_fallback_extracts_go_multiline_imports(monkeypatch):
    monkeypatch.setattr(ast_module, "tree_sitter_language_pack", None)

    metadata = EnhancedASTAnalyzer().analyze_file(
        "main.go",
        """package main
import (
    "fmt"
    "os"
)

type Server struct {}
func Handle() {}
""",
    )

    assert metadata.language == "go"
    assert {"fmt", "os"}.issubset(modules(metadata.imports))
    fmt_import = next(item for item in metadata.imports if item.module == "fmt")
    os_import = next(item for item in metadata.imports if item.module == "os")
    assert fmt_import.line == 3
    assert os_import.line == 4
    assert "Handle" in names(metadata.functions)
    assert "Server" in names(metadata.classes)


def test_ast_analyzer_fallback_python_broken_syntax_returns_empty_metadata(monkeypatch):
    monkeypatch.setattr(ast_module, "tree_sitter_language_pack", None)

    metadata = EnhancedASTAnalyzer().analyze_file("broken.py", "def broken(:\n  pass")

    assert metadata.language == "python"
    assert metadata.functions == []
    assert metadata.classes == []
    assert metadata.imports == []
