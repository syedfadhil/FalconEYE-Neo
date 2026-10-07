"""Structural metadata models extracted from AST analysis."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class FunctionInfo:
    """Information about a function extracted from AST."""
    name: str
    line: int
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    is_async: bool = False
    decorators: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "line": self.line,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "is_async": self.is_async,
            "decorators": self.decorators,
        }


@dataclass
class ImportInfo:
    """Information about an import statement."""
    statement: str
    line: int
    module: str
    imported_names: List[str] = field(default_factory=list)
    is_relative: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "statement": self.statement,
            "line": self.line,
            "module": self.module,
            "imported_names": self.imported_names,
            "is_relative": self.is_relative,
        }


@dataclass
class CallInfo:
    """Information about a function call."""
    function: str
    line: int
    call_type: str = "call"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "function": self.function,
            "line": self.line,
            "call_type": self.call_type,
        }


@dataclass
class ClassInfo:
    """Information about a class definition."""
    name: str
    line: int
    bases: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "line": self.line,
            "bases": self.bases,
            "methods": self.methods,
        }


@dataclass
class ControlFlowNode:
    """Node in control flow graph."""
    node_type: str  # if, while, for, try, etc.
    line: int
    condition: Optional[str] = None
    children: List["ControlFlowNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_type": self.node_type,
            "line": self.line,
            "condition": self.condition,
            "children": [child.to_dict() for child in self.children],
        }


@dataclass
class DataFlowInfo:
    """Data flow information for security analysis."""
    variable: str
    defined_at: int
    used_at: List[int] = field(default_factory=list)
    is_tainted: bool = False  # Determined by AI, not pattern matching
    flows_to: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variable": self.variable,
            "defined_at": self.defined_at,
            "used_at": self.used_at,
            "is_tainted": self.is_tainted,
            "flows_to": self.flows_to,
        }


@dataclass
class StructuralMetadata:
    """
    Complete structural metadata for a code file.

    This metadata is used to provide rich context to the AI,
    NOT for pattern-based vulnerability detection.
    """
    file_path: str
    language: str
    functions: List[FunctionInfo] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)
    calls: List[CallInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    control_flow: List[ControlFlowNode] = field(default_factory=list)
    data_flows: List[DataFlowInfo] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "language": self.language,
            "functions": [f.to_dict() for f in self.functions],
            "imports": [i.to_dict() for i in self.imports],
            "calls": [c.to_dict() for c in self.calls],
            "classes": [c.to_dict() for c in self.classes],
            "dependencies": self.dependencies,
            "control_flow": [cf.to_dict() for cf in self.control_flow],
            "data_flows": [df.to_dict() for df in self.data_flows],
            "stats": {
                "function_count": len(self.functions),
                "import_count": len(self.imports),
                "call_count": len(self.calls),
                "class_count": len(self.classes),
                "dependency_count": len(self.dependencies),
            }
        }

    @property
    def has_security_sensitive_imports(self) -> bool:
        """
        Check if file has potentially security-sensitive imports.

        NOTE: This is just a flag for AI context, not a vulnerability indicator.
        The AI makes all security decisions.
        """
        # This is metadata, not pattern matching for vulnerabilities
        return len(self.imports) > 0

    @property
    def complexity_score(self) -> int:
        """
        Basic complexity indicator for AI context.

        Not used for vulnerability detection.
        """
        return (
            len(self.functions) * 2 +
            len(self.classes) * 3 +
            len(self.control_flow) +
            len(self.calls)
        )