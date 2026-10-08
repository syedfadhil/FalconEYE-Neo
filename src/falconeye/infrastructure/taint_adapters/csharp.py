"""
C# concrete implementation of ``TaintLanguageAdapter``.

Uses tree-sitter's C# grammar (via ``tree_sitter_language_pack``, the
same source the existing AST analyzer uses) to walk source and emit
``FunctionDef`` entries with canonical names of the form:

    csharp:Namespace.Sub.ClassName.MethodName
    csharp:Namespace.Sub.ClassName.<init>            (constructor)
    csharp:Namespace.Sub.ClassName.get_PropertyName  (property getter)
    csharp:Namespace.Sub.ClassName.set_PropertyName  (property setter)

Namespace and class context are assembled by walking the AST from
each definition site back up to the enclosing containers. This
handles nested classes (``Outer.Inner.Method``) and file-scoped
namespaces (C# 10+ ``namespace Foo;``) uniformly.

Framework detection reads ``.csproj`` files under ``project_root``
and maps ``<PackageReference Include="X">`` entries to spec names:

    Microsoft.AspNetCore.*         -> csharp_aspnet
    Microsoft.EntityFrameworkCore* -> csharp_efcore
    Dapper*                        -> csharp_dapper

Deliberately shallow: this is auto-detection for the common case, not
build-graph analysis. Operators who need pinned specs set
``framework_autodetect: false`` in ``config.yaml`` and list ``specs:``
explicitly.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import tree_sitter_language_pack
except ModuleNotFoundError:  # pragma: no cover
    tree_sitter_language_pack = None

from ...domain.models.taint import FunctionDef, Location, UnresolvedCall
from ...domain.services.taint_adapter import TaintLanguageAdapter


# Tree-sitter C# node types this adapter cares about. Kept in module
# scope so tests can reference the same set.
_TYPE_CONTAINER_NODES = frozenset({
    "class_declaration",
    "struct_declaration",
    "interface_declaration",
    "record_declaration",
    # record structs share record_declaration in tree-sitter's grammar.
})

_METHOD_LIKE_NODES = frozenset({
    "method_declaration",
    "constructor_declaration",
    "destructor_declaration",
    "local_function_statement",
    "operator_declaration",
    "conversion_operator_declaration",
})

_PROPERTY_LIKE_NODES = frozenset({
    "property_declaration",
    "indexer_declaration",
})

_ACCESSOR_KEYWORDS = frozenset({"get", "set", "init"})


# Framework-detection mapping. Ordered from most specific to most
# generic so the first match wins on ambiguous package prefixes.
_FRAMEWORK_MAP: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"^Microsoft\.AspNetCore(\.|$)"),         "csharp_aspnet"),
    (re.compile(r"^Microsoft\.EntityFrameworkCore(\.|$)"), "csharp_efcore"),
    (re.compile(r"^Dapper(\.|$)"),                        "csharp_dapper"),
)


class CSharpTaintAdapter(TaintLanguageAdapter):
    """C# adapter. Function extraction and framework detection.

    Call extraction, name resolution, and graph construction are
    deferred to follow-up patches -- see ``taint_specs/SCHEMA.md`` §12
    for the phased rollout.
    """

    language = "csharp"

    def __init__(self) -> None:
        self._parser = None
        if tree_sitter_language_pack is not None:
            try:
                self._parser = tree_sitter_language_pack.get_parser("csharp")
            except Exception:  # pragma: no cover
                # Same graceful-degrade posture as EnhancedASTAnalyzer:
                # a missing grammar makes extraction return [] rather
                # than crashing the scan.
                self._parser = None

    # ------------------------------------------------------------------
    # extract_functions
    # ------------------------------------------------------------------

    def extract_functions(self, source: str, file_path: str) -> List[FunctionDef]:
        if self._parser is None:
            return []

        tree = self._parser.parse(source.encode("utf-8"))
        root = tree.root_node

        out: List[FunctionDef] = []
        self._walk_and_extract(
            node=root,
            source_bytes=source.encode("utf-8"),
            file_path=file_path,
            namespace_stack=[],
            type_stack=[],
            out=out,
        )
        return out

    def _walk_and_extract(
        self,
        *,
        node,
        source_bytes: bytes,
        file_path: str,
        namespace_stack: List[str],
        type_stack: List[str],
        out: List[FunctionDef],
    ) -> None:
        """Recursive DFS. Extends context stacks as we descend into
        namespace and type nodes; emits FunctionDefs at method-like nodes.
        """
        t = node.type

        if t == "compilation_unit":
            # File-scoped namespaces (C# 10+) are SIBLINGS of the
            # declarations they scope, not parents:
            #   compilation_unit
            #     file_scoped_namespace_declaration        # "namespace X;"
            #     class_declaration                        # <-- sibling
            # We sweep children sequentially, extending the working
            # namespace stack when we see a file-scoped decl, and
            # applying that extended stack to every subsequent sibling.
            current_ns = list(namespace_stack)
            for child in node.children:
                if child.type == "file_scoped_namespace_declaration":
                    name = self._text_of_field(child, "name", source_bytes)
                    if name:
                        current_ns.append(name)
                    continue  # do not recurse into the decl itself
                self._walk_and_extract(
                    node=child,
                    source_bytes=source_bytes,
                    file_path=file_path,
                    namespace_stack=current_ns,
                    type_stack=type_stack,
                    out=out,
                )
            return

        # Block-scoped namespace: children ARE inside the namespace.
        if t == "namespace_declaration":
            name = self._text_of_field(node, "name", source_bytes)
            new_ns = list(namespace_stack)
            if name:
                new_ns.append(name)
            for child in node.children:
                self._walk_and_extract(
                    node=child,
                    source_bytes=source_bytes,
                    file_path=file_path,
                    namespace_stack=new_ns,
                    type_stack=type_stack,
                    out=out,
                )
            return

        if t in _TYPE_CONTAINER_NODES:
            name = self._text_of_field(node, "name", source_bytes)
            new_type_stack = type_stack + ([name] if name else [])
            for child in node.children:
                self._walk_and_extract(
                    node=child,
                    source_bytes=source_bytes,
                    file_path=file_path,
                    namespace_stack=namespace_stack,
                    type_stack=new_type_stack,
                    out=out,
                )
            return

        if t in _METHOD_LIKE_NODES:
            fn = self._function_def_from_method(
                node=node,
                source_bytes=source_bytes,
                file_path=file_path,
                namespace_stack=namespace_stack,
                type_stack=type_stack,
            )
            if fn is not None:
                out.append(fn)
            # Local functions can nest more local functions; keep walking.
            for child in node.children:
                self._walk_and_extract(
                    node=child,
                    source_bytes=source_bytes,
                    file_path=file_path,
                    namespace_stack=namespace_stack,
                    type_stack=type_stack,
                    out=out,
                )
            return

        if t in _PROPERTY_LIKE_NODES:
            for fn in self._function_defs_from_property(
                node=node,
                source_bytes=source_bytes,
                file_path=file_path,
                namespace_stack=namespace_stack,
                type_stack=type_stack,
            ):
                out.append(fn)
            # Properties don't nest interesting definitions, but keep
            # walking in case tree-sitter parses an odd shape.
            for child in node.children:
                self._walk_and_extract(
                    node=child,
                    source_bytes=source_bytes,
                    file_path=file_path,
                    namespace_stack=namespace_stack,
                    type_stack=type_stack,
                    out=out,
                )
            return

        # Generic recursion for anything not specifically handled.
        for child in node.children:
            self._walk_and_extract(
                node=child,
                source_bytes=source_bytes,
                file_path=file_path,
                namespace_stack=namespace_stack,
                type_stack=type_stack,
                out=out,
            )

    # ------------------------------------------------------------------
    # extract_calls
    # ------------------------------------------------------------------

    def extract_calls(self, source: str, file_path: str) -> List[UnresolvedCall]:
        if self._parser is None:
            return []

        source_bytes = source.encode("utf-8")
        tree = self._parser.parse(source_bytes)
        out: List[UnresolvedCall] = []
        self._walk_and_extract_calls(
            node=tree.root_node,
            source_bytes=source_bytes,
            file_path=file_path,
            namespace_stack=[],
            type_stack=[],
            current_function=None,
            out=out,
        )
        return out

    def _walk_and_extract_calls(
        self,
        *,
        node,
        source_bytes: bytes,
        file_path: str,
        namespace_stack: List[str],
        type_stack: List[str],
        current_function: Optional[str],
        out: List[UnresolvedCall],
    ) -> None:
        """DFS mirroring _walk_and_extract, but tracking a current
        enclosing function and emitting UnresolvedCall entries at
        invocation / object-creation / property-assign nodes.

        Calls that appear outside any function body (top-level
        statements) have ``current_function == None`` and are
        dropped -- they cannot participate in a call-graph edge.
        """
        t = node.type

        # File-scoped namespace: siblings, not children (same
        # handling as _walk_and_extract).
        if t == "compilation_unit":
            current_ns = list(namespace_stack)
            for child in node.children:
                if child.type == "file_scoped_namespace_declaration":
                    name = self._text_of_field(child, "name", source_bytes)
                    if name:
                        current_ns.append(name)
                    continue
                self._walk_and_extract_calls(
                    node=child,
                    source_bytes=source_bytes,
                    file_path=file_path,
                    namespace_stack=current_ns,
                    type_stack=type_stack,
                    current_function=current_function,
                    out=out,
                )
            return

        if t == "namespace_declaration":
            name = self._text_of_field(node, "name", source_bytes)
            new_ns = list(namespace_stack)
            if name:
                new_ns.append(name)
            for child in node.children:
                self._walk_and_extract_calls(
                    node=child,
                    source_bytes=source_bytes,
                    file_path=file_path,
                    namespace_stack=new_ns,
                    type_stack=type_stack,
                    current_function=current_function,
                    out=out,
                )
            return

        if t in _TYPE_CONTAINER_NODES:
            name = self._text_of_field(node, "name", source_bytes)
            new_type_stack = type_stack + ([name] if name else [])
            for child in node.children:
                self._walk_and_extract_calls(
                    node=child,
                    source_bytes=source_bytes,
                    file_path=file_path,
                    namespace_stack=namespace_stack,
                    type_stack=new_type_stack,
                    current_function=current_function,
                    out=out,
                )
            return

        # Entering a method-like node ESTABLISHES the enclosing
        # function context for its body. Nested local functions
        # inherit their PARENT method as in_function -- we do not
        # emit local functions as FunctionDefs (see
        # _function_def_from_method), so attributing their calls
        # to the parent is the only sensible choice.
        if t in _METHOD_LIKE_NODES and t != "local_function_statement":
            method_canonical = self._method_canonical_name(
                node=node,
                source_bytes=source_bytes,
                namespace_stack=namespace_stack,
                type_stack=type_stack,
            )
            for child in node.children:
                self._walk_and_extract_calls(
                    node=child,
                    source_bytes=source_bytes,
                    file_path=file_path,
                    namespace_stack=namespace_stack,
                    type_stack=type_stack,
                    current_function=method_canonical or current_function,
                    out=out,
                )
            return

        # Property accessor bodies. Each accessor establishes its
        # own enclosing function (get_X or set_X); we walk the
        # accessors individually so calls attribute correctly.
        if t in _PROPERTY_LIKE_NODES:
            prop_name = self._text_of_field(node, "name", source_bytes)
            if not prop_name and node.type == "indexer_declaration":
                prop_name = "Item"
            for accessor_kind, accessor_node in self._iter_accessors(node):
                resolved_kind = "set" if accessor_kind == "init" else accessor_kind
                if not prop_name:
                    continue
                member = f"{resolved_kind}_{prop_name}"
                accessor_canonical = self._assemble_canonical(
                    namespace_stack=namespace_stack,
                    type_stack=type_stack,
                    member=member,
                )
                for child in accessor_node.children:
                    self._walk_and_extract_calls(
                        node=child,
                        source_bytes=source_bytes,
                        file_path=file_path,
                        namespace_stack=namespace_stack,
                        type_stack=type_stack,
                        current_function=accessor_canonical or current_function,
                        out=out,
                    )
            return

        # ---- CALL SITE NODES ----
        if t == "invocation_expression" and current_function is not None:
            function_field = node.child_by_field_name("function")
            args_field = node.child_by_field_name("arguments")
            if function_field is not None:
                callee = source_bytes[
                    function_field.start_byte:function_field.end_byte
                ].decode("utf-8", errors="replace")
                out.append(UnresolvedCall(
                    callee_text=callee,
                    location=self._location_of(node, file_path),
                    in_function=current_function,
                    positional_arg_count=self._count_positional_args(args_field),
                ))
            # Recurse to catch chained calls: foo.Bar().Baz() has
            # an outer invocation_expression whose function is a
            # member_access_expression containing the inner call.
            for child in node.children:
                self._walk_and_extract_calls(
                    node=child,
                    source_bytes=source_bytes,
                    file_path=file_path,
                    namespace_stack=namespace_stack,
                    type_stack=type_stack,
                    current_function=current_function,
                    out=out,
                )
            return

        if t == "object_creation_expression" and current_function is not None:
            type_field = node.child_by_field_name("type")
            args_field = node.child_by_field_name("arguments")
            if type_field is not None:
                callee = source_bytes[
                    type_field.start_byte:type_field.end_byte
                ].decode("utf-8", errors="replace")
                out.append(UnresolvedCall(
                    callee_text=callee,
                    location=self._location_of(node, file_path),
                    in_function=current_function,
                    positional_arg_count=self._count_positional_args(args_field),
                    is_constructor=True,
                ))
            # Constructor args may themselves contain calls or
            # constructors -- recurse into the whole subtree.
            for child in node.children:
                self._walk_and_extract_calls(
                    node=child,
                    source_bytes=source_bytes,
                    file_path=file_path,
                    namespace_stack=namespace_stack,
                    type_stack=type_stack,
                    current_function=current_function,
                    out=out,
                )
            return

        if t == "assignment_expression" and current_function is not None:
            left = node.child_by_field_name("left")
            # Only member-access LHS is potentially a property setter.
            # Plain identifier LHS (`x = y`) and element access
            # (`x[i] = y`) don't map to property setters we can
            # resolve against the taint spec vocabulary today.
            if left is not None and left.type == "member_access_expression":
                lhs_text = source_bytes[
                    left.start_byte:left.end_byte
                ].decode("utf-8", errors="replace")
                out.append(UnresolvedCall(
                    callee_text=lhs_text,
                    location=self._location_of(node, file_path),
                    in_function=current_function,
                    # Property setters take a single value; matches
                    # `tainted_args: [value]` in the spec YAMLs.
                    positional_arg_count=1,
                    is_property_set=True,
                ))
            # RHS may itself contain calls (e.g. `x.Prop = Foo()`).
            for child in node.children:
                self._walk_and_extract_calls(
                    node=child,
                    source_bytes=source_bytes,
                    file_path=file_path,
                    namespace_stack=namespace_stack,
                    type_stack=type_stack,
                    current_function=current_function,
                    out=out,
                )
            return

        # Default recursion.
        for child in node.children:
            self._walk_and_extract_calls(
                node=child,
                source_bytes=source_bytes,
                file_path=file_path,
                namespace_stack=namespace_stack,
                type_stack=type_stack,
                current_function=current_function,
                out=out,
            )

    def _method_canonical_name(
        self,
        *,
        node,
        source_bytes: bytes,
        namespace_stack: List[str],
        type_stack: List[str],
    ) -> Optional[str]:
        """Compute the canonical name of a method-like node.

        Mirrors the naming rules in _function_def_from_method, but
        without building a FunctionDef. Kept as a separate helper
        so the walker can attribute calls without re-doing
        parameter extraction, modifier checks, etc.
        """
        if node.type == "constructor_declaration":
            member = "<init>"
        elif node.type == "destructor_declaration":
            member = "Finalize"
        elif node.type in ("operator_declaration", "conversion_operator_declaration"):
            op_text = self._text_of_field(node, "operator", source_bytes)
            member = f"op_{op_text}" if op_text else None
        else:
            member = self._text_of_field(node, "name", source_bytes)
        if not member:
            return None
        return self._assemble_canonical(
            namespace_stack=namespace_stack,
            type_stack=type_stack,
            member=member,
        )

    def _iter_accessors(self, prop_node):
        """Yield (kind, accessor_node) for each accessor on a property.

        Companion to _find_property_accessors -- that helper returns
        just the kind strings; this one returns the node too so the
        call walker can descend into each accessor with the right
        canonical name context.
        """
        results = []
        def visit(n):
            if n.type == "accessor_declaration":
                for child in n.children:
                    if child.type in _ACCESSOR_KEYWORDS:
                        results.append((child.type, n))
                        break
            for c in n.children:
                visit(c)
        visit(prop_node)
        return results

    def _count_positional_args(self, args_node) -> int:
        """Number of arguments in an argument_list, positional or named.

        Named-argument handling is left to the resolver: named args
        disrupt positional matching against ``tainted_args: [N]``,
        and the fix requires knowing the callee's parameter list,
        which the extractor does not have. Total count is still
        useful for overload disambiguation.
        """
        if args_node is None:
            return 0
        return sum(1 for c in args_node.children if c.type == "argument")

    # ------------------------------------------------------------------
    # Method-like -> FunctionDef
    # ------------------------------------------------------------------

    def _function_def_from_method(
        self,
        *,
        node,
        source_bytes: bytes,
        file_path: str,
        namespace_stack: List[str],
        type_stack: List[str],
    ) -> Optional[FunctionDef]:
        method_name: Optional[str]
        is_constructor = False
        accessor = None

        if node.type == "constructor_declaration":
            method_name = "<init>"
            is_constructor = True
        elif node.type == "destructor_declaration":
            # C# destructors compile as `Finalize()`. The tree-sitter
            # `name` field is the class name; canonicalise to
            # `Finalize` so downstream matches the CIL emit.
            method_name = "Finalize"
        elif node.type in ("operator_declaration", "conversion_operator_declaration"):
            # Operator overloads compile to `op_Addition` etc. Rather
            # than reimplement CIL naming here, use the raw operator
            # token as the name suffix (`op_+`, `op_==`) -- specs can
            # match either form via alias, and the LLM verifier will
            # bridge remaining gaps. Precision improvement is a future
            # patch.
            op_text = self._text_of_field(node, "operator", source_bytes)
            method_name = f"op_{op_text}" if op_text else None
        else:
            # method_declaration, local_function_statement
            method_name = self._text_of_field(node, "name", source_bytes)

        if not method_name:
            return None

        # Local functions live inside methods; they can't be resolved
        # by simple namespace.class.name and are rarely themselves
        # sources/sinks. Skip them for now; a later patch may emit
        # them under a synthesized parent name.
        if node.type == "local_function_statement":
            return None

        canonical = self._assemble_canonical(
            namespace_stack=namespace_stack,
            type_stack=type_stack,
            member=method_name,
        )
        if canonical is None:
            # Method outside any type -- shouldn't happen in valid C#
            # but tree-sitter can produce partial trees on broken
            # source. Drop rather than emit an unqualified name.
            return None

        return FunctionDef(
            canonical_name=canonical,
            language=self.language,
            location=self._location_of(node, file_path),
            parameters=self._extract_parameter_names(node, source_bytes),
            return_type=self._extract_return_type(node, source_bytes),
            is_constructor=is_constructor,
            is_static=self._has_modifier(node, "static", source_bytes),
            accessor=accessor,
        )

    def _function_defs_from_property(
        self,
        *,
        node,
        source_bytes: bytes,
        file_path: str,
        namespace_stack: List[str],
        type_stack: List[str],
    ) -> List[FunctionDef]:
        """Emit one FunctionDef per accessor (get_X, set_X)."""
        prop_name = self._text_of_field(node, "name", source_bytes)
        if not prop_name:
            # Indexers have no name; use `Item` (CIL convention).
            if node.type == "indexer_declaration":
                prop_name = "Item"
            else:
                return []

        accessor_kinds = self._find_property_accessors(node)
        out: List[FunctionDef] = []
        for kind in accessor_kinds:
            # `init` accessors compile to setters with a modreq. For
            # taint purposes we treat them as `set_X`.
            resolved_kind = "set" if kind == "init" else kind
            member = f"{resolved_kind}_{prop_name}"
            canonical = self._assemble_canonical(
                namespace_stack=namespace_stack,
                type_stack=type_stack,
                member=member,
            )
            if canonical is None:
                continue
            out.append(FunctionDef(
                canonical_name=canonical,
                language=self.language,
                location=self._location_of(node, file_path),
                parameters=("value",) if resolved_kind == "set" else (),
                return_type=None,
                is_constructor=False,
                is_static=self._has_modifier(node, "static", source_bytes),
                accessor=resolved_kind,
            ))
        return out

    def _find_property_accessors(self, prop_node) -> List[str]:
        """Return the set of accessor kinds ('get'/'set'/'init')
        present on this property node.

        Auto-implemented properties without an explicit accessor body
        (``public string Foo { get; set; }``) still produce
        ``accessor_declaration`` nodes with the keyword as the first
        token, so this uniform walk works for both forms.
        """
        out: List[str] = []
        def visit(n):
            if n.type == "accessor_declaration":
                # First child token is the accessor keyword.
                for child in n.children:
                    if child.type in _ACCESSOR_KEYWORDS:
                        out.append(child.type)
                        break
            for c in n.children:
                visit(c)
        visit(prop_node)
        return out

    # ------------------------------------------------------------------
    # Canonical name assembly
    # ------------------------------------------------------------------

    def _assemble_canonical(
        self,
        *,
        namespace_stack: List[str],
        type_stack: List[str],
        member: str,
    ) -> Optional[str]:
        """Build ``csharp:Ns.Sub.Class.Inner.Member``.

        Returns None when there is no enclosing type -- top-level free
        methods aren't legal C# (unless in a top-level program, which
        the grammar handles differently), and we don't try to synthesize
        a parent for them.
        """
        if not type_stack:
            return None
        parts: List[str] = []
        if namespace_stack:
            parts.extend(namespace_stack)
        parts.extend(type_stack)
        parts.append(member)
        return f"{self.language}:{'.'.join(parts)}"

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _text_of_field(self, node, field_name: str, source_bytes: bytes) -> Optional[str]:
        field = node.child_by_field_name(field_name)
        if field is None:
            return None
        return source_bytes[field.start_byte:field.end_byte].decode("utf-8", errors="replace")

    def _location_of(self, node, file_path: str) -> Location:
        return Location(
            file_path=file_path,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            col_start=node.start_point[1],
            col_end=node.end_point[1],
        )

    def _extract_parameter_names(self, method_node, source_bytes: bytes) -> Tuple[str, ...]:
        """Best-effort parameter name extraction.

        Tree-sitter's C# grammar puts params under a ``parameter_list``
        field on method nodes. Each ``parameter`` has a ``name`` field.
        Missing gracefully -- returns () rather than raising if the
        shape doesn't match expectations.
        """
        params = method_node.child_by_field_name("parameters")
        if params is None:
            return ()
        names: List[str] = []
        for child in params.children:
            if child.type != "parameter":
                continue
            name_node = child.child_by_field_name("name")
            if name_node is None:
                continue
            names.append(
                source_bytes[name_node.start_byte:name_node.end_byte]
                .decode("utf-8", errors="replace")
            )
        return tuple(names)

    def _extract_return_type(self, method_node, source_bytes: bytes) -> Optional[str]:
        """Return the text of the ``type`` / ``returns`` field, if any."""
        type_node = (
            method_node.child_by_field_name("type")
            or method_node.child_by_field_name("returns")
        )
        if type_node is None:
            return None
        return source_bytes[type_node.start_byte:type_node.end_byte].decode(
            "utf-8", errors="replace"
        )

    def _has_modifier(self, node, keyword: str, source_bytes: bytes) -> bool:
        """Look for a `modifier` child containing ``keyword``.

        Handles the tree-sitter shape where modifiers are separate
        anonymous nodes (not a `modifiers` field) directly under the
        method node.
        """
        for child in node.children:
            if child.type == "modifier":
                text = source_bytes[child.start_byte:child.end_byte].decode(
                    "utf-8", errors="replace"
                )
                if text == keyword:
                    return True
        return False

    # ------------------------------------------------------------------
    # detect_frameworks
    # ------------------------------------------------------------------

    def detect_frameworks(self, project_root: Path) -> List[str]:
        if not project_root.exists():
            return []
        detected: set[str] = set()
        # rglob may hit large trees; cap the traversal to avoid a
        # pathological run on someone's home directory. Real repos
        # rarely exceed a few dozen .csproj files.
        seen = 0
        MAX = 500
        for csproj in project_root.rglob("*.csproj"):
            seen += 1
            if seen > MAX:
                break
            try:
                text = csproj.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for pkg in _CSPROJ_PACKAGE_RE.findall(text):
                for pattern, spec in _FRAMEWORK_MAP:
                    if pattern.match(pkg):
                        detected.add(spec)
                        break
        return sorted(detected)


# `<PackageReference Include="Microsoft.AspNetCore.App" Version="8.0.0" />`
# The Include attribute is what we want; version and other attributes
# vary and are ignored.
_CSPROJ_PACKAGE_RE = re.compile(
    r'<PackageReference\s+[^>]*Include\s*=\s*"([^"]+)"',
    re.IGNORECASE,
)
