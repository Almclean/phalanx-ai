"""
parser.py — AST-based code unit extraction for Python, TypeScript, Rust, Go, C, C++.

Extracts functions, classes, structs, impls, interfaces etc. as discrete
"code units" that become the L1 summarization targets.

Also handles doc comment extraction (preceding sibling strategy) for
Rust ///, Go //, TypeScript JSDoc /** */, and Python docstrings.

Text/doc file discovery is handled by discover_doc_files().
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import re

from tree_sitter import Language, Parser, Node
import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp
import tree_sitter_python as tspython
import tree_sitter_typescript as tsts
import tree_sitter_rust as tsrust
import tree_sitter_go as tsgo


# ---------------------------------------------------------------------------
# Language registry
# ---------------------------------------------------------------------------

_LANGUAGES: dict[str, Language] = {
    "c": Language(tsc.language()),
    "cpp": Language(tscpp.language()),
    "python": Language(tspython.language()),
    "typescript": Language(tsts.language_typescript()),
    "tsx": Language(tsts.language_tsx()),
    "rust": Language(tsrust.language()),
    "go": Language(tsgo.language()),
}

_EXT_MAP: dict[str, str] = {
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".mts": "typescript",
    ".cts": "typescript",
    ".js": "typescript",
    ".mjs": "typescript",
    ".cjs": "typescript",
    ".rs": "rust",
    ".go": "go",
}

# Text/doc file extensions worth summarizing separately
_DOC_EXTENSIONS: set[str] = {
    ".md", ".mdx", ".rst", ".txt",
    ".yaml", ".yml",
    ".toml",
    ".json",
}

_DOC_SKIP_PATTERNS: set[str] = {
    "package-lock.json", "yarn.lock", "Cargo.lock",
    "poetry.lock", "composer.lock",
    "CHANGELOG.md", "CHANGES.md",
}

_TOP_LEVEL_NODES: dict[str, set[str]] = {
    "c": {
        "function_definition",
        "struct_specifier",
        "enum_specifier",
        "type_definition",
    },
    "cpp": {
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "enum_specifier",
        "namespace_definition",
        "template_declaration",
        "function_template_declaration",
        "type_definition",
    },
    "python": {
        "function_definition",
        "async_function_definition",
        "class_definition",
        "decorated_definition",
    },
    "typescript": {
        "function_declaration",
        "function_expression",
        "arrow_function",
        "class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "export_statement",
    },
    "tsx": {
        "function_declaration",
        "function_expression",
        "arrow_function",
        "class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "export_statement",
    },
    "rust": {
        "function_item",
        "struct_item",
        "enum_item",
        "impl_item",
        "trait_item",
        "type_item",
        "mod_item",
        "macro_definition",
    },
    "go": {
        "function_declaration",
        "method_declaration",
        "type_declaration",
        "const_declaration",
        "var_declaration",
    },
}

_COMMENT_NODES: dict[str, set[str]] = {
    "c": {"comment"},
    "cpp": {"comment"},
    "python": {"comment"},
    "typescript": {"comment"},
    "tsx": {"comment"},
    "rust": {"line_comment", "block_comment"},
    "go": {"comment"},
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CodeUnit:
    name: str
    kind: str
    source: str
    language: str
    file_path: str
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    doc_comment: Optional[str] = None
    parent_name: Optional[str] = None


@dataclass
class FileUnits:
    path: str
    language: str
    units: list[CodeUnit] = field(default_factory=list)
    raw_source: str = ""
    parse_error: bool = False


@dataclass
class DocFile:
    path: str
    extension: str
    content: str
    size_bytes: int


# ---------------------------------------------------------------------------
# Comment extraction
# ---------------------------------------------------------------------------

def _extract_preceding_comment(node: Node, source_bytes: bytes, language: str) -> Optional[str]:
    parent = node.parent
    if parent is None:
        return None

    comment_types = _COMMENT_NODES.get(language, set())
    if not comment_types:
        return None

    siblings = list(parent.children)
    try:
        idx = siblings.index(node)
    except ValueError:
        return None

    comments: list[str] = []
    for i in range(idx - 1, -1, -1):
        sib = siblings[i]
        if sib.type in comment_types:
            text = source_bytes[sib.start_byte:sib.end_byte].decode("utf-8", errors="replace")
            comments.insert(0, text)
        elif sib.is_named:
            break

    if not comments:
        return None

    raw = "\n".join(comments)

    # For Rust: filter to only /// doc comments (not plain // comments)
    if language == "rust":
        doc_lines = [l for l in raw.splitlines() if re.match(r'^\s*///', l)]
        if not doc_lines:
            return None
        raw = "\n".join(doc_lines)

    # Clean up markers: ///, //, #, /**, */
    cleaned = re.sub(r'^\s*(/{1,3}!?|#+|\*+)\s?', '', raw, flags=re.MULTILINE)
    cleaned = re.sub(r'/\*+|\*+/', '', cleaned)
    cleaned = re.sub(r'@\w+\s+\S+', '', cleaned)  # strip @param, @returns etc
    cleaned = cleaned.strip()
    return cleaned if len(cleaned) > 5 else None


def _extract_python_docstring(node: Node, source_bytes: bytes) -> Optional[str]:
    for child in node.children:
        if child.type == "block":
            for stmt in child.children:
                if stmt.type == "expression_statement":
                    for sub in stmt.children:
                        if sub.type in ("string", "concatenated_string"):
                            raw = source_bytes[sub.start_byte:sub.end_byte].decode("utf-8", errors="replace")
                            cleaned = re.sub(r'^["\'\s]+|["\'\s]+$', '', raw).strip()
                            return cleaned if cleaned else None
                break
    return None


# ---------------------------------------------------------------------------
# Name extraction
# ---------------------------------------------------------------------------

def _find_identifier_text(node: Node, source_bytes: bytes) -> Optional[str]:
    """Return the first identifier-like token in a subtree."""
    target_types = {
        "identifier",
        "type_identifier",
        "field_identifier",
        "namespace_identifier",
        "name",
    }
    stack = [node]
    while stack:
        current = stack.pop()
        for child in current.children:
            if child.type in target_types:
                return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
            stack.append(child)
    return None


def _get_node_name(node: Node, source_bytes: bytes) -> str:
    if node.type == "function_definition":
        for child in node.children:
            if child.type in ("function_declarator", "pointer_declarator", "reference_declarator"):
                name = _find_identifier_text(child, source_bytes)
                if name:
                    return name

    if node.type == "namespace_definition":
        for child in node.children:
            if child.type in ("namespace_identifier", "identifier", "name"):
                return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")

    if node.type in ("template_declaration", "function_template_declaration"):
        for child in node.children:
            if child.type == "template_parameter_list":
                continue
            name = _find_identifier_text(child, source_bytes)
            if name:
                return name

    if node.type == "type_declaration":
        for child in node.children:
            if child.type == "type_spec":
                for sub in child.children:
                    if sub.type in ("type_identifier", "identifier"):
                        return source_bytes[sub.start_byte:sub.end_byte].decode("utf-8", errors="replace")

    if node.type == "method_declaration":
        found_receiver = False
        for child in node.children:
            if child.type == "parameter_list" and not found_receiver:
                found_receiver = True
                continue
            if found_receiver and child.type in ("field_identifier", "identifier"):
                return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
        identifiers = [c for c in node.children if c.type in ("identifier", "field_identifier", "type_identifier")]
        if len(identifiers) >= 2:
            return source_bytes[identifiers[1].start_byte:identifiers[1].end_byte].decode("utf-8", errors="replace")

    for child in node.children:
        if child.type in ("identifier", "name", "type_identifier", "field_identifier"):
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")

    text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
    m = re.search(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', text)
    return m.group(1) if m else "<anonymous>"


def _get_node_kind(node_type: str) -> str:
    mapping = {
        "function_definition": "function",
        "async_function_definition": "async function",
        "function_declaration": "function",
        "function_expression": "function",
        "arrow_function": "arrow function",
        "class_definition": "class",
        "class_declaration": "class",
        "class_specifier": "class",
        "struct_specifier": "struct",
        "namespace_definition": "namespace",
        "template_declaration": "template",
        "function_template_declaration": "template function",
        "interface_declaration": "interface",
        "type_alias_declaration": "type alias",
        "enum_declaration": "enum",
        "enum_specifier": "enum",
        "enum_item": "enum",
        "export_statement": "export",
        "function_item": "function",
        "struct_item": "struct",
        "impl_item": "impl block",
        "trait_item": "trait",
        "type_item": "type alias",
        "mod_item": "module",
        "macro_definition": "macro",
        "method_declaration": "method",
        "type_declaration": "type",
        "const_declaration": "const",
        "var_declaration": "var",
        "type_definition": "typedef",
        "decorated_definition": "decorated definition",
    }
    return mapping.get(node_type, node_type)


# ---------------------------------------------------------------------------
# AST traversal
# ---------------------------------------------------------------------------

def _collect_units(
    node: Node,
    source_bytes: bytes,
    language: str,
    file_path: str,
    units: list[CodeUnit],
    parent_name: Optional[str] = None,
):
    target_types = _TOP_LEVEL_NODES.get(language, set())

    if node.type in target_types:
        name = _get_node_name(node, source_bytes)
        kind = _get_node_kind(node.type)
        source = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

        if len(source) > 8000:
            source = source[:8000] + "\n... [truncated]"

        docstring = _extract_python_docstring(node, source_bytes) if language == "python" else None
        doc_comment = _extract_preceding_comment(node, source_bytes, language)

        # Rust test wrappers (`mod tests` / `mod test`) are large containers that
        # duplicate the enclosed test functions; skip the wrapper unit itself.
        is_rust_test_wrapper = (
            language == "rust"
            and kind == "module"
            and name.lower() in {"tests", "test"}
        )

        if not is_rust_test_wrapper:
            unit = CodeUnit(
                name=name,
                kind=kind,
                source=source,
                language=language,
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                docstring=docstring,
                doc_comment=doc_comment,
                parent_name=parent_name,
            )
            units.append(unit)

        if node.type in (
            "class_definition", "class_declaration", "impl_item",
            "trait_item", "mod_item", "decorated_definition",
            "namespace_definition", "template_declaration", "function_template_declaration",
        ):
            for child in node.children:
                _collect_units(child, source_bytes, language, file_path, units, name)
        return

    for child in node.children:
        _collect_units(child, source_bytes, language, file_path, units, parent_name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_file(path: Path) -> Optional[FileUnits]:
    ext = path.suffix.lower()
    language = _EXT_MAP.get(ext)
    if language is None:
        return None

    lang_obj = _LANGUAGES.get(language)
    if lang_obj is None:
        return None

    try:
        source_bytes = path.read_bytes()
    except (OSError, PermissionError):
        return None

    parser = Parser(lang_obj)
    try:
        tree = parser.parse(source_bytes)
    except Exception:
        return FileUnits(
            path=str(path), language=language, parse_error=True,
            raw_source=source_bytes.decode("utf-8", errors="replace"),
        )

    units: list[CodeUnit] = []
    _collect_units(tree.root_node, source_bytes, language, str(path), units)

    return FileUnits(
        path=str(path), language=language, units=units,
        raw_source=source_bytes.decode("utf-8", errors="replace"),
    )


def parse_doc_file(path: Path, max_size_bytes: int = 100_000) -> Optional[DocFile]:
    """Read a documentation/config file. Returns None if too large, binary, or skippable."""
    if path.name in _DOC_SKIP_PATTERNS:
        return None
    if path.suffix.lower() not in _DOC_EXTENSIONS:
        return None
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size > max_size_bytes or size == 0:
        return None
    try:
        content = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeDecodeError):
        return None
    return DocFile(path=str(path), extension=path.suffix.lower(), content=content, size_bytes=size)


_DEFAULT_EXCLUDE_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__",
    ".venv", "venv", "env", "dist", "build", "target",
    ".next", ".nuxt", "vendor", "third_party", "generated",
    ".mypy_cache", ".pytest_cache", ".tox", "coverage", ".cargo",
})


def discover_files(root: Path, extra_excludes: set[str] | None = None) -> list[Path]:
    excludes = _DEFAULT_EXCLUDE_DIRS | (extra_excludes or set())
    files: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in excludes for part in p.parts):
            continue
        if p.suffix.lower() in _EXT_MAP:
            files.append(p)
    return sorted(files)


def discover_doc_files(root: Path, extra_excludes: set[str] | None = None) -> list[Path]:
    excludes = _DEFAULT_EXCLUDE_DIRS | (extra_excludes or set())
    files: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in excludes for part in p.parts):
            continue
        if p.name in _DOC_SKIP_PATTERNS:
            continue
        if p.suffix.lower() in _DOC_EXTENSIONS:
            files.append(p)
    return sorted(files)
