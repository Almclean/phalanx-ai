from pathlib import Path

from parser import discover_doc_files, discover_files, discover_files_with_exclusions, parse_file


def test_parse_c_file_extracts_key_units(tmp_path: Path):
    source = """
    // Adds two numbers.
    int add(int a, int b) {
        return a + b;
    }

    struct Point {
        int x;
        int y;
    };

    enum Mode {
        MODE_A,
        MODE_B
    };

    typedef unsigned int u32;
    """
    path = tmp_path / "sample.c"
    path.write_text(source)

    parsed = parse_file(path)
    assert parsed is not None
    names = {u.name for u in parsed.units}
    kinds = {u.kind for u in parsed.units}

    assert "add" in names
    assert "function" in kinds
    assert "struct" in kinds
    assert "enum" in kinds
    assert "typedef" in kinds


def test_parse_cpp_file_extracts_namespace_and_template_nodes(tmp_path: Path):
    source = """
    namespace math {
    template <typename T>
    class Box {
      public:
        T value;
    };

    template <typename T>
    T twice(T v) {
        return v + v;
    }
    } // namespace math
    """
    path = tmp_path / "sample.cpp"
    path.write_text(source)

    parsed = parse_file(path)
    assert parsed is not None
    names = {u.name for u in parsed.units}
    kinds = {u.kind for u in parsed.units}

    assert "math" in names
    assert "namespace" in kinds
    assert "class" in kinds or "template" in kinds or "template function" in kinds
    assert "Box" in names or "twice" in names


def test_rust_mod_tests_wrapper_is_excluded_but_test_functions_remain(tmp_path: Path):
    source = """
    #[cfg(test)]
    mod tests {
        #[test]
        fn test_add() {
            assert_eq!(2 + 2, 4);
        }
    }

    fn prod_fn() -> i32 {
        1
    }
    """
    path = tmp_path / "lib.rs"
    path.write_text(source)

    parsed = parse_file(path)
    assert parsed is not None

    assert not any(u.kind == "module" and u.name in {"tests", "test"} for u in parsed.units)
    assert any(u.kind == "function" and u.name == "test_add" for u in parsed.units)
    assert any(u.kind == "function" and u.name == "prod_fn" for u in parsed.units)


def test_header_files_are_code_not_doc_files(tmp_path: Path):
    (tmp_path / "README.md").write_text("# Docs")
    (tmp_path / "config.json").write_text('{"x": 1}')
    (tmp_path / "types.h").write_text("int add(int a, int b);")
    (tmp_path / "types.hpp").write_text("class Foo {};")
    (tmp_path / "main.c").write_text("int main() { return 0; }")

    doc_paths = {p.name for p in discover_doc_files(tmp_path)}
    code_paths = {p.name for p in discover_files(tmp_path)}

    assert "README.md" in doc_paths
    assert "config.json" in doc_paths
    assert "types.h" not in doc_paths
    assert "types.hpp" not in doc_paths
    assert "types.h" in code_paths
    assert "types.hpp" in code_paths
    assert "main.c" in code_paths


def test_discover_files_reports_excluded_directory_paths(tmp_path: Path):
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "main.c").write_text("int main() { return 0; }")

    (tmp_path / "node_modules" / "pkg").mkdir(parents=True, exist_ok=True)
    (tmp_path / "node_modules" / "pkg" / "ignored.c").write_text("int x() { return 1; }")

    (tmp_path / "vendor" / "lib").mkdir(parents=True, exist_ok=True)
    (tmp_path / "vendor" / "lib" / "ignored2.c").write_text("int y() { return 2; }")

    files, excluded = discover_files_with_exclusions(tmp_path)

    assert {p.name for p in files} == {"main.c"}
    assert "node_modules" in excluded
    assert "vendor" in excluded


def test_discover_files_supports_relative_path_excludes(tmp_path: Path):
    (tmp_path / "src" / "external").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "external" / "skip.c").write_text("int skip(void) { return 0; }")
    (tmp_path / "src" / "keep").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "keep" / "keep.c").write_text("int keep(void) { return 1; }")

    files, excluded = discover_files_with_exclusions(tmp_path, {"src/external"})

    assert {p.name for p in files} == {"keep.c"}
    assert "src/external" in excluded
