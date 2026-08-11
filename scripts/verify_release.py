"""在不运行论文实验的情况下检查发布包。"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


REQUIRED = (
    "README.md",
    "pyproject.toml",
    "CITATION.cff",
    "LICENSE_STATUS.md",
    "SOURCE_MANIFEST.sha256",
    "configs/RID_PHASE1_CASE_FOLDS.csv",
    "docs/01_项目概览.md",
    "docs/02_端到端流程.md",
    "docs/03_代码架构.md",
    "docs/04_实验复现.md",
    "docs/05_结果与边界.md",
    "docs/06_工程质量.md",
    "docs/07_项目讲解.md",
    "docs/08_发布清单.md",
    "protocol/QGRAPH_A2_PROTOCOL.json",
    "protocol/QGRAPH_A2_COMPUTE_MANIFEST.json",
    "src/structured_instance_selection/__init__.py",
    "src/structured_instance_selection/demo.py",
    "src/structured_instance_selection/rid_qgraph_core.py",
    "src/structured_instance_selection/rid_qgraph_train_group.py",
    "src/structured_instance_selection/rid_qgraph_a2_train_group.py",
    "tests/test_core_pipeline.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(root: Path) -> list[str]:
    errors: list[str] = []
    manifest = root / "SOURCE_MANIFEST.sha256"
    tracked: set[str] = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            errors.append(f"第 {line_number} 行格式错误")
            continue
        expected, relative = match.groups()
        tracked.add(relative)
        path = root / Path(relative)
        if not path.is_file():
            errors.append(f"清单文件缺失：{relative}")
        elif sha256(path) != expected:
            errors.append(f"哈希不一致：{relative}")

    ignored_parts = {"__pycache__", ".pytest_cache", ".git", "*.egg-info"}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "SOURCE_MANIFEST.sha256"
        and not any(part in ignored_parts or part.endswith(".egg-info") for part in path.parts)
        and path.suffix != ".pyc"
    }
    for relative in sorted(actual - tracked):
        errors.append(f"未登记文件：{relative}")
    return errors


def validate_markdown_links(root: Path) -> list[str]:
    errors: list[str] = []
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for document in root.rglob("*.md"):
        text = document.read_text(encoding="utf-8")
        for target in pattern.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative = target.split("#", 1)[0]
            if relative and not (document.parent / relative).is_file():
                source = document.relative_to(root).as_posix()
                errors.append(f"文档链接失效：{source} -> {target}")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    missing = [item for item in REQUIRED if not (root / item).is_file()]
    if missing:
        print("失败：缺少文件：" + ", ".join(missing))
        return 1

    for relative in (
        "protocol/QGRAPH_A2_PROTOCOL.json",
        "protocol/QGRAPH_A2_COMPUTE_MANIFEST.json",
    ):
        json.loads((root / relative).read_text(encoding="utf-8"))

    fold_text = (root / "configs/RID_PHASE1_CASE_FOLDS.csv").read_text(encoding="utf-8")
    header = fold_text.splitlines()[0].split(",")
    expected = {"protocol_id", "outer_fold", "case_id", "role"}
    if set(header) != expected:
        print(f"失败：病例划分字段错误：{header}")
        return 1

    path_pattern = re.compile(r"(?:[A-Za-z]:\\|/Users/|/home/)")
    text_extensions = {".md", ".toml", ".cff", ".py", ".json", ".csv"}
    leaked = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in text_extensions:
            if path.resolve() == Path(__file__).resolve():
                continue
            if path_pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
                leaked.append(path.relative_to(root).as_posix())
    if leaked:
        print("失败：发现本机绝对路径：" + ", ".join(leaked))
        return 1

    manifest_errors = validate_manifest(root)
    if manifest_errors:
        print("失败：文件清单校验未通过")
        for error in manifest_errors:
            print(f"- {error}")
        return 1

    link_errors = validate_markdown_links(root)
    if link_errors:
        print("失败：文档链接校验未通过")
        for error in link_errors:
            print(f"- {error}")
        return 1

    print("通过：必需文件")
    print("通过：协议 JSON")
    print("通过：病例划分字段")
    print("通过：本机路径检查")
    print("通过：SHA-256 文件清单")
    print("通过：文档链接")
    return 0


if __name__ == "__main__":
    sys.exit(main())
