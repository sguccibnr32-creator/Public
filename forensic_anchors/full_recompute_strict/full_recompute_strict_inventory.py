"""
Strict 100% recompute — inventory + static classifier

Inventories ALL .py under D:/ドキュメント/エントロピー (recursive),
classifies each into: executable / library / parse_error / external_excluded,
and emits inventory.json + inventory.txt for the runner to consume.

Classification:
  - executable     : has `if __name__ == "__main__":` block (top-level)
  - library        : importable module, no __main__ block
  - parse_error    : SyntaxError or other AST parse failure
  - external_excluded : matched an exclusion rule (release archives, .venv,
                       __pycache__, .git, node_modules, etc.)

This runner is part of the 5.2.M v4-class strict full recompute forensic anchor.
"""

import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("D:/ドキュメント/エントロピー")

# Directories to exclude from inventory.
# Rationale:
#   - .git/.venv/__pycache__: not source
#   - github_release/v478_release: 過去 release archive、現行 canonical の重複
#   - github_release/forensic_anchors/full_recompute*: 自己参照避け (本 anchor 自身)
#   - latex_*/arxiv_submission: 投稿パッケージの中間 dir
#   - phase_a_output / phase_b_output: 出力 .txt のみで .py 無し想定
EXCLUDE_DIR_PATTERNS = [
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    "github_release/v478_release",
    "github_release/forensic_anchors/full_recompute_strict",
    "github_release/forensic_anchors/full_recompute",  # 既存 5.2.M v3 の runner と重複避け
    "/arxiv_submission/",
    "/arxiv/",
    "/build/",
    "/dist/",
    ".egg-info",
]


def is_excluded(path: Path) -> bool:
    s = str(path).replace("\\", "/")
    return any(pat in s for pat in EXCLUDE_DIR_PATTERNS)


def has_top_level_main_block(source: str) -> tuple[bool, str | None]:
    """Return (has_main, error_or_None)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}"
    except Exception as e:
        return False, f"AST error: {type(e).__name__}: {e}"

    for node in tree.body:
        if isinstance(node, ast.If):
            test = node.test
            # Match: __name__ == "__main__"
            if isinstance(test, ast.Compare) and len(test.comparators) == 1:
                left = test.left
                comp = test.comparators[0]
                if (
                    isinstance(left, ast.Name) and left.id == "__name__"
                    and isinstance(comp, ast.Constant) and comp.value == "__main__"
                ):
                    return True, None
                # Also handle the reversed form: "__main__" == __name__
                if (
                    isinstance(comp, ast.Name) and comp.id == "__name__"
                    and isinstance(left, ast.Constant) and left.value == "__main__"
                ):
                    return True, None
    return False, None


def classify_one(path: Path) -> dict:
    rec = {
        "path": str(path).replace("\\", "/"),
        "size": path.stat().st_size,
        "sha256": None,
        "category": None,
        "error": None,
    }
    try:
        raw = path.read_bytes()
        rec["sha256"] = hashlib.sha256(raw).hexdigest()
    except Exception as e:
        rec["category"] = "io_error"
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec

    try:
        source = raw.decode("utf-8-sig")  # auto-strip BOM if present
    except UnicodeDecodeError:
        try:
            source = raw.decode("cp932")
        except UnicodeDecodeError as e:
            rec["category"] = "encoding_error"
            rec["error"] = f"{type(e).__name__}: {e}"
            return rec
    # also strip any zero-width BOM that survived decode (defensive)
    if source.startswith("﻿"):
        source = source[1:]

    has_main, err = has_top_level_main_block(source)
    if err:
        rec["category"] = "parse_error"
        rec["error"] = err
    elif has_main:
        rec["category"] = "executable"
    else:
        rec["category"] = "library"
    return rec


def main() -> int:
    print(f"[strict-inventory] scanning {ROOT}", flush=True)
    all_py = []
    excluded_count = 0
    for p in ROOT.rglob("*.py"):
        if is_excluded(p):
            excluded_count += 1
            continue
        all_py.append(p)
    all_py.sort()
    print(f"[strict-inventory] candidate files: {len(all_py)} (excluded: {excluded_count})", flush=True)

    records = []
    for i, p in enumerate(all_py, 1):
        rec = classify_one(p)
        records.append(rec)
        if i % 200 == 0:
            print(f"  classified {i}/{len(all_py)}", flush=True)

    summary = {
        "root": str(ROOT).replace("\\", "/"),
        "total": len(records),
        "by_category": {},
    }
    for cat in ("executable", "library", "parse_error", "encoding_error", "io_error"):
        summary["by_category"][cat] = sum(1 for r in records if r["category"] == cat)

    inv_dir = Path(__file__).parent
    json_path = inv_dir / "full_recompute_strict_inventory.json"
    txt_path = inv_dir / "full_recompute_strict_inventory.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"# Strict full-recompute inventory\n")
        f.write(f"root: {summary['root']}\n")
        f.write(f"total: {summary['total']}\n")
        for cat, n in summary["by_category"].items():
            f.write(f"  {cat}: {n}\n")
        f.write("\n# Per-file records\n")
        for r in records:
            f.write(f"{r['category']:16s} {r['sha256'][:12] if r['sha256'] else '------------':12s}  {r['path']}\n")
            if r["error"]:
                f.write(f"                                     ERROR: {r['error']}\n")

    print(f"[strict-inventory] wrote {json_path.name} + {txt_path.name}", flush=True)
    print(f"[strict-inventory] summary: {summary['by_category']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
