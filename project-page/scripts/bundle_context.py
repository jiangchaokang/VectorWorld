#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import codecs
import fnmatch
import hashlib
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

"""
python3 scripts/bundle_context.py \
  src/components/vectorworld \
  src/paper.mdx \
  .github/workflows/astro.yml package.json src/components/Picture.astro src/components/Video.astro src/components/Wide.astro src/components/starwind/tabs/Tabs.astro src/lib/render-pdf.ts src/pages/index.astro src/styles/global.css astro.config.ts src/components/Links.astro src/components/ThemeToggle.astro \
  -o scripts/project_context.txt
"""


DEFAULT_EXCLUDE_DIRS = {
    ".git", ".hg", ".svn",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules", ".next", "dist", "build", "target", "out", "coverage",
    ".venv", "venv", "env", ".idea", ".vscode",
}

DEFAULT_EXCLUDE_GLOBS = [
    ".DS_Store",
]

TEXT_NAME_HINTS = {
    "dockerfile",
    "makefile",
    "cmakelists.txt",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    ".npmrc",
    ".yarnrc",
    ".prettierrc",
    ".eslintrc",
    ".env",
    ".env.example",
}

TEXT_EXT_HINTS = {
    ".txt", ".md", ".markdown", ".rst", ".adoc",
    ".py", ".pyi", ".ipynb",
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".properties",
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
    ".html", ".htm", ".css", ".scss", ".sass", ".less", ".svg", ".xml",
    ".vue", ".svelte", ".astro",
    ".java", ".kt", ".kts", ".groovy", ".gradle",
    ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh",
    ".m", ".mm", ".rs", ".go", ".rb", ".php", ".pl", ".pm", ".lua",
    ".swift", ".dart", ".r", ".jl",
    ".sql", ".graphql", ".gql", ".proto",
    ".csv", ".tsv",
    ".tex",
}

BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".tif", ".tiff", ".psd",
    ".pdf",
    ".zip", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".tar", ".jar", ".war",
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac",
    ".mp4", ".mov", ".avi", ".mkv", ".wmv", ".webm",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    ".exe", ".dll", ".so", ".dylib", ".class", ".pyc", ".pyo",
    ".o", ".a", ".obj", ".bin", ".dat",
    ".db", ".sqlite", ".sqlite3",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".apk", ".ipa",
}


def split_csv_args(values: list[str] | None) -> list[str]:
    result: list[str] = []
    for item in values or []:
        for part in item.split(","):
            part = part.strip()
            if part:
                result.append(part)
    return result


def parse_size(size_text: str | None) -> int | None:
    if not size_text:
        return None

    s = size_text.strip().upper()
    units = {
        "K": 1024,
        "M": 1024 ** 2,
        "G": 1024 ** 3,
    }

    if s.isdigit():
        return int(s)

    if len(s) >= 2 and s[-1] in units:
        num = float(s[:-1])
        return int(num * units[s[-1]])

    raise ValueError(f"无法解析大小参数: {size_text!r}，示例: 500K / 2M / 1G / 12345")


def display_path(path: Path, absolute_paths: bool = False) -> str:
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path.absolute()

    if absolute_paths:
        return resolved.as_posix()

    cwd = Path.cwd().resolve()
    try:
        return resolved.relative_to(cwd).as_posix()
    except ValueError:
        return resolved.as_posix()


def file_matches_globs(path: Path, patterns: list[str], absolute_paths: bool = False) -> bool:
    if not patterns:
        return False

    candidates = [
        path.name,
        path.as_posix(),
        display_path(path, absolute_paths=absolute_paths),
    ]

    lower_candidates = [c.lower() for c in candidates]

    for pattern in patterns:
        p = pattern.replace("\\", "/")
        p_lower = p.lower()
        for c, lc in zip(candidates, lower_candidates):
            if fnmatch.fnmatch(c, p) or fnmatch.fnmatch(lc, p_lower):
                return True
    return False


def is_probably_binary(sample: bytes) -> bool:
    if not sample:
        return False

    if b"\x00" in sample:
        return True

    suspicious = 0
    for b in sample:
        if b in (7, 8, 9, 10, 12, 13, 27):
            continue
        if 32 <= b <= 126:
            continue
        if b >= 128:
            continue
        suspicious += 1

    return (suspicious / len(sample)) > 0.30


def is_text_file(path: Path) -> bool:
    name_lower = path.name.lower()
    suffix_lower = path.suffix.lower()

    if suffix_lower in BINARY_EXTS:
        return False

    if name_lower in TEXT_NAME_HINTS or suffix_lower in TEXT_EXT_HINTS:
        return True

    try:
        with path.open("rb") as f:
            sample = f.read(8192)
    except OSError:
        return False

    return not is_probably_binary(sample)


def build_candidate_encodings(raw: bytes, encodings: list[str]) -> list[str]:
    candidates: list[str] = []

    if raw.startswith(codecs.BOM_UTF8):
        candidates.append("utf-8-sig")
    elif raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        candidates.append("utf-16")
    elif raw.startswith(codecs.BOM_UTF32_LE) or raw.startswith(codecs.BOM_UTF32_BE):
        candidates.append("utf-32")

    for enc in encodings:
        if enc not in candidates:
            candidates.append(enc)

    return candidates


def read_text_file(path: Path, encodings: list[str]) -> tuple[str, str, bytes]:
    raw = path.read_bytes()
    candidates = build_candidate_encodings(raw, encodings)

    for enc in candidates:
        try:
            return raw.decode(enc), enc, raw
        except (UnicodeDecodeError, LookupError):
            continue

    fallback = encodings[0] if encodings else "utf-8"
    return raw.decode(fallback, errors="replace"), f"{fallback} (errors=replace)", raw


def count_lines(text: str) -> int:
    if not text:
        return 0
    return len(text.splitlines())


def collect_files(
    inputs: list[str],
    include_globs: list[str],
    exclude_globs: list[str],
    exclude_dirs: set[str],
    follow_symlinks: bool,
    max_file_size: int | None,
    absolute_paths: bool,
) -> tuple[list[Path], Counter, list[str]]:
    stats = Counter()
    warnings: list[str] = []
    found: list[Path] = []
    seen: set[str] = set()

    def consider_file(path: Path) -> None:
        try:
            resolved_key = str(path.resolve())
        except Exception:
            resolved_key = str(path.absolute())

        if resolved_key in seen:
            stats["duplicate_skipped"] += 1
            return

        if path.is_symlink() and not follow_symlinks:
            stats["symlink_file_skipped"] += 1
            return

        if include_globs and not file_matches_globs(path, include_globs, absolute_paths=absolute_paths):
            stats["include_filtered"] += 1
            return

        if exclude_globs and file_matches_globs(path, exclude_globs, absolute_paths=absolute_paths):
            stats["exclude_filtered"] += 1
            return

        try:
            st = path.stat()
        except OSError as e:
            warnings.append(f"[WARN] 无法读取文件状态: {path} -> {e}")
            stats["stat_error"] += 1
            return

        if max_file_size is not None and st.st_size > max_file_size:
            stats["too_large_skipped"] += 1
            return

        if not is_text_file(path):
            stats["non_text_skipped"] += 1
            return

        seen.add(resolved_key)
        found.append(path)
        stats["accepted"] += 1

    for item in inputs:
        p = Path(item).expanduser()

        if not p.exists():
            warnings.append(f"[WARN] 输入不存在: {item}")
            stats["missing_input"] += 1
            continue

        if p.is_file():
            consider_file(p)
            continue

        if p.is_dir():
            for root, dirs, files in os.walk(p, followlinks=follow_symlinks):
                root_path = Path(root)

                new_dirs = []
                for d in sorted(dirs):
                    full_dir = root_path / d

                    if d in exclude_dirs:
                        stats["excluded_dir_skipped"] += 1
                        continue

                    if full_dir.is_symlink() and not follow_symlinks:
                        stats["symlink_dir_skipped"] += 1
                        continue

                    new_dirs.append(d)

                dirs[:] = new_dirs

                for filename in sorted(files):
                    consider_file(root_path / filename)
            continue

        warnings.append(f"[WARN] 既不是文件也不是目录: {item}")
        stats["invalid_input"] += 1

    found.sort(key=lambda p: display_path(p, absolute_paths=absolute_paths).lower())
    return found, stats, warnings


def write_bundle(
    output_path: Path,
    files: list[Path],
    inputs: list[str],
    stats: Counter,
    warnings: list[str],
    encodings: list[str],
    absolute_paths: bool,
    follow_symlinks: bool,
    include_globs: list[str],
    exclude_globs: list[str],
    exclude_dirs: set[str],
    max_file_size: int | None,
) -> tuple[int, list[str]]:
    write_warnings = list(warnings)
    written_count = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as out:
        out.write("CONTEXT_BUNDLE_VERSION: 1\n")
        out.write(f"GENERATED_AT: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}\n")
        out.write(f"WORKING_DIRECTORY: {Path.cwd().resolve().as_posix()}\n")
        out.write("OUTPUT_ENCODING: utf-8\n")
        out.write("\n")

        out.write("INPUTS_BEGIN\n")
        for item in inputs:
            out.write(f"- {item}\n")
        out.write("INPUTS_END\n\n")

        out.write("SETTINGS_BEGIN\n")
        out.write(f"FOLLOW_SYMLINKS: {follow_symlinks}\n")
        out.write(f"ABSOLUTE_PATHS: {absolute_paths}\n")
        out.write(f"MAX_FILE_SIZE_BYTES: {max_file_size if max_file_size is not None else 'unlimited'}\n")
        out.write(f"INCLUDE_GLOBS: {', '.join(include_globs) if include_globs else '*'}\n")
        out.write(f"EXCLUDE_GLOBS: {', '.join(exclude_globs) if exclude_globs else '(none)'}\n")
        out.write(f"EXCLUDE_DIRS: {', '.join(sorted(exclude_dirs))}\n")
        out.write(f"DECODING_CANDIDATES: {', '.join(encodings)}\n")
        out.write("SETTINGS_END\n\n")

        out.write("SUMMARY_BEGIN\n")
        out.write(f"TOTAL_ACCEPTED_FILES: {len(files)}\n")
        for key in sorted(stats.keys()):
            out.write(f"{key}: {stats[key]}\n")
        out.write("SUMMARY_END\n\n")

        if write_warnings:
            out.write("WARNINGS_BEGIN\n")
            for w in write_warnings:
                out.write(w + "\n")
            out.write("WARNINGS_END\n\n")

        out.write("FILE_INDEX_BEGIN\n")
        for idx, path in enumerate(files, start=1):
            out.write(f"{idx}\t{display_path(path, absolute_paths=absolute_paths)}\n")
        out.write("FILE_INDEX_END\n\n")

        total = len(files)
        for idx, path in enumerate(files, start=1):
            path_str = display_path(path, absolute_paths=absolute_paths)

            try:
                text, detected_encoding, raw = read_text_file(path, encodings)
            except OSError as e:
                msg = f"[WARN] 文件读取失败，已跳过: {path_str} -> {e}"
                write_warnings.append(msg)
                continue
            line_count = count_lines(text)

            out.write("=" * 10 + "\n")
            out.write("BEGIN_FILE\n")
            # out.write(f"INDEX: {idx}/{total}\n")
            out.write(f"PATH: {path_str}\n")
            # out.write(f"SIZE_BYTES: {len(raw)}\n")
            out.write(f"LINES: {line_count}\n")
            # out.write(f"ENCODING: {detected_encoding}\n")
            out.write("=" * 10 + "\n")

            out.write(text)
            if text and not text.endswith(("\n", "\r")):
                out.write("\n")

            out.write("=" * 10 + "\n")
            out.write("END_FILE\n")
            out.write(f"PATH: {path_str}\n")
            out.write("=" * 10 + "\n\n")

            written_count += 1

    return written_count, write_warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将多个文件/目录中的文本内容汇总到一个 txt 文件，便于作为 LLM 上下文。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "inputs",
        nargs="+",
        help="输入的文件或目录，可传多个",
    )
    parser.add_argument(
        "-o", "--output",
        default="llm_context_bundle.txt",
        help="输出 txt 文件路径",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="仅包含匹配这些 glob 的文件。支持多次传入或逗号分隔，例如: '*.py,*.ts,*.tsx,*.astro'",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="排除匹配这些 glob 的文件。支持多次传入或逗号分隔，例如: '*.lock,*.min.js'",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="排除目录名。支持多次传入或逗号分隔，例如: node_modules,.git,dist",
    )
    parser.add_argument(
        "--max-file-size",
        default="",
        help="单文件大小上限，例如: 500K / 2M / 1G。不传则不限",
    )
    parser.add_argument(
        "--encodings",
        default="utf-8,utf-8-sig,gb18030,cp1252",
        help="解码尝试顺序，逗号分隔",
    )
    parser.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="跟随符号链接",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="输出中使用绝对路径，而不是相对当前目录的路径",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        max_file_size = parse_size(args.max_file_size)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    include_globs = split_csv_args(args.include)
    exclude_globs = DEFAULT_EXCLUDE_GLOBS + split_csv_args(args.exclude)
    exclude_dirs = DEFAULT_EXCLUDE_DIRS | set(split_csv_args(args.exclude_dir))
    encodings = split_csv_args([args.encodings])

    files, stats, warnings = collect_files(
        inputs=args.inputs,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        exclude_dirs=exclude_dirs,
        follow_symlinks=args.follow_symlinks,
        max_file_size=max_file_size,
        absolute_paths=args.absolute_paths,
    )

    output_path = Path(args.output).expanduser()

    written_count, final_warnings = write_bundle(
        output_path=output_path,
        files=files,
        inputs=args.inputs,
        stats=stats,
        warnings=warnings,
        encodings=encodings,
        absolute_paths=args.absolute_paths,
        follow_symlinks=args.follow_symlinks,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        exclude_dirs=exclude_dirs,
        max_file_size=max_file_size,
    )

    print(f"[OK] 已输出到: {output_path.resolve()}")
    print(f"[OK] 汇总文件数: {written_count}")

    if final_warnings:
        print(f"[WARN] 警告数: {len(final_warnings)}")
        for w in final_warnings[:20]:
            print(w)
        if len(final_warnings) > 20:
            print(f"... 其余 {len(final_warnings) - 20} 条警告已写入输出文件头部。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())