#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import keyword
import os
import re
import shutil
import subprocess
import sys
import tokenize
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


CJK_RE = re.compile(
    r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\u3000-\u303F\uFF00-\uFFEF]"
)
ENCODING_COOKIE_RE = re.compile(r"coding[:=]\s*([-\w.]+)")

CACHE_VERSION = "zh2en_repo_v2_raw_model_output"

TOP_LEVEL_IGNORE_DEFAULT = {
    ".git",
    ".hg",
    ".svn",
    ".zh2en_work",
    "outputs",
    "build",
    "dist",
}

ANYWHERE_IGNORE_DEFAULT = {
    "__pycache__",
    ".venv",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "node_modules",
    ".idea",
    ".vscode",
    ".ipynb_checkpoints",
    "wandb",
    "checkpoints",
}

TEXT_EXTS = {".md", ".rst", ".txt"}
YAML_EXTS = {".yaml", ".yml"}
SHELL_EXTS = {".sh", ".bash"}
PYTHON_EXTS = {".py", ".pyi"}

MAX_SCAN_FILE_SIZE = 5 * 1024 * 1024  # 5MB

SAFE_YAML_VALUE_KEYS = {
    "description",
    "desc",
    "help",
    "comment",
    "comments",
    "note",
    "notes",
    "summary",
    "title",
    "caption",
    "message",
    "messages",
    "usage",
    "example",
    "examples",
    "epilog",
    "doc",
    "docs",
}

SAFE_PY_STRING_KW_NAMES = set(SAFE_YAML_VALUE_KEYS) | {
    "label",
    "prompt",
}

SAFE_PY_FIRST_ARG_CALL_TAILS = {
    "print",
    "debug",
    "info",
    "warning",
    "warn",
    "error",
    "exception",
    "critical",
    "set_description",
    "set_description_str",
}

SAFE_EXCEPTION_TAILS = {
    "Exception",
    "RuntimeError",
    "ValueError",
    "TypeError",
    "KeyError",
    "IndexError",
    "AssertionError",
    "NotImplementedError",
    "ImportError",
    "FileNotFoundError",
}

PROTECT_PATTERNS = [
    re.compile(r"`[^`\n]+`"),
    re.compile(r"\$\{[^}\n]+\}|\$[A-Za-z_][A-Za-z0-9_]*"),
    re.compile(r"%\([^)]+\)[#0\- +]?(?:\d+|\*)?(?:\.(?:\d+|\*))?[hlL]?[diouxXeEfFgGcrs]"),
    re.compile(r"%[#0\- +]?(?:\d+|\*)?(?:\.(?:\d+|\*))?[hlL]?[diouxXeEfFgGcrs]"),
    re.compile(r"\{[^{}\n]*\}"),
    re.compile(r"--?[A-Za-z0-9][A-Za-z0-9_-]*"),
    re.compile(r"(?:[A-Za-z_][A-Za-z0-9_]*)(?:[./:][A-Za-z0-9_./:-]+)+"),
    re.compile(r"[A-Za-z_][A-Za-z0-9_]*\([^()\n]*\)"),
    re.compile(r"\b[A-Z_][A-Z0-9_]{2,}\b"),
    re.compile(r"https?://\S+"),
]

DIRECTIVE_PREFIX_REGEXES = [
    r"type:\s*ignore(?:\[[^\]]+\])?",
    r"noqa(?:\s*:\s*[A-Z0-9,\s-]+)?",
    r"pylint:\s*[A-Za-z0-9_,=\-\s]+",
    r"fmt:\s*(?:off|on|skip)",
    r"yapf:\s*(?:disable|enable)",
    r"isort:\s*[A-Za-z0-9_,=\-\s]+",
    r"pyright:\s*[A-Za-z0-9_,=\-\s]+",
    r"mypy:\s*[A-Za-z0-9_,=\-\s]+",
    r"pragma:\s*no\s+cover",
    r"ruff:\s*[A-Za-z0-9_,=\-\s]+",
    r"nosec(?:\s+[A-Z0-9]+)?",
    r"shellcheck\s+[A-Za-z0-9_,=\-\s]+",
]

_CHAR_PUNCT_MAP = {
    "，": ", ",
    "。": ". ",
    "：": ": ",
    "；": "; ",
    "（": "(",
    "）": ")",
    "【": "[",
    "】": "]",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "、": ", ",
    "！": "! ",
    "？": "? ",
    "《": '"',
    "》": '"',
    "　": " ",
    "…": "...",
}
if any(len(k) != 1 for k in _CHAR_PUNCT_MAP):
    raise ValueError("All punctuation map keys must be exactly one character.")

PUNCT_TRANSLATE_TABLE = str.maketrans(_CHAR_PUNCT_MAP)


@dataclass(frozen=True)
class Replacement:
    start: int
    end: int
    new_text: str


@dataclass(frozen=True)
class DocstringOccurrence:
    start: int
    end: int
    value: str


@dataclass(frozen=True)
class IdentifierDefinition:
    line: int
    col: int
    old_name: str
    kind: str


@dataclass(frozen=True)
class PythonStringOccurrence:
    start: int
    end: int
    line: int
    col: int
    value: str
    reason: str


def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def detect_newline(raw: bytes) -> str:
    if b"\r\n" in raw:
        return "\r\n"
    if b"\r" in raw:
        return "\r"
    return "\n"


def read_python_text(path: Path) -> Tuple[str, str, str]:
    raw = path.read_bytes()
    encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
    text = raw.decode(encoding)
    return normalize_newlines(text), encoding, detect_newline(raw)


def read_utf8_text(path: Path) -> Tuple[str, str, str]:
    raw = path.read_bytes()
    if b"\x00" in raw:
        raise UnicodeDecodeError("utf-8", raw, 0, 1, "binary file")
    for enc in ("utf-8", "utf-8-sig"):
        try:
            return normalize_newlines(raw.decode(enc)), enc, detect_newline(raw)
        except UnicodeDecodeError:
            pass
    raise UnicodeDecodeError("utf-8", raw, 0, 1, f"cannot decode {path}")


def write_text_atomic(path: Path, text: str, encoding: str, newline: str) -> None:
    text = normalize_newlines(text)
    if newline != "\n":
        text = text.replace("\n", newline)
    data = text.encode(encoding)
    tmp = path.with_name(path.name + ".zh2en_tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def build_line_index(text: str) -> Tuple[List[str], List[int]]:
    lines = text.splitlines(keepends=True)
    if not lines:
        lines = [""]
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    return lines, starts


def lc_to_offset(line_starts: Sequence[int], line: int, col: int) -> int:
    max_line = len(line_starts)
    if line < 1 or line > max_line:
        raise ValueError(f"line out of range: {line}")
    base = line_starts[line - 1]
    if line == max_line:
        if col != 0:
            raise ValueError(f"invalid EOF column: {col}")
        return base
    next_base = line_starts[line]
    if base + col > next_base:
        raise ValueError(f"column out of range: line={line}, col={col}")
    return base + col


def utf8_byte_col_to_char_col(line_text: str, byte_col: int) -> int:
    encoded = line_text.encode("utf-8")
    if byte_col < 0 or byte_col > len(encoded):
        raise ValueError(f"byte column out of range: {byte_col}")
    try:
        return len(encoded[:byte_col].decode("utf-8"))
    except UnicodeDecodeError:
        return len(encoded[:byte_col].decode("utf-8", errors="ignore"))


def ast_node_to_char_offsets(
    lines: Sequence[str],
    line_starts: Sequence[int],
    node: ast.AST,
) -> Optional[Tuple[int, int]]:
    lineno = getattr(node, "lineno", None)
    col_offset = getattr(node, "col_offset", None)
    end_lineno = getattr(node, "end_lineno", None)
    end_col_offset = getattr(node, "end_col_offset", None)

    if None in (lineno, col_offset, end_lineno, end_col_offset):
        return None
    if not (1 <= lineno <= len(lines) and 1 <= end_lineno <= len(lines)):
        return None

    start_char_col = utf8_byte_col_to_char_col(lines[lineno - 1], int(col_offset))
    end_char_col = utf8_byte_col_to_char_col(lines[end_lineno - 1], int(end_col_offset))
    start = line_starts[lineno - 1] + start_char_col
    end = line_starts[end_lineno - 1] + end_char_col
    return start, end


def ast_node_to_char_lc(lines: Sequence[str], node: ast.AST) -> Optional[Tuple[int, int]]:
    lineno = getattr(node, "lineno", None)
    col_offset = getattr(node, "col_offset", None)
    if lineno is None or col_offset is None:
        return None
    if not (1 <= lineno <= len(lines)):
        return None
    col = utf8_byte_col_to_char_col(lines[lineno - 1], int(col_offset))
    return int(lineno), col


def apply_replacements(text: str, replacements: List[Replacement]) -> str:
    if not replacements:
        return text
    replacements = sorted(replacements, key=lambda x: (x.start, x.end))
    for i in range(1, len(replacements)):
        if replacements[i].start < replacements[i - 1].end:
            raise ValueError("overlapping replacements detected")
    out = text
    for rep in reversed(replacements):
        out = out[:rep.start] + rep.new_text + out[rep.end:]
    return out


def split_line_ending(line: str) -> Tuple[str, str]:
    if line.endswith("\n"):
        return line[:-1], "\n"
    return line, ""


def preview_text(text: str, limit: int = 200) -> str:
    text = text.replace("\n", "\\n").replace("\r", "\\r")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def normalize_english_punctuation(text: str) -> str:
    out = text.translate(PUNCT_TRANSLATE_TABLE)
    out = out.replace("\u00A0", " ")
    out = re.sub(r"[ \t]+([,.;:!?])", r"\1", out)
    out = re.sub(r"([(\[])\s+", r"\1", out)
    out = re.sub(r"\s+([)\]])", r"\1", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r" *\n", "\n", out)
    return out


def merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def load_term_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--term-map must be a JSON object: {source_term: target_term}")
    out: Dict[str, str] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError("--term-map keys and values must both be strings")
        if k:
            out[k] = v
    return out


def apply_term_map(text: str, term_map: Dict[str, str]) -> str:
    out = text
    for src, dst in sorted(term_map.items(), key=lambda kv: len(kv[0]), reverse=True):
        if src and src in out:
            out = out.replace(src, dst)
    return out


def collect_protected_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for pat in PROTECT_PATTERNS:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end()))
    return merge_spans(spans)


def chunked(seq: Sequence[str], n: int) -> Iterator[Sequence[str]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


class OfflineTranslator:
    def __init__(
        self,
        model_dir: Path,
        cache_path: Path,
        term_map: Optional[Dict[str, str]] = None,
        batch_size: int = 8,
        num_beams: int = 4,
        max_source_length: int = 256,
        max_new_tokens: int = 256,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
    ):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as e:
            raise RuntimeError(
                "Missing dependencies. Install: pip install torch transformers sentencepiece"
            ) from e

        self.torch = torch
        self.model_dir = model_dir.resolve()
        self.cache_path = cache_path
        self.term_map = term_map or {}
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.max_source_length = max_source_length
        self.max_new_tokens = max_new_tokens
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.cache: Dict[str, str] = {}

        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.cache = {str(k): str(v) for k, v in data.items()}
            except Exception:
                self.cache = {}

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_dir), local_files_only=True)

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.src_lang and hasattr(self.tokenizer, "src_lang"):
            try:
                self.tokenizer.src_lang = self.src_lang
            except Exception:
                pass

        self.forced_bos_token_id: Optional[int] = None
        if self.tgt_lang and hasattr(self.tokenizer, "lang_code_to_id"):
            try:
                lang_code_to_id = getattr(self.tokenizer, "lang_code_to_id", {})
                if self.tgt_lang in lang_code_to_id:
                    self.forced_bos_token_id = int(lang_code_to_id[self.tgt_lang])
            except Exception:
                self.forced_bos_token_id = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def _cache_key(self, text: str) -> str:
        return json.dumps(
            {
                "cache_version": CACHE_VERSION,
                "model_dir": str(self.model_dir),
                "src_lang": self.src_lang,
                "tgt_lang": self.tgt_lang,
                "text": text,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def translate_many(self, texts: Sequence[str]) -> List[str]:
        if not texts:
            return []

        results: List[Optional[str]] = [None] * len(texts)
        pending: List[Tuple[int, str, str]] = []

        for i, text in enumerate(texts):
            if not has_cjk(text):
                results[i] = text
                continue
            key = self._cache_key(text)
            cached = self.cache.get(key)
            if cached is not None:
                results[i] = cached
                continue
            pending.append((i, text, key))

        for batch in chunked([p[1] for p in pending], max(1, self.batch_size)):
            inputs = self.tokenizer(
                list(batch),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_source_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            gen_kwargs = {
                "num_beams": self.num_beams,
                "max_new_tokens": self.max_new_tokens,
                "early_stopping": True,
            }
            if self.forced_bos_token_id is not None:
                gen_kwargs["forced_bos_token_id"] = self.forced_bos_token_id

            with self.torch.inference_mode():
                outputs = self.model.generate(**inputs, **gen_kwargs)

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded = [x.strip() for x in decoded]

            start_index = 0
            while start_index < len(pending):
                break

            # write back in the same pending order for this chunk
            chunk_pending = pending[: len(batch)]
            pending = pending[len(batch) :]
            for (idx, src_text, key), translated in zip(chunk_pending, decoded):
                if not translated:
                    translated = src_text
                self.cache[key] = translated
                results[idx] = translated

        return [r if r is not None else texts[i] for i, r in enumerate(results)]

    def translate_text(self, text: str) -> str:
        return self.translate_many([text])[0]


def translate_plain_fragment(text: str, translator: OfflineTranslator) -> str:
    if not has_cjk(text):
        return text
    working = apply_term_map(text, translator.term_map)
    if not has_cjk(working):
        return normalize_english_punctuation(working)
    translated = translator.translate_text(working).strip()
    if not translated:
        return text
    return normalize_english_punctuation(translated)


def translate_mixed_text(text: str, translator: OfflineTranslator) -> str:
    if not has_cjk(text):
        return text

    spans = collect_protected_spans(text)
    if not spans:
        return translate_plain_fragment(text, translator)

    out: List[str] = []
    last = 0
    for s, e in spans:
        if last < s:
            out.append(translate_plain_fragment(text[last:s], translator))
        out.append(text[s:e])
        last = e
    if last < len(text):
        out.append(translate_plain_fragment(text[last:], translator))

    joined = "".join(out)
    joined = re.sub(r"[ \t]{2,}", " ", joined)
    joined = re.sub(r" *\n", "\n", joined)
    return joined


def translate_line_bodies(bodies: Sequence[str], translator: OfflineTranslator) -> List[str]:
    results = list(bodies)
    for i, body in enumerate(bodies):
        m = re.match(r"^([ \t]*)(.*?)([ \t]*)$", body, re.S)
        if m:
            lead, core, tail = m.groups()
        else:
            lead, core, tail = "", body, ""
        if not has_cjk(core):
            continue
        results[i] = lead + translate_mixed_text(core, translator) + tail
    return results


def translate_line_body(body: str, translator: OfflineTranslator) -> str:
    return translate_line_bodies([body], translator)[0]


def translate_multiline_text(text: str, translator: OfflineTranslator) -> str:
    if not has_cjk(text):
        return text
    lines = text.splitlines(keepends=True)
    if not lines:
        return text
    bodies: List[str] = []
    eols: List[str] = []
    for line in lines:
        body, eol = split_line_ending(line)
        bodies.append(body)
        eols.append(eol)
    translated_bodies = translate_line_bodies(bodies, translator)
    return "".join(b + e for b, e in zip(translated_bodies, eols))


def translate_docstring_text(text: str, translator: OfflineTranslator) -> str:
    if not has_cjk(text):
        return text

    lines = text.splitlines(keepends=True)
    if not lines:
        return text

    out: List[str] = []
    buffer_bodies: List[str] = []
    buffer_eols: List[str] = []
    inside_fence = False
    fence_marker: Optional[str] = None

    def flush_buffer() -> None:
        if not buffer_bodies:
            return
        translated = translate_line_bodies(buffer_bodies, translator)
        for body, eol in zip(translated, buffer_eols):
            out.append(body + eol)
        buffer_bodies.clear()
        buffer_eols.clear()

    for line in lines:
        body, eol = split_line_ending(line)
        stripped = body.lstrip()

        if re.match(r"^(```|~~~)", stripped):
            flush_buffer()
            marker = stripped[:3]
            if not inside_fence:
                inside_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                inside_fence = False
                fence_marker = None
            out.append(body + eol)
            continue

        if stripped.startswith(">>>") or stripped.startswith("..."):
            flush_buffer()
            out.append(body + eol)
            continue

        if inside_fence:
            flush_buffer()
            out.append(body + eol)
            continue

        buffer_bodies.append(body)
        buffer_eols.append(eol)

    flush_buffer()
    return "".join(out)


def split_directive_prefix(body: str) -> Tuple[str, Optional[str]]:
    stripped = body.strip()
    if not stripped:
        return "", body
    for pat in DIRECTIVE_PREFIX_REGEXES:
        m = re.match(rf"^({pat})(\s+)(.*)$", body, re.I)
        if m and m.group(3):
            return m.group(1) + m.group(2), m.group(3)
        if re.fullmatch(pat, stripped, re.I):
            return body, None
    return "", body


def translate_hash_comment(comment: str, translator: OfflineTranslator) -> str:
    m = re.match(r"^(#+)(\s*)(.*)$", comment, re.S)
    if not m:
        return comment
    hashes, space, body = m.groups()
    preserved, translatable = split_directive_prefix(body)
    if translatable is None or not has_cjk(translatable):
        return comment
    translated = translate_line_body(translatable, translator).strip()
    if not translated:
        return comment
    return f"{hashes}{space}{preserved}{translated}"


def python_docstring_literal(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"""', '\\"""')
    return f'"""{escaped}"""'


def quote_python_string(text: str) -> str:
    if "\n" in text:
        return python_docstring_literal(text)
    return json.dumps(text, ensure_ascii=False)


def english_words(text: str) -> List[str]:
    text = normalize_english_punctuation(text)
    return re.findall(r"[A-Za-z0-9]+", text)


def build_identifier(words: Sequence[str], kind: str) -> str:
    if kind == "class":
        ident = "".join(w[:1].upper() + w[1:] for w in words) if words else "RenamedClass"
        if ident and ident[0].isdigit():
            ident = "Class" + ident
        return ident or "RenamedClass"

    if kind == "constant":
        ident = "_".join(w.upper() for w in words) if words else "RENAMED_CONSTANT"
        ident = re.sub(r"_+", "_", ident).strip("_") or "RENAMED_CONSTANT"
        if ident[0].isdigit():
            ident = "CONST_" + ident
        return ident

    ident = "_".join(w.lower() for w in words) if words else "renamed_name"
    ident = re.sub(r"_+", "_", ident).strip("_") or "renamed_name"
    if ident[0].isdigit():
        ident = "var_" + ident
    if keyword.iskeyword(ident):
        ident += "_"
    return ident


def suggest_identifier(old_name: str, kind: str, translator: OfflineTranslator) -> str:
    translated = translate_plain_fragment(old_name, translator)
    words = english_words(translated)
    if kind == "class":
        return build_identifier(words, "class")
    return build_identifier(words, "variable")


def collect_docstring_occurrences(text: str, filename: str) -> Tuple[ast.AST, List[DocstringOccurrence]]:
    tree = ast.parse(text, filename=filename, type_comments=True)
    lines, line_starts = build_line_index(text)
    items: List[DocstringOccurrence] = []

    def capture(body: Sequence[ast.stmt]) -> None:
        if not body:
            return
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            offsets = ast_node_to_char_offsets(lines, line_starts, first.value)
            if offsets is not None:
                items.append(DocstringOccurrence(offsets[0], offsets[1], first.value.value))

    capture(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            capture(node.body)

    return tree, items


class ASTDefinitionCollector(ast.NodeVisitor):
    def __init__(self, lines: Sequence[str]):
        self.lines = lines
        self.items: List[IdentifierDefinition] = []

    def _add(self, node: ast.AST, name: str, kind: str) -> None:
        if not has_cjk(name):
            return
        pos = ast_node_to_char_lc(self.lines, node)
        if pos is None:
            return
        line, col = pos
        self.items.append(IdentifierDefinition(line, col, name, kind))

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._add(node, node.id, "variable")
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        self._add(node, node.arg, "parameter")
        self.generic_visit(node)


def collect_def_class_identifier_tokens(tokens: Sequence[tokenize.TokenInfo]) -> List[IdentifierDefinition]:
    out: List[IdentifierDefinition] = []
    for i, tok in enumerate(tokens):
        if tok.type == tokenize.NAME and tok.string in ("def", "class"):
            kind = "function" if tok.string == "def" else "class"
            j = i + 1
            while j < len(tokens) and tokens[j].type in {
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.COMMENT,
            }:
                j += 1
            if j < len(tokens):
                name_tok = tokens[j]
                if name_tok.type == tokenize.NAME and has_cjk(name_tok.string):
                    out.append(
                        IdentifierDefinition(
                            line=name_tok.start[0],
                            col=name_tok.start[1],
                            old_name=name_tok.string,
                            kind=kind,
                        )
                    )
    return out


def collect_python_identifier_definitions(
    tree: ast.AST,
    lines: Sequence[str],
    tokens: Sequence[tokenize.TokenInfo],
) -> List[IdentifierDefinition]:
    collector = ASTDefinitionCollector(lines)
    collector.visit(tree)

    items = list(collector.items)
    items.extend(collect_def_class_identifier_tokens(tokens))

    dedup: Dict[Tuple[int, int, str, str], IdentifierDefinition] = {}
    for item in items:
        dedup[(item.line, item.col, item.old_name, item.kind)] = item

    return sorted(dedup.values(), key=lambda x: (x.line, x.col, x.old_name, x.kind))


def is_span_inside_any(span: Tuple[int, int], containers: Sequence[Tuple[int, int]]) -> bool:
    s, e = span
    for cs, ce in containers:
        if cs <= s and e <= ce:
            return True
    return False


def get_call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = get_call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


class SafePythonStringCollector(ast.NodeVisitor):
    def __init__(
        self,
        lines: Sequence[str],
        line_starts: Sequence[int],
        doc_ranges: Sequence[Tuple[int, int]],
    ):
        self.lines = lines
        self.line_starts = line_starts
        self.doc_ranges = doc_ranges
        self.items: List[PythonStringOccurrence] = []

    def _add_const(self, node: ast.AST, reason: str) -> None:
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            return
        if not has_cjk(node.value):
            return
        offsets = ast_node_to_char_offsets(self.lines, self.line_starts, node)
        if offsets is None:
            return
        if is_span_inside_any(offsets, self.doc_ranges):
            return
        pos = ast_node_to_char_lc(self.lines, node)
        if pos is None:
            return
        self.items.append(
            PythonStringOccurrence(
                start=offsets[0],
                end=offsets[1],
                line=pos[0],
                col=pos[1],
                value=node.value,
                reason=reason,
            )
        )

    def visit_Call(self, node: ast.Call) -> None:
        full = get_call_name(node.func)
        tail = full.rsplit(".", 1)[-1] if full else ""

        if node.args:
            if tail in SAFE_PY_FIRST_ARG_CALL_TAILS or full.endswith(".add_argument"):
                self._add_const(node.args[0], f"call:{full or tail}:arg0")

        for kw in node.keywords:
            if kw.arg and kw.arg.lower() in SAFE_PY_STRING_KW_NAMES:
                self._add_const(kw.value, f"kw:{kw.arg}")

        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        if isinstance(node.exc, ast.Call):
            full = get_call_name(node.exc.func)
            tail = full.rsplit(".", 1)[-1] if full else ""
            if tail in SAFE_EXCEPTION_TAILS and node.exc.args:
                self._add_const(node.exc.args[0], f"raise:{tail}")
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        if node.msg is not None:
            self._add_const(node.msg, "assert:msg")
        self.generic_visit(node)


def collect_safe_python_string_occurrences(
    tree: ast.AST,
    lines: Sequence[str],
    line_starts: Sequence[int],
    doc_ranges: Sequence[Tuple[int, int]],
) -> List[PythonStringOccurrence]:
    collector = SafePythonStringCollector(lines, line_starts, doc_ranges)
    collector.visit(tree)

    dedup: Dict[Tuple[int, int, str], PythonStringOccurrence] = {}
    for item in collector.items:
        dedup[(item.start, item.end, item.reason)] = item
    return sorted(dedup.values(), key=lambda x: (x.start, x.end, x.reason))


def sniff_kind(path: Path) -> Optional[str]:
    suffix = path.suffix.lower()
    if suffix in PYTHON_EXTS:
        return "py"
    if suffix in YAML_EXTS:
        return "yaml"
    if suffix in SHELL_EXTS:
        return "sh"
    if suffix in TEXT_EXTS:
        return "text"
    if suffix == "":
        try:
            first = path.read_bytes().splitlines()[:1]
        except Exception:
            return None
        if first and first[0].startswith(b"#!"):
            line = first[0].decode("utf-8", "ignore").lower()
            if re.search(r"\bpython(?:3(?:\.\d+)?)?\b", line):
                return "py"
            if re.search(r"\b(?:bash|sh|dash|ksh|mksh)\b", line):
                return "sh"
    return None


def should_skip(rel_path: Path, top_ignore: set[str], anywhere_ignore: set[str]) -> bool:
    parts = rel_path.parts
    if not parts:
        return False
    if parts[0] in top_ignore:
        return True
    if any(part in anywhere_ignore for part in parts):
        return True
    return False


def iter_repo_files(root: Path, top_ignore: set[str], anywhere_ignore: set[str]) -> Iterable[Path]:
    root = root.resolve()
    for cur, dirs, files in os.walk(root):
        rel_cur = Path(cur).resolve().relative_to(root)
        dirs[:] = [d for d in dirs if not should_skip(rel_cur / d, top_ignore, anywhere_ignore)]
        for name in files:
            rel = rel_cur / name
            if should_skip(rel, top_ignore, anywhere_ignore):
                continue
            full = root / rel
            if full.is_symlink():
                continue
            yield full


def process_python_file(
    path: Path,
    repo_root: Path,
    translator: OfflineTranslator,
    apply: bool,
    identifier_occurrence_rows: List[List[object]],
    identifier_definition_rows: List[List[object]],
    python_string_rows: List[List[object]],
    python_string_manifest_items: List[dict],
    errors: List[str],
    translate_docstrings: bool,
    translate_python_safe_strings: bool,
) -> bool:
    rel = path.relative_to(repo_root)

    try:
        text, enc, newline = read_python_text(path)
    except Exception as e:
        errors.append(f"[PY_READ] {rel}: {e}")
        return False

    lines, line_starts = build_line_index(text)

    try:
        tree, doc_occurs = collect_docstring_occurrences(text, str(path))
    except Exception as e:
        errors.append(f"[PY_AST] {rel}: {e}")
        return False

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except Exception as e:
        errors.append(f"[PY_TOKENIZE] {rel}: {e}")
        return False

    doc_ranges = [(d.start, d.end) for d in doc_occurs]
    replacements: List[Replacement] = []

    for tok in tokens:
        if tok.type == tokenize.NAME and has_cjk(tok.string):
            identifier_occurrence_rows.append([str(rel), tok.start[0], tok.start[1], tok.string])

    definitions = collect_python_identifier_definitions(tree, lines, tokens)
    for item in definitions:
        suggestion = suggest_identifier(item.old_name, item.kind, translator)
        identifier_definition_rows.append(
            [str(rel), item.line, item.col, item.kind, item.old_name, suggestion]
        )

    safe_string_occurs = collect_safe_python_string_occurrences(tree, lines, line_starts, doc_ranges)
    safe_string_ranges = [(x.start, x.end) for x in safe_string_occurs]

    if translate_docstrings:
        for occ in doc_occurs:
            if not has_cjk(occ.value):
                continue
            old_token = text[occ.start : occ.end]
            new_value = translate_docstring_text(occ.value, translator)
            new_token = python_docstring_literal(new_value)
            if new_token != old_token:
                replacements.append(Replacement(occ.start, occ.end, new_token))

    if translate_python_safe_strings:
        for occ in safe_string_occurs:
            old_token = text[occ.start : occ.end]
            new_value = translate_multiline_text(occ.value, translator)
            new_token = quote_python_string(new_value)
            if new_token != old_token:
                replacements.append(Replacement(occ.start, occ.end, new_token))

    for tok in tokens:
        if tok.type != tokenize.STRING:
            continue
        try:
            start = lc_to_offset(line_starts, tok.start[0], tok.start[1])
            end = lc_to_offset(line_starts, tok.end[0], tok.end[1])
        except ValueError:
            continue

        span = (start, end)
        if is_span_inside_any(span, doc_ranges) or is_span_inside_any(span, safe_string_ranges):
            continue

        try:
            value = ast.literal_eval(tok.string)
        except Exception:
            continue

        if isinstance(value, str) and has_cjk(value):
            suggestion = translate_multiline_text(value, translator)
            python_string_rows.append(
                [str(rel), tok.start[0], tok.start[1], preview_text(value), preview_text(suggestion)]
            )
            python_string_manifest_items.append(
                {
                    "file": str(rel),
                    "line": tok.start[0],
                    "col": tok.start[1],
                    "old_text": value,
                    "new_text": suggestion,
                }
            )

    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        if tok.start[0] == 1 and tok.string.startswith("#!"):
            continue
        if tok.start[0] <= 2 and ENCODING_COOKIE_RE.search(tok.string):
            continue

        new_comment = translate_hash_comment(tok.string, translator)
        if new_comment == tok.string:
            continue

        try:
            start = lc_to_offset(line_starts, tok.start[0], tok.start[1])
            end = lc_to_offset(line_starts, tok.end[0], tok.end[1])
        except ValueError as e:
            errors.append(f"[PY_COMMENT_OFFSET] {rel}:{tok.start[0]}:{tok.start[1]}: {e}")
            continue

        replacements.append(Replacement(start, end, new_comment))

    try:
        new_text = apply_replacements(text, replacements)
    except Exception as e:
        errors.append(f"[PY_REPLACE] {rel}: {e}")
        return False

    changed = new_text != text
    if changed and apply:
        try:
            write_text_atomic(path, new_text, enc, newline)
        except Exception as e:
            errors.append(f"[PY_WRITE] {rel}: {e}")
            return False
    return changed


def process_text_file(
    path: Path,
    repo_root: Path,
    translator: OfflineTranslator,
    apply: bool,
    errors: List[str],
) -> bool:
    rel = path.relative_to(repo_root)
    try:
        text, enc, newline = read_utf8_text(path)
    except Exception as e:
        errors.append(f"[TEXT_READ] {rel}: {e}")
        return False

    out_parts: List[str] = []
    buffer_bodies: List[str] = []
    buffer_eols: List[str] = []
    changed = False
    inside_fence = False
    fence_marker: Optional[str] = None

    def flush_buffer() -> None:
        nonlocal changed
        if not buffer_bodies:
            return
        translated = translate_line_bodies(buffer_bodies, translator)
        for old_body, new_body, eol in zip(buffer_bodies, translated, buffer_eols):
            if new_body != old_body:
                changed = True
            out_parts.append(new_body + eol)
        buffer_bodies.clear()
        buffer_eols.clear()

    for line in text.splitlines(keepends=True):
        body, eol = split_line_ending(line)
        stripped = body.lstrip()

        if path.suffix.lower() == ".md" and re.match(r"^(```|~~~)", stripped):
            flush_buffer()
            marker = stripped[:3]
            if not inside_fence:
                inside_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                inside_fence = False
                fence_marker = None
            out_parts.append(body + eol)
            continue

        if inside_fence:
            flush_buffer()
            out_parts.append(body + eol)
            continue

        buffer_bodies.append(body)
        buffer_eols.append(eol)

    flush_buffer()

    new_text = "".join(out_parts)
    if not changed:
        changed = new_text != text

    if changed and apply:
        try:
            write_text_atomic(path, new_text, enc, newline)
        except Exception as e:
            errors.append(f"[TEXT_WRITE] {rel}: {e}")
            return False
    return changed


def find_shell_comment_index(line: str) -> Optional[int]:
    if re.match(r"^\s*#!", line):
        return None

    in_single = False
    in_double = False
    in_backtick = False
    escaped = False

    for i, ch in enumerate(line):
        if in_single:
            if ch == "'":
                in_single = False
            continue

        if in_double:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_double = False
            continue

        if in_backtick:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "`":
                in_backtick = False
            continue

        if ch == "'":
            in_single = True
        elif ch == '"':
            in_double = True
        elif ch == "`":
            in_backtick = True
        elif ch == "#":
            prev = line[i - 1] if i > 0 else ""
            if i == 0 or prev.isspace() or prev in ";|&()":
                return i

    return None


def maybe_start_heredoc(line_without_comment: str) -> Optional[Dict[str, object]]:
    m = re.search(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1", line_without_comment)
    if not m:
        return None
    return {"delimiter": m.group(2), "strip_tabs": "<<-" in m.group(0)}


def heredoc_should_end(line: str, state: Dict[str, object]) -> bool:
    candidate = line.lstrip("\t") if bool(state["strip_tabs"]) else line
    return candidate == state["delimiter"]


def process_shell_file(
    path: Path,
    repo_root: Path,
    translator: OfflineTranslator,
    apply: bool,
    errors: List[str],
    translate_inline_comments: bool,
) -> bool:
    rel = path.relative_to(repo_root)
    try:
        text, enc, newline = read_utf8_text(path)
    except Exception as e:
        errors.append(f"[SH_READ] {rel}: {e}")
        return False

    out: List[str] = []
    changed = False
    heredoc: Optional[Dict[str, object]] = None

    for line in text.splitlines(keepends=True):
        body, eol = split_line_ending(line)

        if heredoc is not None:
            out.append(body + eol)
            if heredoc_should_end(body, heredoc):
                heredoc = None
            continue

        idx = find_shell_comment_index(body)
        code_part = body if idx is None else body[:idx]
        new_body = body

        if idx is not None:
            comment = body[idx:]
            if translate_inline_comments or not code_part.strip():
                new_comment = translate_hash_comment(comment, translator)
                if new_comment != comment:
                    new_body = code_part + new_comment
                    changed = True

        heredoc = maybe_start_heredoc(code_part)
        out.append(new_body + eol)

    new_text = "".join(out)
    if changed and apply:
        try:
            write_text_atomic(path, new_text, enc, newline)
        except Exception as e:
            errors.append(f"[SH_WRITE] {rel}: {e}")
            return False

    return changed


def find_yaml_comment_index(line: str) -> Optional[int]:
    in_single = False
    in_double = False
    i = 0

    while i < len(line):
        ch = line[i]

        if in_single:
            if ch == "'" and i + 1 < len(line) and line[i + 1] == "'":
                i += 2
                continue
            if ch == "'":
                in_single = False
            i += 1
            continue

        if in_double:
            if ch == "\\" and i + 1 < len(line):
                i += 2
                continue
            if ch == '"':
                in_double = False
            i += 1
            continue

        if ch == "'":
            in_single = True
        elif ch == '"':
            in_double = True
        elif ch == "#":
            prev = line[i - 1] if i > 0 else ""
            if i == 0 or prev.isspace():
                return i

        i += 1

    return None


def is_yaml_block_scalar_header(code_part: str) -> bool:
    s = code_part.rstrip()
    return bool(re.search(r"(?:^|:\s*|-\s*)[>|][+-]?\d*$", s))


def translate_yaml_safe_scalars(node, translator: OfflineTranslator, context_key: Optional[str] = None) -> bool:
    changed = False
    try:
        from ruamel.yaml.scalarstring import ScalarString  # type: ignore
    except Exception:
        ScalarString = str  # type: ignore

    if isinstance(node, dict):
        for key in list(node.keys()):
            value = node[key]
            key_text = str(key)

            if isinstance(value, str) and key_text.lower() in SAFE_YAML_VALUE_KEYS and has_cjk(value):
                new_value = translate_multiline_text(value, translator)
                if new_value != value:
                    node[key] = type(value)(new_value) if isinstance(value, ScalarString) else new_value
                    changed = True
            else:
                changed = translate_yaml_safe_scalars(value, translator, key_text) or changed

    elif isinstance(node, list):
        for i, item in enumerate(node):
            if isinstance(item, str) and context_key and context_key.lower() in SAFE_YAML_VALUE_KEYS and has_cjk(item):
                new_value = translate_multiline_text(item, translator)
                if new_value != item:
                    node[i] = item.__class__(new_value) if item.__class__ is not str else new_value
                    changed = True
            else:
                changed = translate_yaml_safe_scalars(item, translator, context_key) or changed

    return changed


def translate_yaml_comments_raw(text: str, translator: OfflineTranslator) -> str:
    out: List[str] = []
    block_indent: Optional[int] = None

    for line in text.splitlines(keepends=True):
        body, eol = split_line_ending(line)
        stripped = body.lstrip(" ")
        indent = len(body) - len(stripped)

        if block_indent is not None:
            if stripped == "" or indent > block_indent:
                out.append(body + eol)
                continue
            block_indent = None

        idx = find_yaml_comment_index(body)
        code_part = body if idx is None else body[:idx]
        new_body = body

        if idx is not None:
            comment = body[idx:]
            new_comment = translate_hash_comment(comment, translator)
            if new_comment != comment:
                new_body = code_part + new_comment

        if is_yaml_block_scalar_header(code_part):
            block_indent = indent

        out.append(new_body + eol)

    return "".join(out)


def process_yaml_file(
    path: Path,
    repo_root: Path,
    translator: OfflineTranslator,
    apply: bool,
    errors: List[str],
    translate_safe_values: bool = True,
) -> bool:
    rel = path.relative_to(repo_root)
    try:
        text, enc, newline = read_utf8_text(path)
    except Exception as e:
        errors.append(f"[YAML_READ] {rel}: {e}")
        return False

    working = text

    if translate_safe_values:
        try:
            from ruamel.yaml import YAML  # type: ignore

            yaml = YAML(typ="rt")
            yaml.preserve_quotes = True
            data = yaml.load(working)
            if data is not None and translate_yaml_safe_scalars(data, translator):
                buf = io.StringIO()
                yaml.dump(data, buf)
                working = normalize_newlines(buf.getvalue())
        except Exception:
            pass

    working = translate_yaml_comments_raw(working, translator)
    changed = working != text

    if changed and apply:
        try:
            write_text_atomic(path, working, enc, newline)
        except Exception as e:
            errors.append(f"[YAML_WRITE] {rel}: {e}")
            return False

    return changed


def collect_name_token_positions(text: str, target_name: str) -> List[Tuple[int, int, int]]:
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except Exception:
        return []
    _lines, line_starts = build_line_index(text)

    out: List[Tuple[int, int, int]] = []
    for tok in tokens:
        if tok.type == tokenize.NAME and tok.string == target_name:
            try:
                offset = lc_to_offset(line_starts, tok.start[0], tok.start[1])
            except ValueError:
                continue
            out.append((tok.start[0], tok.start[1], offset))
    return out


def resolve_python_name_offset(text: str, line: int, col: int, old_name: str) -> Optional[int]:
    positions = collect_name_token_positions(text, old_name)
    if not positions:
        return None

    for ln, cl, off in positions:
        if ln == line and cl == col:
            return off

    same_line = [p for p in positions if p[0] == line]
    if same_line:
        same_line.sort(key=lambda p: abs(p[1] - col))
        return same_line[0][2]

    positions.sort(key=lambda p: (abs(p[0] - line), abs(p[1] - col)))
    return positions[0][2]


def collect_python_string_token_positions(
    text: str,
    old_text: str,
) -> List[Tuple[int, int, int, int, str]]:
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except Exception:
        return []
    _lines, line_starts = build_line_index(text)

    out: List[Tuple[int, int, int, int, str]] = []
    for tok in tokens:
        if tok.type != tokenize.STRING:
            continue
        try:
            value = ast.literal_eval(tok.string)
        except Exception:
            continue
        if not isinstance(value, str) or value != old_text:
            continue
        try:
            start = lc_to_offset(line_starts, tok.start[0], tok.start[1])
            end = lc_to_offset(line_starts, tok.end[0], tok.end[1])
        except ValueError:
            continue
        out.append((tok.start[0], tok.start[1], start, end, tok.string))
    return out


def resolve_python_string_span(
    text: str,
    line: int,
    col: int,
    old_text: str,
) -> Optional[Tuple[int, int, str]]:
    positions = collect_python_string_token_positions(text, old_text)
    if not positions:
        return None

    for ln, cl, start, end, token_text in positions:
        if ln == line and cl == col:
            return start, end, token_text

    same_line = [p for p in positions if p[0] == line]
    if same_line:
        same_line.sort(key=lambda p: abs(p[1] - col))
        ln, cl, start, end, token_text = same_line[0]
        return start, end, token_text

    positions.sort(key=lambda p: (abs(p[0] - line), abs(p[1] - col)))
    ln, cl, start, end, token_text = positions[0]
    return start, end, token_text


def token_text_is_fstring_or_bytes(token_text: str) -> bool:
    m = re.match(r"(?i)^([rubf]*)", token_text.strip())
    prefix = set(m.group(1).lower()) if m else set()
    return "f" in prefix or "b" in prefix


def apply_python_string_manifest(repo_root: Path, manifest_path: Path, errors: List[str]) -> List[str]:
    changed_files: List[str] = []

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("python_strings", [])
    if not isinstance(items, list) or not items:
        return changed_files

    grouped: Dict[str, List[dict]] = defaultdict(list)
    for raw in items:
        try:
            file_ = str(raw["file"])
            line = int(raw["line"])
            col = int(raw["col"])
            old_text = str(raw["old_text"])
            new_text = str(raw["new_text"])
        except Exception as e:
            errors.append(f"[PY_STRING_MANIFEST_ITEM] invalid item {raw!r}: {e}")
            continue
        grouped[file_].append(
            {
                "file": file_,
                "line": line,
                "col": col,
                "old_text": old_text,
                "new_text": new_text,
            }
        )

    for file_, entries in grouped.items():
        path = (repo_root / file_).resolve()
        if not path.exists():
            errors.append(f"[PY_STRING_MANIFEST] missing file: {file_}")
            continue

        try:
            text, enc, newline = read_python_text(path)
        except Exception as e:
            errors.append(f"[PY_STRING_MANIFEST_READ] {file_}: {e}")
            continue

        resolved: Dict[Tuple[int, int], Replacement] = {}

        for item in sorted(entries, key=lambda x: (-x["line"], -x["col"])):
            if item["old_text"] == item["new_text"]:
                continue

            found = resolve_python_string_span(
                text=text,
                line=item["line"],
                col=item["col"],
                old_text=item["old_text"],
            )
            if found is None:
                errors.append(
                    f"[PY_STRING_MANIFEST] cannot resolve string token for "
                    f"{file_}:{item['line']}:{item['col']} old_text={item['old_text']!r}"
                )
                continue

            start, end, token_text = found
            if token_text_is_fstring_or_bytes(token_text):
                errors.append(
                    f"[PY_STRING_MANIFEST] skip f-string/bytes at "
                    f"{file_}:{item['line']}:{item['col']}"
                )
                continue

            rep = Replacement(start, end, quote_python_string(item["new_text"]))
            key = (start, end)
            if key in resolved and resolved[key].new_text != rep.new_text:
                errors.append(
                    f"[PY_STRING_MANIFEST] conflicting replacements at "
                    f"{file_}:{item['line']}:{item['col']}"
                )
                continue
            resolved[key] = rep

        replacements = sorted(resolved.values(), key=lambda x: (x.start, x.end))
        if not replacements:
            continue

        try:
            new_text = apply_replacements(text, replacements)
        except Exception as e:
            errors.append(f"[PY_STRING_MANIFEST_REPLACE] {file_}: {e}")
            continue

        if new_text != text:
            try:
                write_text_atomic(path, new_text, enc, newline)
                changed_files.append(file_)
            except Exception as e:
                errors.append(f"[PY_STRING_MANIFEST_WRITE] {file_}: {e}")

    return changed_files


def apply_rope_manifest(repo_root: Path, manifest_path: Path, errors: List[str]) -> None:
    try:
        from rope.base.project import Project  # type: ignore
        from rope.base import libutils  # type: ignore
        from rope.refactor.rename import Rename  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing Rope. Install: pip install rope") from e

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("python_identifiers", [])
    if not isinstance(items, list) or not items:
        return

    dedup: Dict[Tuple[str, int, int, str, str], dict] = {}
    for raw in items:
        try:
            file_ = str(raw["file"])
            line = int(raw["line"])
            col = int(raw["col"])
            old_name = str(raw["old_name"])
            new_name = str(raw["new_name"])
        except Exception as e:
            errors.append(f"[ROPE_MANIFEST_ITEM] invalid item {raw!r}: {e}")
            continue
        dedup[(file_, line, col, old_name, new_name)] = {
            "file": file_,
            "line": line,
            "col": col,
            "old_name": old_name,
            "new_name": new_name,
        }

    items2 = list(dedup.values())
    items2.sort(key=lambda x: (x["file"], -x["line"], -x["col"]))

    project = Project(str(repo_root), ropefolder=None)
    try:
        for item in items2:
            file_path = (repo_root / item["file"]).resolve()
            old_name = item["old_name"]
            new_name = item["new_name"]

            if old_name == new_name:
                continue

            if not file_path.exists():
                errors.append(f"[ROPE] missing file: {item['file']}")
                continue

            if not new_name.isidentifier() or keyword.iskeyword(new_name):
                errors.append(
                    f"[ROPE] invalid new identifier {new_name!r} for "
                    f"{item['file']}:{item['line']}:{item['col']}"
                )
                continue

            try:
                text, _, _ = read_python_text(file_path)
                offset = resolve_python_name_offset(text, item["line"], item["col"], old_name)
                if offset is None:
                    errors.append(
                        f"[ROPE] cannot resolve current offset for "
                        f"{item['file']}:{item['line']}:{item['col']} name={old_name!r}"
                    )
                    continue

                resource = libutils.path_to_resource(project, str(file_path))
                if resource is None:
                    errors.append(f"[ROPE] resource not found: {item['file']}")
                    continue

                changes = Rename(project, resource, offset).get_changes(new_name, docs=True)
                project.do(changes)
            except Exception as e:
                errors.append(
                    f"[ROPE] {item['file']}:{item['line']}:{item['col']} "
                    f"{old_name!r}->{new_name!r}: {e}"
                )
    finally:
        project.close()


def read_text_best_effort(path: Path) -> Optional[str]:
    try:
        if path.stat().st_size > MAX_SCAN_FILE_SIZE:
            return None
    except Exception:
        return None

    kind = sniff_kind(path)
    try:
        if kind == "py":
            text, _, _ = read_python_text(path)
            return text
        text, _, _ = read_utf8_text(path)
        return text
    except Exception:
        return None


def scan_remaining_cjk(repo_root: Path, top_ignore: set[str], anywhere_ignore: set[str]) -> List[List[object]]:
    rows: List[List[object]] = []
    for path in iter_repo_files(repo_root, top_ignore, anywhere_ignore):
        text = read_text_best_effort(path)
        if text is None:
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            m = CJK_RE.search(line)
            if m:
                rows.append(
                    [
                        str(path.relative_to(repo_root)),
                        line_no,
                        m.start(),
                        preview_text(line),
                    ]
                )
    return rows


def scan_cjk_path_names(repo_root: Path, top_ignore: set[str], anywhere_ignore: set[str]) -> List[List[object]]:
    rows: List[List[object]] = []
    root = repo_root.resolve()

    if has_cjk(root.name):
        rows.append([".", root.name])

    for cur, dirs, files in os.walk(root):
        rel_cur = Path(cur).resolve().relative_to(root)
        dirs[:] = [d for d in dirs if not should_skip(rel_cur / d, top_ignore, anywhere_ignore)]

        for name in dirs:
            rel = rel_cur / name
            if has_cjk(name):
                rows.append([str(rel), name])

        for name in files:
            rel = rel_cur / name
            if should_skip(rel, top_ignore, anywhere_ignore):
                continue
            if has_cjk(name):
                rows.append([str(rel), name])

    return rows


def validate_python(repo_root: Path, top_ignore: set[str], anywhere_ignore: set[str]) -> List[str]:
    import py_compile

    errors: List[str] = []
    for path in iter_repo_files(repo_root, top_ignore, anywhere_ignore):
        if sniff_kind(path) != "py":
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as e:
            errors.append(f"[PY_COMPILE] {path.relative_to(repo_root)}: {e}")
    return errors


def validate_yaml(repo_root: Path, top_ignore: set[str], anywhere_ignore: set[str]) -> List[str]:
    errors: List[str] = []
    try:
        from ruamel.yaml import YAML  # type: ignore
    except Exception as e:
        return [f"[YAML_VALIDATION] ruamel.yaml not installed: {e}"]

    yaml = YAML(typ="rt")
    for path in iter_repo_files(repo_root, top_ignore, anywhere_ignore):
        if sniff_kind(path) != "yaml":
            continue
        try:
            text, _, _ = read_utf8_text(path)
            yaml.load(text)
        except Exception as e:
            errors.append(f"[YAML_PARSE] {path.relative_to(repo_root)}: {e}")
    return errors


def shell_check_command(path: Path) -> Optional[List[str]]:
    bash = shutil.which("bash")
    sh = shutil.which("sh")

    try:
        first = path.read_bytes().splitlines()[:1]
    except Exception:
        first = []

    shebang = first[0].decode("utf-8", "ignore").lower() if first else ""

    if path.suffix.lower() == ".bash":
        if bash:
            return [bash, "-n", str(path)]
        if sh:
            return [sh, "-n", str(path)]
        return None

    if "bash" in shebang:
        if bash:
            return [bash, "-n", str(path)]
        if sh:
            return [sh, "-n", str(path)]
        return None

    if re.search(r"\b(sh|dash|ksh|mksh)\b", shebang):
        if sh:
            return [sh, "-n", str(path)]
        if bash:
            return [bash, "-n", str(path)]
        return None

    if bash:
        return [bash, "-n", str(path)]
    if sh:
        return [sh, "-n", str(path)]
    return None


def validate_shell(repo_root: Path, top_ignore: set[str], anywhere_ignore: set[str]) -> List[str]:
    errors: List[str] = []
    for path in iter_repo_files(repo_root, top_ignore, anywhere_ignore):
        if sniff_kind(path) != "sh":
            continue
        cmd = shell_check_command(path)
        if cmd is None:
            continue
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            errors.append(f"[SH_PARSE] {path.relative_to(repo_root)}: {proc.stderr.strip()}")
    return errors


def write_tsv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safe-first zh->en repository translator")
    parser.add_argument("--repo", type=Path, required=True, help="Repository root")
    parser.add_argument("--model", type=Path, help="Local HF model directory (required unless --verify)")
    parser.add_argument("--apply", action="store_true", help="Write changes in place")
    parser.add_argument("--verify", action="store_true", help="Only run validation + residual scan")
    parser.add_argument("--rename-manifest", type=Path, help="Reviewed Rope rename manifest JSON")
    parser.add_argument("--python-string-manifest", type=Path, help="Reviewed risky Python string manifest JSON")
    parser.add_argument("--shell-inline-comments", action="store_true", help="Also translate inline shell comments")
    parser.add_argument("--no-docstrings", action="store_true", help="Do not translate Python docstrings")
    parser.add_argument("--no-python-safe-strings", action="store_true", help="Do not auto-translate safe Python user-facing strings")
    parser.add_argument("--no-yaml-safe-values", action="store_true", help="Do not translate YAML safe whitelist values")
    parser.add_argument("--term-map", type=Path, help="Optional JSON glossary: {ChineseTerm: EnglishTerm}")
    parser.add_argument("--report-dir", type=Path, default=None, help="Default: <repo>/.zh2en_work")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--src-lang", type=str, default=None, help="Optional multilingual model source language, e.g. zho_Hans")
    parser.add_argument("--tgt-lang", type=str, default=None, help="Optional multilingual model target language, e.g. eng_Latn")
    parser.add_argument("--top-ignore", action="append", default=[], help="Extra top-level directory to skip")
    parser.add_argument("--anywhere-ignore", action="append", default=[], help="Extra directory name to skip anywhere")
    args = parser.parse_args()

    if args.verify and args.apply:
        parser.error("--verify and --apply cannot be used together")
    if not args.verify and not args.model:
        parser.error("--model is required unless --verify is set")
    if args.rename_manifest and not args.apply:
        parser.error("--rename-manifest only makes sense together with --apply")
    if args.python_string_manifest and not args.apply:
        parser.error("--python-string-manifest only makes sense together with --apply")
    if args.batch_size < 1 or args.num_beams < 1 or args.max_source_length < 1 or args.max_new_tokens < 1:
        parser.error("--batch-size / --num-beams / --max-source-length / --max-new-tokens must all be positive integers")
    if not args.repo.exists() or not args.repo.is_dir():
        parser.error("--repo must be an existing directory")
    if args.model and not args.model.exists():
        parser.error("--model path does not exist")
    if args.term_map and not args.term_map.exists():
        parser.error("--term-map path does not exist")
    if args.rename_manifest and not args.rename_manifest.exists():
        parser.error("--rename-manifest path does not exist")
    if args.python_string_manifest and not args.python_string_manifest.exists():
        parser.error("--python-string-manifest path does not exist")

    return args


def main() -> int:
    args = parse_args()
    repo_root = args.repo.resolve()
    report_dir = args.report_dir.resolve() if args.report_dir else (repo_root / ".zh2en_work")

    top_ignore = set(TOP_LEVEL_IGNORE_DEFAULT) | set(args.top_ignore)
    anywhere_ignore = set(ANYWHERE_IGNORE_DEFAULT) | set(args.anywhere_ignore)

    try:
        rel_report = report_dir.relative_to(repo_root)
        if rel_report.parts:
            top_ignore.add(rel_report.parts[0])
    except ValueError:
        pass

    modified_files: List[str] = []
    identifier_occurrence_rows: List[List[object]] = []
    identifier_definition_rows: List[List[object]] = []
    python_string_rows: List[List[object]] = []
    python_string_manifest_items: List[dict] = []
    processing_errors: List[str] = []

    if not args.verify:
        try:
            term_map = load_term_map(args.term_map)
        except Exception as e:
            print(f"[error] failed to load --term-map: {e}", file=sys.stderr)
            return 2

        translator = OfflineTranslator(
            model_dir=args.model.resolve(),
            cache_path=report_dir / "translation_cache.json",
            term_map=term_map,
            batch_size=args.batch_size,
            num_beams=args.num_beams,
            max_source_length=args.max_source_length,
            max_new_tokens=args.max_new_tokens,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
        )

        for path in iter_repo_files(repo_root, top_ignore, anywhere_ignore):
            kind = sniff_kind(path)
            if kind is None:
                continue

            changed = False

            if kind == "py":
                changed = process_python_file(
                    path=path,
                    repo_root=repo_root,
                    translator=translator,
                    apply=args.apply,
                    identifier_occurrence_rows=identifier_occurrence_rows,
                    identifier_definition_rows=identifier_definition_rows,
                    python_string_rows=python_string_rows,
                    python_string_manifest_items=python_string_manifest_items,
                    errors=processing_errors,
                    translate_docstrings=not args.no_docstrings,
                    translate_python_safe_strings=not args.no_python_safe_strings,
                )
            elif kind == "yaml":
                changed = process_yaml_file(
                    path=path,
                    repo_root=repo_root,
                    translator=translator,
                    apply=args.apply,
                    errors=processing_errors,
                    translate_safe_values=not args.no_yaml_safe_values,
                )
            elif kind == "sh":
                changed = process_shell_file(
                    path=path,
                    repo_root=repo_root,
                    translator=translator,
                    apply=args.apply,
                    errors=processing_errors,
                    translate_inline_comments=args.shell_inline_comments,
                )
            elif kind == "text":
                changed = process_text_file(
                    path=path,
                    repo_root=repo_root,
                    translator=translator,
                    apply=args.apply,
                    errors=processing_errors,
                )

            if changed:
                modified_files.append(str(path.relative_to(repo_root)))

        translator.save_cache()

        if identifier_definition_rows:
            template = {
                "python_identifiers": [
                    {
                        "file": row[0],
                        "line": row[1],
                        "col": row[2],
                        "kind": row[3],
                        "old_name": row[4],
                        "new_name": row[5],
                    }
                    for row in identifier_definition_rows
                ]
            }
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "rename_manifest.template.json").write_text(
                json.dumps(template, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        if python_string_manifest_items:
            dedup: Dict[Tuple[str, int, int, str], dict] = {}
            for item in python_string_manifest_items:
                key = (item["file"], item["line"], item["col"], item["old_text"])
                dedup[key] = item
            template = {"python_strings": list(dedup.values())}
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "python_string_manifest.template.json").write_text(
                json.dumps(template, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        if args.apply and args.python_string_manifest:
            try:
                changed_from_manifest = apply_python_string_manifest(
                    repo_root=repo_root,
                    manifest_path=args.python_string_manifest.resolve(),
                    errors=processing_errors,
                )
                modified_files.extend(changed_from_manifest)
            except Exception as e:
                processing_errors.append(f"[PY_STRING_MANIFEST] {e}")

        if args.apply and args.rename_manifest:
            try:
                apply_rope_manifest(
                    repo_root=repo_root,
                    manifest_path=args.rename_manifest.resolve(),
                    errors=processing_errors,
                )
            except Exception as e:
                processing_errors.append(f"[ROPE] {e}")

    validation_errors: List[str] = []
    validation_errors.extend(processing_errors)
    validation_errors.extend(validate_python(repo_root, top_ignore, anywhere_ignore))
    validation_errors.extend(validate_yaml(repo_root, top_ignore, anywhere_ignore))
    validation_errors.extend(validate_shell(repo_root, top_ignore, anywhere_ignore))

    residual_rows = scan_remaining_cjk(repo_root, top_ignore, anywhere_ignore)
    path_rows = scan_cjk_path_names(repo_root, top_ignore, anywhere_ignore)

    modified_files = sorted(set(modified_files))
    report_dir.mkdir(parents=True, exist_ok=True)

    (report_dir / "modified_files.txt").write_text(
        "\n".join(modified_files) + ("\n" if modified_files else ""),
        encoding="utf-8",
    )

    write_tsv(
        report_dir / "python_identifier_occurrences.tsv",
        ["file", "line", "col(0-based)", "name"],
        identifier_occurrence_rows,
    )

    write_tsv(
        report_dir / "python_identifier_definitions.tsv",
        ["file", "line", "col(0-based)", "kind", "old_name", "suggested_new_name"],
        identifier_definition_rows,
    )

    write_tsv(
        report_dir / "python_string_literals.tsv",
        ["file", "line", "col(0-based)", "preview", "suggested_translation_preview"],
        python_string_rows,
    )

    write_tsv(
        report_dir / "remaining_cjk.tsv",
        ["file", "line", "col(0-based)", "snippet"],
        residual_rows,
    )

    write_tsv(
        report_dir / "cjk_path_names.tsv",
        ["relative_path", "name_with_cjk"],
        path_rows,
    )

    (report_dir / "validation_errors.txt").write_text(
        "\n".join(validation_errors) + ("\n" if validation_errors else ""),
        encoding="utf-8",
    )

    (report_dir / "summary.json").write_text(
        json.dumps(
            {
                "mode": "verify" if args.verify else ("apply" if args.apply else "audit"),
                "modified_files": len(modified_files),
                "python_identifier_occurrences": len(identifier_occurrence_rows),
                "python_identifier_definitions": len(identifier_definition_rows),
                "python_string_literals": len(python_string_rows),
                "validation_errors": len(validation_errors),
                "remaining_cjk": len(residual_rows),
                "cjk_path_names": len(path_rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[mode] {'verify' if args.verify else ('apply' if args.apply else 'audit')}")
    print(f"[report] {report_dir}")
    print(f"[modified/would-modify files] {len(modified_files)}")
    print(f"[python identifier occurrences] {len(identifier_occurrence_rows)}")
    print(f"[python identifier definitions] {len(identifier_definition_rows)}")
    print(f"[python string literals] {len(python_string_rows)}")
    print(f"[validation errors] {len(validation_errors)}")
    print(f"[remaining CJK] {len(residual_rows)}")
    print(f"[CJK path names] {len(path_rows)}")

    if not shutil.which("bash") and not shutil.which("sh"):
        print("[warning] neither bash nor sh found; shell syntax validation was skipped.", file=sys.stderr)

    if validation_errors:
        print("Validation errors found. See validation_errors.txt", file=sys.stderr)
    if residual_rows:
        print("Residual CJK found. See remaining_cjk.tsv", file=sys.stderr)
    if path_rows:
        print("CJK path names found. See cjk_path_names.tsv", file=sys.stderr)

    return 1 if validation_errors or residual_rows or path_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())


"""


"""