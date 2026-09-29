#!/usr/bin/env python3
"""Fetch the current manuals and external repositories declared by this skill."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import pypandoc

ROOT = Path(__file__).resolve().parents[1]
CATALOG_REL = Path("references/sources.json")
MANUAL_ROOT_REL = Path("references/manuals")
REPOSITORY_ROOT_REL = Path("references/repos")
RESOURCE_NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")
IGNORED_LINKED_PAGES = {"genindex.html", "search.html"}
BARE_PRE_RE = re.compile(r"(<pre\b[^>]*>)(?!\s*<code\b)(.*?)(</pre>)", re.IGNORECASE | re.DOTALL)


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.hrefs.append(href)


class _MainHtmlParser(HTMLParser):
    """Keep the article HTML from NVIDIA's current and legacy Sphinx themes."""

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=False)
        self._base_url = base_url
        self._root_tag: str | None = None
        self._root_depth = 0
        self._chunks: list[str] = []
        self.found_main = False

    @staticmethod
    def _is_content_root(tag: str, attrs: dict[str, str | None]) -> bool:
        classes = set((attrs.get("class") or "").split())
        return (tag == "article" and "bd-article" in classes) or (
            tag == "div" and attrs.get("role") == "main" and "document" in classes
        )

    def _starttag(self, tag: str, attrs: list[tuple[str, str | None]], *, closed: bool) -> str:
        rendered = []
        for key, value in attrs:
            if value is None:
                rendered.append(key)
                continue
            if key in {"href", "src"}:
                value = urljoin(self._base_url, value)
            rendered.append(f'{key}="{html.escape(value, quote=True)}"')
        suffix = " /" if closed else ""
        attributes = " " + " ".join(rendered) if rendered else ""
        return f"<{tag}{attributes}{suffix}>"

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if self._root_tag is None:
            if self.found_main or not self._is_content_root(tag, attributes):
                return
            self._root_tag = tag
            self._root_depth = 1
            self.found_main = True
        elif tag == self._root_tag:
            self._root_depth += 1
        self._chunks.append(self._starttag(tag, attrs, closed=False))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._root_tag is not None:
            self._chunks.append(self._starttag(tag, attrs, closed=True))

    def handle_endtag(self, tag: str) -> None:
        if self._root_tag is None:
            return
        self._chunks.append(f"</{tag}>")
        if tag == self._root_tag:
            self._root_depth -= 1
            if not self._root_depth:
                self._root_tag = None

    def handle_data(self, data: str) -> None:
        if self._root_tag is not None:
            self._chunks.append(data)

    def handle_entityref(self, name: str) -> None:
        if self._root_tag is not None:
            self._chunks.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self._root_tag is not None:
            self._chunks.append(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        if self._root_tag is not None:
            self._chunks.append(f"<!--{data}-->")

    def fragment(self) -> str:
        return "".join(self._chunks)


def _read_catalog(skill_root: Path) -> dict:
    catalog_path = skill_root / CATALOG_REL
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"manuals", "repositories"}:
        raise ValueError(f"{catalog_path}: expected manuals and repositories")
    for kind in ("manuals", "repositories"):
        entries = data[kind]
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"{catalog_path}: {kind} must be a non-empty object")
        for name, entry in entries.items():
            if not RESOURCE_NAME_RE.fullmatch(name) or not isinstance(entry, dict):
                raise ValueError(f"{catalog_path}: invalid {kind} entry {name!r}")
            if not isinstance(entry.get("title"), str) or not entry["title"].strip():
                raise ValueError(f"{catalog_path}: {kind}.{name}.title must be non-empty")
            url = entry.get("url")
            if not isinstance(url, str) or urlparse(url).scheme != "https":
                raise ValueError(f"{catalog_path}: {kind}.{name}.url must use https")
            if "linked_pages" in entry and not isinstance(entry["linked_pages"], bool):
                raise ValueError(f"{catalog_path}: {kind}.{name}.linked_pages must be boolean")
    return data


def _fetch_html(url: str) -> str:
    request = urllib.request.Request(
        url, headers={"Cache-Control": "no-cache", "User-Agent": "tirx-wiki-reference-fetch/1"}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def _linked_pages(root_url: str, root_html: str) -> list[str]:
    parser = _LinkParser()
    parser.feed(root_html)
    root = urlparse(root_url)
    root_prefix = root.path.rsplit("/", 1)[0] + "/"
    pages = [root_url]
    for href in parser.hrefs:
        candidate = urldefrag(urljoin(root_url, href)).url
        parsed = urlparse(candidate)
        if parsed.scheme != root.scheme or parsed.netloc != root.netloc:
            continue
        if not parsed.path.startswith(root_prefix) or not parsed.path.endswith(".html"):
            continue
        if Path(parsed.path).name in IGNORED_LINKED_PAGES:
            continue
        if candidate not in pages:
            pages.append(candidate)
    return pages


def _extract_main_html(url: str, page_html: str) -> str:
    parser = _MainHtmlParser(url)
    parser.feed(page_html)
    if not parser.found_main:
        raise RuntimeError(f"NVIDIA manual content root not found: {url}")
    return BARE_PRE_RE.sub(r"\1<code>\2</code>\3", parser.fragment())


def _convert_rst(title: str, pages: list[tuple[str, str]]) -> str:
    pandoc = pypandoc.get_pandoc_path()
    document = []
    for url, fragment in pages:
        escaped_url = html.escape(url, quote=True)
        opening_end = fragment.find(">") + 1
        if not opening_end:
            raise RuntimeError(f"manual article has no opening tag: {url}")
        source = f'<p>Source: <a href="{escaped_url}">{escaped_url}</a></p>'
        document.append(fragment[:opening_end] + source + fragment[opening_end:])
    result = subprocess.run(
        [pandoc, "--from=html", "--to=rst", "--wrap=none"],
        input="\n".join(document),
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"pandoc failed while converting {title}: {result.stderr.strip()}")
    return result.stdout


def _fetch_manual(skill_root: Path, name: str, entry: dict) -> Path:
    url = entry["url"]
    root_html = _fetch_html(url)
    pages = _linked_pages(url, root_html) if entry.get("linked_pages") else [url]
    documents = []
    for page in pages:
        page_html = root_html if page == url else _fetch_html(page)
        documents.append((page, _extract_main_html(page, page_html)))

    destination = skill_root / MANUAL_ROOT_REL / f"{name}.rst"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_convert_rst(entry["title"], documents), encoding="utf-8")
    return destination


def _clone_repository(skill_root: Path, name: str, entry: dict) -> Path:
    destination = skill_root / REPOSITORY_ROOT_REL / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=destination.parent) as raw_temp:
        checkout = Path(raw_temp) / "checkout"
        result = subprocess.run(
            ["git", "clone", "-q", "--depth", "1", "--no-tags", entry["url"], str(checkout)],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(f"failed to clone {entry['url']}: {result.stderr.strip()}")
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink(missing_ok=True)
        checkout.replace(destination)
    return destination


def materialize_references(skill_root: Path = ROOT, excluded: set[str] | None = None) -> list[Path]:
    """Refresh every declared live reference except explicitly excluded resources."""
    catalog = _read_catalog(skill_root)
    excluded_resources = excluded or set()
    declared = {
        *(f"manuals/{name}" for name in catalog["manuals"]),
        *(f"repositories/{name}" for name in catalog["repositories"]),
    }
    unknown = excluded_resources - declared
    if unknown:
        raise ValueError(f"unknown excluded resources: {', '.join(sorted(unknown))}")

    jobs = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for name, entry in catalog["manuals"].items():
            if f"manuals/{name}" not in excluded_resources:
                jobs.append(executor.submit(_fetch_manual, skill_root, name, entry))
        for name, entry in catalog["repositories"].items():
            if f"repositories/{name}" not in excluded_resources:
                jobs.append(executor.submit(_clone_repository, skill_root, name, entry))
        return [job.result() for job in jobs]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="KIND/NAME",
        help="skip one declared manuals/NAME or repositories/NAME resource",
    )
    arguments = parser.parse_args()
    for path in materialize_references(excluded=set(arguments.exclude)):
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
