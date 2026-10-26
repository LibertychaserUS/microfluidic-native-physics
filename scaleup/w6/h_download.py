#!/usr/bin/env python3
"""H_PROTOCOL_v1 §2 step 1-2: download a candidate dataset's files into an isolated
directory, record SHA-256 digests, and produce a header-only intake report.

The report lists file names, sizes, digests, zip members, workbook sheet names, header
rows and row counts. It never prints cell values below the header row and never
computes statistics of any column. Downloaded files are untrusted: nothing here is
executed or imported from them.

Usage: python h_download.py --code A1 --out <data dir> --manifest <json> [--url URL ...] [--skip-download]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import openpyxl

UA = "mf-reliability-audit/1.0 (+https://github.com/LibertychaserUS/microfluidic-native-physics)"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)


def describe_xlsx(data: bytes) -> list[dict]:
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    out = []
    for ws in wb.worksheets:
        rows = ws.iter_rows(values_only=True)
        header = None
        n = 0
        for row in rows:
            if header is None:
                if any(c is not None for c in row):
                    header = [str(c) if c is not None else "" for c in row]
                continue
            if any(c is not None for c in row):
                n += 1
        out.append({"sheet": ws.title, "header": header, "n_data_rows": n})
    return out


def describe_csv(data: bytes) -> dict:
    text = data.decode("utf-8", errors="replace").splitlines()
    header = text[0] if text else ""
    return {"header": header[:2000], "n_data_rows": sum(1 for line in text[1:] if line.strip())}


def describe_docx(data: bytes) -> dict:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml")
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    root = ET.fromstring(xml)
    tables = []
    for tbl in root.iter(f"{{{ns['w']}}}tbl"):
        trs = list(tbl.iter(f"{{{ns['w']}}}tr"))
        first = trs[0] if trs else None
        header = []
        if first is not None:
            for tc in first.iter(f"{{{ns['w']}}}tc"):
                header.append("".join(t.text or "" for t in tc.iter(f"{{{ns['w']}}}t")).strip())
        tables.append({"n_rows_including_header": len(trs), "header": header})
    return {"n_tables": len(tables), "tables": tables}


def describe_member(name: str, data: bytes) -> dict:
    lower = name.lower()
    try:
        if lower.endswith((".xlsx", ".xlsm")):
            return {"kind": "xlsx", "sheets": describe_xlsx(data)}
        if lower.endswith((".csv", ".txt", ".tsv", ".dat")):
            return {"kind": "text", **describe_csv(data)}
        if lower.endswith(".docx"):
            return {"kind": "docx", **describe_docx(data)}
    except Exception as exc:  # report, never crash the intake
        return {"kind": "unreadable", "error": str(exc)[:200]}
    return {"kind": Path(name).suffix.lower() or "none"}


def describe_file(path: Path) -> dict:
    data = path.read_bytes()
    entry = {"file": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    if path.suffix.lower() == ".zip":
        members = []
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                m = {"member": info.filename, "bytes": info.file_size}
                if re.search(r"\.(xlsx|xlsm|csv|txt|tsv|dat|docx)$", info.filename, re.I) and info.file_size < 50_000_000:
                    m.update(describe_member(info.filename, z.read(info)))
                members.append(m)
        entry["zip_members"] = members
    else:
        entry.update(describe_member(path.name, data))
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--url", action="append", default=[])
    parser.add_argument("--name", action="append", default=[], help="file name for each --url (same order)")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        if len(args.name) != len(args.url):
            raise SystemExit("--name must be given once per --url")
        for url, name in zip(args.url, args.name):
            dest = args.out / name
            if dest.exists():
                print("exists, not re-downloading:", name)
                continue
            download(url, dest)
            print("downloaded", name, dest.stat().st_size, "bytes")
    report = {"dataset_code": args.code, "directory": str(args.out), "sources": dict(zip(args.name, args.url)),
              "files": [describe_file(p) for p in sorted(args.out.iterdir()) if p.is_file()],
              "policy": "header-only intake; no cell values below header rows were read or printed"}
    args.manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for f in report["files"]:
        print(f"- {f['file']} {f['bytes']} B sha256={f['sha256'][:16]}… kind={f.get('kind','zip' if 'zip_members' in f else '')}")
        for s in f.get("sheets", []) or []:
            print(f"    sheet {s['sheet']!r}: {s['n_data_rows']} rows; header={s['header']}")
        for m in f.get("zip_members", []) or []:
            print(f"    member {m['member']} ({m['bytes']} B) kind={m.get('kind')}")
            for s in m.get("sheets", []) or []:
                print(f"        sheet {s['sheet']!r}: {s['n_data_rows']} rows; header={s['header']}")
            if m.get("kind") == "text":
                print(f"        header={m['header'][:300]} rows={m['n_data_rows']}")
        if f.get("kind") == "docx":
            for t in f["tables"]:
                print(f"    table rows={t['n_rows_including_header']} header={t['header']}")
        if f.get("kind") == "text":
            print(f"    header={f['header'][:300]} rows={f['n_data_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
