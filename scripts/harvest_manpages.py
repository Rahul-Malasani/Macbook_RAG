#!/usr/bin/env python3
"""
harvest_manpages.py
Builds the raw corpus by extracting man pages for a CURATED list of CLI tools.

Why curated (not `man -k .` / every installed page):
  - Reproducible & describable corpus ("~N common Unix CLI tools") instead of
    "whatever happened to be installed on this Mac" (machine-dependent).
  - Controlled variance: avoids giant/odd pages (bash, zsh) that would dominate
    the chunk distribution.
  - Deliberate confusable clusters (grep/rg/ack, scp/rsync/sftp, find/locate/fd)
    so retrieval has to disambiguate -> the ablation study has headroom.

Outputs:
  data/raw/man_<tool>.txt        one cleaned plaintext page per tool
  data/corpus_manifest.json      versioned manifest; corpus_version = content hash

Idempotent: re-running regenerates the files and recomputes the SAME corpus_version
when nothing changed. corpus_version is what every eval run gets tagged with, so
results are always anchored to a known corpus.
"""

import os
import json
import hashlib
import subprocess
import datetime

CANDIDATES_FILE = os.path.join(os.path.dirname(__file__), "manpage_candidates.txt")
RAW_DIR = "data/raw"
MANIFEST_PATH = "data/corpus_manifest.json"


def read_candidates(path):
    """Read tool names, stripping '#' comments and blank lines; dedupe in order."""
    tools, seen = [], set()
    with open(path) as f:
        for line in f:
            name = line.split("#", 1)[0].strip()
            if name and name not in seen:
                seen.add(name)
                tools.append(name)
    return tools


def fetch_manpage(tool):
    """Return cleaned plaintext for `man <tool>`, or None if there is no page."""
    # MANWIDTH fixes line wrapping so output is stable across machines/terminals.
    env = {**os.environ, "MANWIDTH": "80"}
    man = subprocess.run(
        ["man", tool], capture_output=True, text=True, errors="replace", env=env
    )
    if man.returncode != 0 or not man.stdout.strip():
        return None
    # `man` emits backspace-overstrike sequences for bold/underline when piped;
    # `col -bx` strips them and converts tabs to spaces -> clean plaintext.
    col = subprocess.run(
        ["col", "-bx"], input=man.stdout, capture_output=True, text=True, errors="replace"
    )
    text = (col.stdout if col.returncode == 0 else man.stdout).strip()
    return text + "\n" if text else None


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    candidates = read_candidates(CANDIDATES_FILE)
    print(f"[harvest] {len(candidates)} candidate tools")

    docs, missing = [], []
    for tool in candidates:
        text = fetch_manpage(tool)
        if text is None:
            missing.append(tool)
            continue
        fname = f"man_{tool}.txt"
        with open(os.path.join(RAW_DIR, fname), "w") as f:
            f.write(text)
        docs.append({
            "tool": tool,
            "filename": fname,
            "bytes": len(text.encode("utf-8")),
            "sha256": sha256(text),
        })

    docs.sort(key=lambda d: d["tool"])
    # corpus_version = hash over (tool, content-hash) pairs -> stable & content-addressed
    fingerprint = "\n".join(f'{d["tool"]}:{d["sha256"]}' for d in docs)
    corpus_version = sha256(fingerprint)[:12]

    manifest = {
        "corpus_version": corpus_version,
        "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "doc_count": len(docs),
        "total_bytes": sum(d["bytes"] for d in docs),
        "docs": docs,
        "missing": sorted(missing),
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    kb = manifest["total_bytes"] / 1024
    print(f"[harvest] wrote {len(docs)} pages ({kb:.0f} KB) to '{RAW_DIR}'")
    print(f"[harvest] missing ({len(missing)}): {', '.join(sorted(missing)) or 'none'}")
    print(f"[harvest] corpus_version = {corpus_version}")


if __name__ == "__main__":
    main()
