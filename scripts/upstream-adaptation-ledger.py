#!/usr/bin/env python3
"""Record and verify selective adaptations from newer DS4 history.

The ledger is deliberately repository-owned while the supporting evidence is
stored under DS4_RESEARCH_ROOT.  A donor commit is never a promotion source by
itself: an adapted entry must point at a new, independently tested commit on
the current research branch (or a successor branch from main).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = 1
DEFAULT_LEDGER = "docs/UPSTREAM-ADAPTATIONS.json"
DECISIONS = {"adapted", "rejected", "deferred", "already-present"}


class LedgerError(RuntimeError):
    pass


def run_git(repo: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            input=input_bytes,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode(errors="replace").strip()
        raise LedgerError(
            f"git {' '.join(args)} failed{': ' + detail if detail else ''}"
        ) from error


def resolve_commit(repo: Path, name: str) -> str:
    value = run_git(repo, "rev-parse", "--verify", f"{name}^{{commit}}").decode().strip()
    if len(value) != 40:
        raise LedgerError(f"could not resolve commit {name!r}")
    return value


def patch_bytes(repo: Path, commit: str, paths: list[str]) -> bytes:
    args = ["diff", "--binary", "--no-ext-diff", f"{commit}^", commit, "--"]
    args.extend(paths)
    return run_git(repo, *args)


def patch_id(repo: Path, commit: str, paths: list[str]) -> str:
    args = ["diff", "--no-ext-diff", "--unified=0", f"{commit}^", commit, "--"]
    args.extend(paths)
    diff = run_git(repo, *args)
    if not diff:
        return "empty"
    try:
        output = run_git(repo, "patch-id", "--stable", input_bytes=diff)
    except LedgerError:
        return "unavailable"
    fields = output.decode(errors="replace").split()
    return fields[0] if fields else "empty"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_ledger(path: Path) -> dict:
    if not path.exists():
        return {"schema": SCHEMA, "entries": []}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LedgerError(f"cannot read ledger {path}: {error}") from error
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise LedgerError(f"ledger {path} has unsupported schema")
    entries = value.get("entries")
    if not isinstance(entries, list):
        raise LedgerError(f"ledger {path} entries must be a list")
    return value


def write_ledger(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        stream.write(encoded)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def normalize_paths(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if not value or Path(value).is_absolute() or value.startswith("../"):
            raise LedgerError(f"ledger paths must be repository-relative: {value!r}")
        result.append(value)
    return sorted(set(result))


def cmd_record(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    ledger_path = (repo / args.ledger).resolve()
    if repo not in ledger_path.parents:
        raise LedgerError("ledger must be inside the repository")
    donor = resolve_commit(repo, args.donor)
    paths = normalize_paths(args.path)
    decision = args.decision
    if decision not in DECISIONS:
        raise LedgerError(f"decision must be one of {sorted(DECISIONS)}")
    if decision == "adapted" and not args.result_commit:
        raise LedgerError("adapted entries require --result-commit")
    if decision == "adapted" and not paths:
        raise LedgerError("adapted entries require at least one --path")
    if decision != "adapted" and not args.reason:
        raise LedgerError("non-adapted entries require --reason")

    donor_patch = patch_bytes(repo, donor, paths)
    entry = {
        "donor_commit": donor,
        "donor_commit_name": args.donor,
        "donor_patch_id": patch_id(repo, donor, paths),
        "donor_patch_sha256": sha256_bytes(donor_patch),
        "decision": decision,
        "paths": paths,
        "reason": args.reason or "",
        "evidence": sorted(set(args.evidence)),
        "source": {
            "title": args.source_title or "",
            "url": args.source_url or "",
            "access_date": args.source_access_date or "",
        },
        "research_branch": args.branch or run_git(repo, "branch", "--show-current").decode().strip(),
        "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    if args.result_commit:
        result = resolve_commit(repo, args.result_commit)
        result_patch = patch_bytes(repo, result, paths)
        entry["result_commit"] = result
        entry["result_patch_id"] = patch_id(repo, result, paths)
        entry["result_patch_sha256"] = sha256_bytes(result_patch)
    else:
        entry["result_commit"] = None
        entry["result_patch_id"] = None
        entry["result_patch_sha256"] = None

    ledger = load_ledger(ledger_path)
    for old in ledger["entries"]:
        if (old.get("donor_commit") == donor and
                old.get("decision") == decision and
                old.get("paths", []) == paths):
            raise LedgerError("matching donor/decision/paths entry already exists")
    ledger["entries"].append(entry)
    ledger["entries"].sort(key=lambda item: (
        str(item.get("donor_commit", "")),
        str(item.get("decision", "")),
        tuple(item.get("paths", [])),
    ))
    write_ledger(ledger_path, ledger)
    print(f"recorded {decision} donor {donor[:12]} in {ledger_path}")
    return 0


def evidence_path(repo: Path, research_root: Path | None, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = (repo / path).resolve()
    if candidate.exists() or research_root is None:
        return candidate
    return (research_root / path).resolve()


def cmd_verify(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    ledger_path = (repo / args.ledger).resolve()
    ledger = load_ledger(ledger_path)
    root_value = args.research_root or os.environ.get("DS4_RESEARCH_ROOT", "")
    research_root = Path(root_value).resolve() if root_value else None
    failures: list[str] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for index, entry in enumerate(ledger["entries"]):
        prefix = f"entry {index}"
        try:
            donor = resolve_commit(repo, str(entry["donor_commit"]))
            paths = normalize_paths(list(entry.get("paths", [])))
            key = (donor, str(entry.get("decision")), tuple(paths))
            if key in seen:
                raise LedgerError("duplicate donor/decision/paths")
            seen.add(key)
            actual_patch = patch_bytes(repo, donor, paths)
            if entry.get("donor_patch_sha256") != sha256_bytes(actual_patch):
                raise LedgerError("donor patch SHA-256 mismatch")
            actual_id = patch_id(repo, donor, paths)
            if entry.get("donor_patch_id") != actual_id:
                raise LedgerError("donor patch-id mismatch")
            decision = entry.get("decision")
            if decision not in DECISIONS:
                raise LedgerError("invalid decision")
            result_name = entry.get("result_commit")
            if decision == "adapted" and not result_name:
                raise LedgerError("adapted entry has no result commit")
            if result_name:
                result = resolve_commit(repo, str(result_name))
                result_patch = patch_bytes(repo, result, paths)
                if entry.get("result_patch_sha256") != sha256_bytes(result_patch):
                    raise LedgerError("result patch SHA-256 mismatch")
                if entry.get("result_patch_id") != patch_id(repo, result, paths):
                    raise LedgerError("result patch-id mismatch")
            if decision != "adapted" and not entry.get("reason"):
                raise LedgerError("non-adapted entry has no reason")
            for evidence in entry.get("evidence", []):
                if not evidence_path(repo, research_root, str(evidence)).exists():
                    raise LedgerError(f"missing evidence {evidence!r}")
        except (KeyError, TypeError, ValueError, LedgerError) as error:
            failures.append(f"{prefix}: {error}")
    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        return 1
    print(f"PASS upstream adaptation ledger ({len(ledger['entries'])} entries)")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--repo", default=".", help="DS4 repository (default: .)")
    root.add_argument("--ledger", default=DEFAULT_LEDGER,
                      help=f"ledger path relative to repo (default: {DEFAULT_LEDGER})")
    sub = root.add_subparsers(dest="command", required=True)

    record = sub.add_parser("record", help="append a donor decision")
    record.add_argument("--donor", required=True, help="donor commit or ref")
    record.add_argument("--decision", required=True, choices=sorted(DECISIONS))
    record.add_argument("--path", action="append", default=[],
                        help="adapted/compared repository path; repeatable")
    record.add_argument("--reason", default="")
    record.add_argument("--result-commit", default="")
    record.add_argument("--evidence", action="append", default=[],
                        help="repo-relative or absolute evidence path; repeatable")
    record.add_argument("--branch", default="")
    record.add_argument("--source-title", default="")
    record.add_argument("--source-url", default="")
    record.add_argument("--source-access-date", default="")
    record.set_defaults(func=cmd_record)

    verify = sub.add_parser("verify", help="recompute all recorded hashes")
    verify.add_argument("--research-root", default="",
                        help="canonical evidence root (default: DS4_RESEARCH_ROOT)")
    verify.set_defaults(func=cmd_verify)
    return root


def main() -> int:
    arguments = parser().parse_args()
    try:
        return arguments.func(arguments)
    except LedgerError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
